"""
Beta-Binomial HMM for CNV detection from B-allele frequency signal.

Manual forward algorithm (log-space) with JAX for likelihood,
NumPyro NUTS for posterior inference on continuous parameters.
Viterbi and forward-backward for decoding.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import betaln, gammaln
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# emission log-likelihood: Beta-Binomial pmf in log-space, vectorized over (T, K)
def log_beta_binomial(k, n, alpha, beta_param):
    """
    Log-pmf of BetaBinomial(n, alpha, beta) at k. Returns (T, K).

    Uses: log P(k|n,a,b) = log C(n,k) + betaln(k+a, n-k+b) - betaln(a,b)
    """
    k_ = k[:, None]
    n_ = n[:, None]
    a_ = alpha[None, :]
    b_ = beta_param[None, :]
    log_comb = gammaln(n_ + 1) - gammaln(k_ + 1) - gammaln(n_ - k_ + 1)
    return log_comb + betaln(k_ + a_, n_ - k_ + b_) - betaln(a_, b_)

# transition matrices with distance-dependent interpolation between identity and base matrix
# using exponential decay: rho_i = exp(-d_i / length_scale)
def make_transition_matrices(log_A_base, distances, length_scale):
    """
    Construct per-interval transition matrices (in log-space) that interpolate
    between the identity matrix and a provided base transition matrix according
    to inter-interval distances.

    Each effective transition matrix is computed as:
        A_eff[i] = rho_i * I + (1 - rho_i) * A_base
    with
        rho_i = exp(-distances[i] / length_scale).

    When two sites are close (distance ≪ length_scale), ρ → 1 and A_eff → I: almost no state change allowed.
    When two sites are far (distance ≫ length_scale), ρ → 0 and A_eff → A_base: the full base transition matrix applies.
    
    NOTE: the length_scale is fixed and not inferred. Must be evaluated.
    Parameters
    ----------
    log_A_base : array-like, shape (K, K)
        Logarithm of the base transition matrix (typically log-probabilities).
        The base matrix used is A_base = exp(log_A_base). K is the number of states.
    distances : array-like, shape (T-1,)
        Non-negative distances between successive observations/positions. One
        effective transition matrix is produced per distance.
    length_scale : float
        Positive length scale controlling how quickly rho decays with distance.
        Larger values cause slower decay (more influence from the identity).

    Returns
    -------
    jnp.ndarray, shape (T-1, K, K)
        Logarithm of the effective transition matrices for each interval.
        Values are clipped to avoid numerical -inf before taking the log.
    """
    K = log_A_base.shape[0]
    A_base = jnp.exp(log_A_base)
    rho = jnp.exp(-distances / length_scale)[:, None, None]
    A_eff = rho * jnp.eye(K)[None, :, :] + (1.0 - rho) * A_base[None, :, :]
    return jnp.log(jnp.clip(A_eff, a_min=1e-30)) # clipping to avoid log(0) = -inf issues


def forward_log_likelihood(log_pi, log_A, log_emit):
    """
    Compute the log marginal likelihood of an observation sequence under a discrete-state HMM
    using the forward algorithm in log-space.

    This function implements a numerically stable, JAX-friendly forward pass:
    - log_pi: initial state log-probabilities at time 0
    - log_A: time-indexed transition log-probabilities between consecutive time steps
    - log_emit: emission log-probabilities for each time and state

    Args:
        log_pi (jnp.ndarray): 1-D array of shape (N,) containing the log-probabilities
            of the N hidden states at time 0 (log p(z_0 = i)).
        log_A (jnp.ndarray): 3-D array of shape (T-1, N, N) where log_A[t, i, j] is the
            log-probability of transitioning from state i at time t to state j at time t+1
            (log p(z_{t+1}=j | z_t=i)). The leading dimension length should be one less
            than the number of time steps in log_emit.
        log_emit (jnp.ndarray): 2-D array of shape (T, N) where log_emit[t, i] is the
            log-probability of observing the data at time t given state i
            (log p(x_t | z_t = i)). The first time index (t=0) is used together with
            log_pi to initialize the forward messages.

    Returns:
        jnp.ndarray: A scalar (0-D array) containing the log marginal likelihood
        log p(x_{0:T-1}) computed as logsumexp of the final forward messages.

    Raises:
        ValueError: If input shapes are incompatible (for example, if log_emit.shape[0]
        != log_A.shape[0] + 1, or if the state dimension N does not match across inputs).

    Notes:
        - All computations are performed in log-space and use jax.nn.logsumexp for
          numerical stability.
        - The implementation is JAX-compatible (uses lax.scan) and thus supports JIT
          compilation and autodifferentiation.
    """
    log_alpha_0 = log_pi + log_emit[0]
    def step(log_alpha_prev, t):
        log_alpha_next = jax.nn.logsumexp(
            log_alpha_prev[:, None] + log_A[t], axis=0
        ) + log_emit[t + 1]
        return log_alpha_next, None
    log_alpha_T, _ = lax.scan(step, log_alpha_0, jnp.arange(log_A.shape[0]))
    return jax.nn.logsumexp(log_alpha_T)



def viterbi_decode(log_pi, log_A, log_emit):
    """
    Perform Viterbi decoding (most likely state sequence() for an
    HMM given log initial probabilities, time-varying log transition matrices,
    and log emission scores. It uses JAX's lax.scan to implement the forward
    (max/argmax) recursion and a backward trace to recover the optimal path.

    Parameters
    ----------
    log_pi : jnp.ndarray, shape (N,)
        Log of the initial state probabilities for N hidden states.
    log_A : jnp.ndarray, shape (T-1, N, N)
        Time-indexed log transition matrices. For each time t in 0..T-2,
        log_A[t] should be an (N, N) array where log_A[t][i, j] is the log
        probability of transitioning from state i at time t to state j at time t+1.
    log_emit : jnp.ndarray, shape (T, N)
        Log emission scores for each time step and state. log_emit[t, j] is the
        log-likelihood (or log-score) of observing the data at time t given state j.

    Returns
    -------
    jnp.ndarray, shape (T,)
        Integer array of length T containing the most likely state indices
        (in 0..N-1) at each time step: [s_0, s_1, ..., s_{T-1}].

    Notes
    -----
    - The implementation assumes T = log_emit.shape[0] >= 1 and that
      log_A has length T-1 (i.e., one transition matrix per step between
      observations).
    - The returned path corresponds to the sequence that maximizes the sum
      of log probabilities: log_pi[s_0] + sum_t (log_A[t-1][s_{t-1}, s_t] + log_emit[t, s_t]).
    """
    T = log_emit.shape[0]
    delta_0 = log_pi + log_emit[0]
    def fwd(delta_prev, t):
        scores = delta_prev[:, None] + log_A[t]
        return (jnp.max(scores, axis=0) + log_emit[t + 1],
                jnp.argmax(scores, axis=0))
    delta_T, bp = lax.scan(fwd, delta_0, jnp.arange(T - 1))
    last = jnp.argmax(delta_T)

    def bwd(state, t):
        prev = bp[t][state]
        return prev, prev

    _, traced = lax.scan(bwd, last, jnp.arange(T - 2, -1, -1))
    return jnp.concatenate([jnp.flip(traced), last[None]])


def forward_backward(log_pi, log_A, log_emit):
    """
    Compute posterior marginals p(z_t = k | y_{1:T}) for a discrete-state HMM
    using the numerically-stable forward–backward algorithm implemented in log-space.

    Parameters
    ----------
    log_pi : array-like, shape (K,)
        Logarithm of the initial state distribution:
        log_pi[k] = log p(z_1 = k).

    log_A : array-like, shape (T-1, K, K)
        Time-dependent log transition matrices. For each time t in 0..T-2,
        log_A[t][i, j] = log p(z_{t+2} = j | z_{t+1} = i).
        Rows correspond to source states (i), columns to target states (j).
        If transitions are time-homogeneous, supply the same (K, K) matrix
        repeated along the time axis to form shape (T-1, K, K).

    log_emit : array-like, shape (T, K)
        Log emission probabilities:
        log_emit[t, k] = log p(y_{t+1} | z_{t+1} = k).
        Here T is the number of time steps.

    Returns
    -------
    posteriors : jnp.ndarray, shape (T, K)
        Posterior marginal probabilities gamma[t, k] = p(z_{t+1} = k | y_{1:T}).
        Each row sums to 1 (within numerical precision) and values are in [0, 1].

    Notes
    -----
    - The implementation performs all intermediate computations in log-space and
      exponentiates only the final, normalized log-marginals to avoid underflow..
    - Requires T >= 1 and dimensions (T, K) / (T-1, K, K) to be consistent; mismatched
      shapes will raise indexing or broadcasting errors.
    """
    T = log_emit.shape[0]
    K = log_pi.shape[0]
    log_a0 = log_pi + log_emit[0]

    def fwd(prev, t):
        nxt = jax.nn.logsumexp(
            prev[:, None] + log_A[t], axis=0
        ) + log_emit[t + 1]
        return nxt, nxt

    _, alphas_tail = lax.scan(fwd, log_a0, jnp.arange(T - 1))
    log_alphas = jnp.concatenate([log_a0[None], alphas_tail], axis=0)

    log_bT = jnp.zeros(K)

    def bwd(nxt, t):
        prev = jax.nn.logsumexp(
            log_A[t] + log_emit[t + 1][None, :] + nxt[None, :], axis=1
        )
        return prev, prev

    _, betas_tail = lax.scan(bwd, log_bT, jnp.arange(T - 2, -1, -1))
    log_betas = jnp.concatenate(
        [jnp.flip(betas_tail, axis=0), log_bT[None]], axis=0
    )

    log_gamma = log_alphas + log_betas
    log_gamma -= jax.nn.logsumexp(log_gamma, axis=1, keepdims=True)
    return jnp.exp(log_gamma)



@dataclass
class BetaBinomialHMM:
    """
    Beta-Binomial HMM for CNV detection from allele counts.
    It forces the 

    States ordered by increasing minor allele fraction:
        state 0 = strongest LOH, state K-1 = neutral (~0.5)

    Input arrays (all methods):
        positions : (T,)  genomic bp, sorted
        depth     : (T,)  total read depth
        alt_depth : (T,)  alternate allele count
    """
    n_states: int = 3
    length_scale: float = 1_000_000.0  # transition distance decay (bp)
    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 2
    seed: int = 0
    mcmc_: Optional[MCMC] = field(default=None, repr=False)
    posterior_samples_: Optional[dict] = field(default=None, repr=False)


    @staticmethod
    def _prepare_data(positions, depth, alt_depth) -> dict:
        """
        That's the core preprocessing of the model. Given that we don't know the phase of the variants,
        we could just observe deviations from the expected true diploidity that is assumed to be centered
        towards 0.5. Doing the folding, we're modeling every site to be comprised in the range of [0, n/2]
        instead of just [0,n], as the BAF will be pushed towards 0 or 1 according to the phase. In this way, 
        the model is symmetric.
        """
        pos = jnp.asarray(positions, dtype=jnp.float32)
        dep = jnp.asarray(depth, dtype=jnp.float32)
        alt = jnp.asarray(alt_depth, dtype=jnp.float32)
        minor = jnp.minimum(alt, dep - alt)
        dists = jnp.clip(jnp.diff(pos), a_min=1.0)
        return {"minor": minor, "depth": dep, "distances": dists}


    def _model(self, minor: jnp.ndarray, depth: jnp.ndarray, distances: jnp.ndarray):
        """
        Core definition of the HMM model using Beta-Binomial emissions and distance-dependent transitions.
        The concept here is based on the reparametrization for each state k:
        - mu_k: expected minor allele fraction for state k, ordered via a Dirichlet partitioning of [mu_base, 0.5]
        - kappa_k: concentration (inverse overdispersion) for state k, sampled from a Gamma distribution
        - alpha_k = mu_k * kappa_k
        - beta_k = (1 - mu_k) * kappa_k 

        - 
        Generative model with:
        - Ordered mu via Dirichlet partitioning of [mu_base, 0.5]
        - Per-state concentration kappa ~ Gamma
        - Distance-dependent transitions with Dirichlet(strong diagonal) base
        - Initial state Dirichlet with neutral preference
        """
        assert minor.ndim == 1 and depth.ndim == 1 and distances.ndim == 1, "Input arrays must be 1D"
        K = self.n_states

        # Means must satisfy 0 < mu_0 < mu_1 < ... < mu_{K-1} < 0.5. 
        # sample mu_base_raw < 0.5 
        mu_base_raw = numpyro.sample("mu_base_raw", dist.Beta(2.0, 10.0))
        mu_base = mu_base_raw * 0.45  #scale to [0, 0.45] to leave room for the Dirichlet partitioning up to 0.5. That's just for safety.
        if K > 1:
            # sample raw increments then scale to fill the gap between mu_base and 0.5
            raw_inc = numpyro.sample("mu_raw_inc", dist.Dirichlet(jnp.ones(K)))
            remaining = 0.5 - mu_base
            cum = mu_base + jnp.cumsum(raw_inc) * remaining
            # K values: mu_base, then K-2 interior, last ~ 0.5
            #NOTE: we're dropping 0.5 because we're assuming that the reference bias will always push
            # the observed BAF below 0.5. 
            mu = jnp.concatenate([mu_base[None], cum[:-1]])  
        else:
            # fallback for K=1: just use mu_base as the single state's mean
            mu = mu_base[None]
        mu = numpyro.deterministic("mu", mu)
        # beta binomial overdispersion, to control the spread of the emission distributions around the means.
        # Higher kappa -> less overdispersion, more concentrated around mu.
        kappa = numpyro.sample(
            "kappa", dist.Gamma(2.0, 0.02).expand([K])
        ) #TODO: evaluate this prior.
        kappa = numpyro.deterministic("kappa_det", kappa)
        # clip to prevent numerical issues in the Beta-Binomial likelihood when alpha or beta are too small. 
        # NOTE: here the clipping is just for the lower bound, but could be evaluated if an upper bound is needed as well

        alpha = jnp.clip(mu * kappa, a_min=1e-4)
        beta_p = jnp.clip((1.0 - mu) * kappa, a_min=1e-4)

        # Base diagonal matrix construction.
        # 100 on the diagonal, 1 off-diagonal, to encourage self-transitions.
        diag_c, off_c = 100.0, 1.0 #TODO: evaluate this prior, as it may be too strong.
        A_conc = jnp.full((K, K), off_c).at[
            jnp.diag_indices(K)
        ].set(diag_c)
        A_rows = []
        for i in range(K):
            A_rows.append(
                numpyro.sample(f"A_row_{i}", dist.Dirichlet(A_conc[i]))
            )
        A_base = jnp.stack(A_rows)
        log_A_base = jnp.log(jnp.clip(A_base, a_min=1e-30))

        # Initial state: 5x preference on the neutral state (last one).
        pi_conc = jnp.ones(K).at[-1].set(5.0)
        pi = numpyro.sample("pi", dist.Dirichlet(pi_conc))
        log_pi = jnp.log(jnp.clip(pi, a_min=1e-30))

        # This trick is to avoid enumerating all the states, so with the fw 
        # we could compute the marginal likelihood over all the hidden state sequences.
        log_emit = log_beta_binomial(minor, depth, alpha, beta_p)
        log_A = make_transition_matrices(
            log_A_base, distances, self.length_scale
        )
        numpyro.factor(
            "obs_log_lik",
            forward_log_likelihood(log_pi, log_A, log_emit)
        )

    
    def fit(self, positions, depth, alt_depth, **kwargs):
        """
        Core NUTS inference method. The input arrays must be sorted by position.
        """
        data = self._prepare_data(positions, depth, alt_depth)
        kernel = NUTS(
            self._model, max_tree_depth=10, target_accept_prob=0.8, **kwargs
        )
        self.mcmc_ = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=True,
        )
        self.mcmc_.run(jax.random.PRNGKey(self.seed), **data)
        self.posterior_samples_ = self.mcmc_.get_samples()
        return self


    def _get_posterior_mean_params(self):
        """
        Compute posterior-mean estimates of HMM parameters from stored posterior samples.

        This method reads posterior samples from self.posterior_samples_ and returns the
        posterior mean of the model parameters. The means are computed by averaging
        over the sample axis (axis=0) for each stored parameter.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            A 4-tuple containing:
            - mu: Posterior mean of the emission/location parameters. This has the same
              shape as a single sample of self.posterior_samples_["mu"] (i.e. the sample
              axis removed).
            - kappa: Posterior mean of the concentration/deterministic kappa
              parameters. Shape matches a single sample of self.posterior_samples_["kappa_det"].
            - A: Posterior mean state transition matrix, constructed by averaging each
              stored row "A_row_{i}" across samples and stacking them in order. Shape
              is (n_states, n_states).
            - pi: Posterior mean of the initial state distribution. Shape matches a
              single sample of self.posterior_samples_["pi"] (typically (n_states,)).
        """
        s = self.posterior_samples_
        # just the mean of posterior samples for each parameter. Note that this is not the best
        # given that we're computing a full posterior, but a single point estimation 
        # may keep things simpler for the decoding.
        mu = jnp.mean(s["mu"], axis=0)
        kappa = jnp.mean(s["kappa_det"], axis=0)
        A_rows = [jnp.mean(s[f"A_row_{i}"], axis=0)
                  for i in range(self.n_states)]
        return mu, kappa, jnp.stack(A_rows), jnp.mean(s["pi"], axis=0)

    def _build_log_params(self, data):
        mu, kappa, A_base, pi = self._get_posterior_mean_params()
        alpha = jnp.clip(mu * kappa, a_min=1e-4)
        beta_p = jnp.clip((1.0 - mu) * kappa, a_min=1e-4)
        log_emit = log_beta_binomial(
            data["minor"], data["depth"], alpha, beta_p
        )
        log_A = make_transition_matrices(
            jnp.log(jnp.clip(A_base, a_min=1e-30)),
            data["distances"], self.length_scale
        )
        log_pi = jnp.log(jnp.clip(pi, a_min=1e-30))
        return log_pi, log_A, log_emit

    # -- decode --------------------------------------------------------------

    def decode(self, positions, depth, alt_depth, method="viterbi"):
        """
        method='viterbi' (MAP) or 'posterior' (marginal argmax).
        Returns (T,) int: 0=strong LOH ... K-1=neutral.
        """
        if self.posterior_samples_ is None:
            raise RuntimeError("Call .fit() first")
        data = self._prepare_data(positions, depth, alt_depth)
        log_pi, log_A, log_emit = self._build_log_params(data)
        if method == "viterbi":
            return np.asarray(viterbi_decode(log_pi, log_A, log_emit))
        elif method == "posterior":
            gamma = forward_backward(log_pi, log_A, log_emit)
            return np.asarray(jnp.argmax(gamma, axis=1))
        raise ValueError(f"Unknown method: {method}")

    def posterior_marginals(self, positions, depth, alt_depth):
        """(T, K) posterior p(z_t=k | data)."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Call .fit() first")
        data = self._prepare_data(positions, depth, alt_depth)
        log_pi, log_A, log_emit = self._build_log_params(data)
        return np.asarray(forward_backward(log_pi, log_A, log_emit))

    def summary(self):
        if self.mcmc_ is None:
            raise RuntimeError("Call .fit() first")
        self.mcmc_.print_summary()

    def get_posterior_params(self):
        """Dict of posterior means: mu, kappa, A, pi."""
        mu, kappa, A, pi = self._get_posterior_mean_params()
        return {"mu": np.asarray(mu), "kappa": np.asarray(kappa),
                "A": np.asarray(A), "pi": np.asarray(pi)}