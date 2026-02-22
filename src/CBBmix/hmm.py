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
@jax.jit
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
@jax.jit
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

@jax.jit
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


@jax.jit
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

@jax.jit
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
class BaseHMM:
    """
    Base class for Beta-Binomial HMMs with shared priors, inference, and decoding.

    States ordered by increasing minor allele fraction:
        state 0 = strongest LOH, state K-1 = neutral (~0.5)
    """
    n_states: int = 3
    length_scale: float = 1_000_000.0  # transition distance decay (bp)
    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 2
    seed: int = 0
    mcmc_: Optional[MCMC] = field(default=None, repr=False)
    posterior_samples_: Optional[dict] = field(default=None, repr=False)

    def _sample_hmm_priors(self):
        """
        Sample shared HMM priors inside a NumPyro model context.

        Samples:
        - mu: ordered minor allele fractions via Dirichlet partitioning of [mu_base, 0.5]
        - kappa: per-state concentration (inverse overdispersion) ~ Gamma
        - A_base: transition matrix rows ~ Dirichlet (strong diagonal bias)
        - pi: initial state distribution ~ Dirichlet (neutral preference)

        Returns
        -------
        tuple (alpha, beta_p, log_A_base, log_pi)
            alpha, beta_p : (K,) Beta-Binomial shape parameters
            log_A_base    : (K, K) log base transition matrix
            log_pi        : (K,) log initial state distribution
        """
        K = self.n_states
        mu_raw = numpyro.sample("mu_raw", dist.Uniform(0.0, 0.5).expand([K]))
        mu = numpyro.deterministic("mu", jnp.sort(mu_raw))
        kappa = numpyro.sample(
            "kappa", dist.Gamma(2.0, 0.02).expand([K])
        )  # TODO: evaluate this prior.
        kappa = numpyro.deterministic("kappa_det", kappa)

        alpha = jnp.clip(mu * kappa, a_min=1e-4)
        beta_p = jnp.clip((1.0 - mu) * kappa, a_min=1e-4)

        # Base diagonal matrix: 100 on diagonal, 1 off-diagonal
        diag_c, off_c = 100.0, 1.0  # TODO: evaluate this prior, as it may be too strong.
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

        # Initial state: 5x preference on the neutral state (last one)
        pi_conc = jnp.ones(K).at[-1].set(5.0)
        pi = numpyro.sample("pi", dist.Dirichlet(pi_conc))
        log_pi = jnp.log(jnp.clip(pi, a_min=1e-30))

        return alpha, beta_p, log_A_base, log_pi

    def _get_posterior_mean_params(self):
        """
        Compute posterior-mean estimates of HMM parameters from stored posterior samples.

        Returns
        -------
        tuple (mu, kappa, A, pi)
            mu    : (K,)    posterior mean emission locations
            kappa : (K,)    posterior mean concentrations
            A     : (K, K)  posterior mean transition matrix
            pi    : (K,)    posterior mean initial state distribution
        """
        s = self.posterior_samples_
        mu = jnp.mean(s["mu"], axis=0)
        kappa = jnp.mean(s["kappa_det"], axis=0)
        A_rows = [jnp.mean(s[f"A_row_{i}"], axis=0)
                  for i in range(self.n_states)]
        return mu, kappa, jnp.stack(A_rows), jnp.mean(s["pi"], axis=0)

    def _get_emission_params(self):
        """
        Compute alpha, beta_p, log_A_base, log_pi from posterior means.

        Returns
        -------
        tuple (alpha, beta_p, log_A_base, log_pi)
        """
        mu, kappa, A_base, pi = self._get_posterior_mean_params()
        alpha = jnp.clip(mu * kappa, a_min=1e-4)
        beta_p = jnp.clip((1.0 - mu) * kappa, a_min=1e-4)
        log_A_base = jnp.log(jnp.clip(A_base, a_min=1e-30))
        log_pi = jnp.log(jnp.clip(pi, a_min=1e-30))
        return alpha, beta_p, log_A_base, log_pi

    def summary(self):
        if self.mcmc_ is None:
            raise RuntimeError("Call .fit() first")
        self.mcmc_.print_summary()

    def get_posterior_params(self):
        """Dict of posterior means: mu, kappa, A, pi."""
        mu, kappa, A, pi = self._get_posterior_mean_params()
        return {"mu": np.asarray(mu), "kappa": np.asarray(kappa),
                "A": np.asarray(A), "pi": np.asarray(pi)}


@dataclass
class BetaBinomialHMM(BaseHMM):
    """
    Beta-Binomial HMM for CNV detection from allele counts.
    Operates at SNP resolution: each site is one HMM time step.

    Input arrays (all methods):
        positions : (T,)  genomic bp, sorted
        depth     : (T,)  total read depth
        alt_depth : (T,)  alternate allele count
    """

    @staticmethod
    def _prepare_data(positions, depth, alt_depth) -> dict:
        """
        Fold BAF to minor allele counts [0, n/2] for phase-agnostic modeling.
        """
        pos = jnp.asarray(positions, dtype=jnp.float32)
        dep = jnp.asarray(depth, dtype=jnp.float32)
        alt = jnp.asarray(alt_depth, dtype=jnp.float32)
        minor = jnp.minimum(alt, dep - alt)
        dists = jnp.clip(jnp.diff(pos), a_min=1.0)
        return {"minor": minor, "depth": dep, "distances": dists}

    def _model(self, minor, depth, distances):
        """NumPyro generative model: SNP-level Beta-Binomial emissions."""
        alpha, beta_p, log_A_base, log_pi = self._sample_hmm_priors()

        log_emit = log_beta_binomial(minor, depth, alpha, beta_p)
        log_A = make_transition_matrices(
            log_A_base, distances, self.length_scale
        )
        numpyro.factor(
            "obs_log_lik",
            forward_log_likelihood(log_pi, log_A, log_emit)
        )

    def fit(self, positions, depth, alt_depth, **kwargs):
        """Run NUTS inference. Input arrays must be sorted by position."""
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

    def _build_log_params(self, data):
        alpha, beta_p, log_A_base, log_pi = self._get_emission_params()
        log_emit = log_beta_binomial(
            data["minor"], data["depth"], alpha, beta_p
        )
        log_A = make_transition_matrices(
            log_A_base, data["distances"], self.length_scale
        )
        return log_pi, log_A, log_emit

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


@dataclass
class GeneClusteredHMM(BaseHMM):
    """
    Beta-Binomial HMM where emissions are aggregated at the Gene level.

    SNP-level log-likelihoods are summed per gene via segment_sum, so the
    HMM trellis operates over genes (not SNPs). Transitions use inter-gene
    distances.

    Extra inputs (beyond positions/depth/alt_depth):
        gene_indices : (N_snps,) int mapping each SNP to gene 0..G-1
        gene_centers : (G,)     genomic position per gene (for transition distances)
    """

    @staticmethod
    def _prepare_data(positions, depth, alt_depth, gene_indices, gene_centers) -> dict:
        """
        Fold BAF to minor allele, compute inter-gene distances.

        Parameters
        ----------
        gene_indices : int array (N_snps,) mapping SNP to gene 0..G-1
        gene_centers : float array (G,) genomic position per gene
        """
        dep = jnp.asarray(depth, dtype=jnp.float32)
        alt = jnp.asarray(alt_depth, dtype=jnp.float32)
        minor = jnp.minimum(alt, dep - alt)

        g_idx = jnp.asarray(gene_indices, dtype=jnp.int32)
        g_pos = jnp.asarray(gene_centers, dtype=jnp.float32)
        dists = jnp.clip(jnp.diff(g_pos), a_min=1.0)

        return {
            "minor": minor,
            "depth": dep,
            "distances": dists,
            "gene_indices": g_idx,
            "n_genes": g_pos.shape[0],
        }

    def _model(self, minor, depth, distances, gene_indices, n_genes):
        """NumPyro generative model: gene-aggregated Beta-Binomial emissions."""
        alpha, beta_p, log_A_base, log_pi = self._sample_hmm_priors()

        # SNP-level log-likelihoods: (N_snps, K)
        log_emit_snps = log_beta_binomial(minor, depth, alpha, beta_p)

        # Aggregate to genes: log P(Gene | State) = sum_i log P(SNP_i | State)
        log_emit_genes = jax.ops.segment_sum(
            log_emit_snps, gene_indices, num_segments=n_genes
        )

        log_A = make_transition_matrices(
            log_A_base, distances, self.length_scale
        )
        numpyro.factor(
            "obs_log_lik",
            forward_log_likelihood(log_pi, log_A, log_emit_genes)
        )

    def fit(self, positions, depth, alt_depth, gene_indices, gene_centers, **kwargs):
        """Run NUTS inference with gene-level aggregation."""
        data = self._prepare_data(
            positions, depth, alt_depth, gene_indices, gene_centers
        )
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

    def _build_log_params(self, data):
        alpha, beta_p, log_A_base, log_pi = self._get_emission_params()

        log_emit_snps = log_beta_binomial(
            data["minor"], data["depth"], alpha, beta_p
        )
        log_emit_genes = jax.ops.segment_sum(
            log_emit_snps, data["gene_indices"], num_segments=data["n_genes"]
        )
        log_A = make_transition_matrices(
            log_A_base, data["distances"], self.length_scale
        )
        return log_pi, log_A, log_emit_genes

    def decode(self, positions, depth, alt_depth, gene_indices, gene_centers,
               method="viterbi"):
        """
        Decode per-gene states.
        Returns (N_genes,) int. Map back to SNPs via gene_states[gene_indices].
        """
        if self.posterior_samples_ is None:
            raise RuntimeError("Call .fit() first")
        data = self._prepare_data(
            positions, depth, alt_depth, gene_indices, gene_centers
        )
        log_pi, log_A, log_emit = self._build_log_params(data)
        if method == "viterbi":
            return np.asarray(viterbi_decode(log_pi, log_A, log_emit))
        elif method == "posterior":
            gamma = forward_backward(log_pi, log_A, log_emit)
            return np.asarray(jnp.argmax(gamma, axis=1))
        raise ValueError(f"Unknown method: {method}")

    def posterior_marginals(self, positions, depth, alt_depth, gene_indices,
                           gene_centers):
        """(N_genes, K) posterior p(z_g=k | data)."""
        if self.posterior_samples_ is None:
            raise RuntimeError("Call .fit() first")
        data = self._prepare_data(
            positions, depth, alt_depth, gene_indices, gene_centers
        )
        log_pi, log_A, log_emit = self._build_log_params(data)
        return np.asarray(forward_backward(log_pi, log_A, log_emit))