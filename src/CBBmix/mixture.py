import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from .utils import beta_binom_logpmf, smart_init_parameters, ab_from_mu_kappa
from .params import BetaBinomialMixtureSpec


def fit_robust_3_component_smart(
    alt,
    depth,
    spec: BetaBinomialMixtureSpec | None = None,
    max_iter=500,
    tol=1e-6,
    n_restarts=5,
    random_state=42,
):
    if spec is None:
        spec = BetaBinomialMixtureSpec()
    alt = np.asarray(alt, float)
    depth = np.asarray(depth, float)
    N = alt.size
    rng = np.random.default_rng(random_state)

    # optimization through EM
    best_fit = None
    for r in range(n_restarts):
        mu = smart_init_parameters(alt, depth, spec.bounds_mu)
        if r > 0:
            mu = np.clip(
                mu + rng.uniform(-0.05, 0.05, 3),
                spec.bounds_mu[:, 0],
                spec.bounds_mu[:, 1],
            )
        kappa = spec.kappa_init.copy()
        pi = spec.pi_init.copy()
        prev_ll = -np.inf
        for _ in range(max_iter):
            # --- E-step ---
            log_resps = np.zeros((N, 3))
            for k in range(3):
                a, b = ab_from_mu_kappa(mu[k], kappa[k])
                log_resps[:, k] = np.log(pi[k] + 1e-10) + beta_binom_logpmf(
                    alt, depth, a, b
                )

            log_norm = logsumexp(log_resps, axis=1)
            resps = np.exp(log_resps - log_norm[:, None])
            ll = log_norm.sum()

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

            # --- M-step ---
            eff = resps.sum(axis=0) + spec.weight_prior
            pi = eff / eff.sum()

            for k in range(3):

                def nll_mu(m):
                    a, b = ab_from_mu_kappa(m, kappa[k])
                    llk = np.sum(resps[:, k] * beta_binom_logpmf(alt, depth, a, b))
                    lp = -0.5 * ((m - spec.mu_prior_mean) / spec.mu_prior_sigma) ** 2
                    return -(llk + lp)

                mu[k] = minimize_scalar(
                    nll_mu,
                    bounds=spec.bounds_mu[k],
                    method="bounded",
                ).x

                def nll_kappa(kv):
                    a, b = ab_from_mu_kappa(mu[k], kv)
                    return -np.sum(resps[:, k] * beta_binom_logpmf(alt, depth, a, b))

                kappa[k] = minimize_scalar(
                    nll_kappa,
                    bounds=spec.bounds_kappa,
                    method="bounded",
                ).x

        # --- BIC ---
        k_params = 3 + 3 + 2  # mu + kappa + pi (simplex)
        bic = -2 * ll + k_params * np.log(N)

        result = {
            "loglik": ll,
            "bic": bic,
            "mu": mu,
            "kappa": kappa,
            "pi": pi,
            "responsibilities": resps,
        }

        if best_fit is None or ll > best_fit["loglik"]:
            best_fit = result

    return best_fit
