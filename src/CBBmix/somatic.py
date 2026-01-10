"""
Somatic 3-component Beta-Binomial mixture model with germline-informed priors.

This module fits a mixture model to somatic variants using MAP-EM,
where priors are derived from germline estimation on the same chromosome arm.

Components:
    0: Subclonal - low VAF variants (VAF ~ 0.05-0.25)
    1: Clonal - variants at expected tumor VAF (VAF ~ 0.3-0.6)
    2: LOH/Amplified - high VAF due to copy number changes (VAF ~ 0.6-0.99)

The germline provides:
    - κ_het → technical dispersion for all components
    - μ_het → baseline allelic balance (deviation indicates LOH)
    - LOH signal → adjusts component priors
"""

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from typing import Optional

from .utils import (
    beta_binom_logpmf,
    ab_from_mu_kappa,
    smart_init_somatic,
    SomaticMixtureSpec,
    SomaticPrior,
    SomaticFitResult,
    GermlineFitResult,
    build_somatic_prior_from_germline,
    build_default_somatic_prior,
)


class SomaticMixture:
    """
    3-component Beta-Binomial mixture model for somatic variants.

    Uses MAP-EM (Maximum A Posteriori Expectation-Maximization)
    with priors derived from germline estimation on the same arm.

    Components:
        0 (SUBCLONAL): Low VAF variants
        1 (CLONAL): Variants at expected tumor VAF
        2 (LOH): High VAF due to loss of heterozygosity

    Parameters
    ----------
    chrom : str
        Chromosome identifier.
    arm : str
        Chromosome arm ('p' or 'q').
    prior : SomaticPrior, optional
        Prior derived from germline. If None, uses defaults.
    spec : SomaticMixtureSpec, optional
        Model specification for bounds.

    Attributes
    ----------
    mu : np.ndarray
        Fitted mean VAF for each component.
    kappa : np.ndarray
        Fitted precision for each component.
    pi : np.ndarray
        Fitted mixing proportions.
    responsibilities : np.ndarray
        Posterior component probabilities (N x 3).
    """

    SUBCLONAL: int = 0
    CLONAL: int = 1
    LOH: int = 2
    COMPONENT_NAMES = ("subclonal", "clonal", "loh")

    def __init__(
        self,
        chrom: str,
        arm: str,
        prior: Optional[SomaticPrior] = None,
        spec: Optional[SomaticMixtureSpec] = None,
    ):
        self.chrom = chrom
        self.arm = arm
        self.spec = spec if spec is not None else SomaticMixtureSpec()

        # Set prior
        if prior is not None:
            self.prior = prior
        else:
            self.prior = build_default_somatic_prior(chrom, arm, self.spec)

        # Parameters (set after fitting)
        self.mu: Optional[np.ndarray] = None
        self.kappa: Optional[np.ndarray] = None
        self.pi: Optional[np.ndarray] = None
        self.responsibilities: Optional[np.ndarray] = None

        # Diagnostics
        self.loglik: float = -np.inf
        self.bic: float = np.inf
        self.converged: bool = False
        self.n_iterations: int = 0
        self.n_variants: int = 0

        # Store data
        self._alt: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None

    @classmethod
    def from_germline(
        cls,
        chrom: str,
        arm: str,
        germline_result: GermlineFitResult,
        tumor_purity: float = 1.0,
        spec: Optional[SomaticMixtureSpec] = None,
    ) -> "SomaticMixture":
        """
        Create SomaticMixture with prior derived from germline fit.

        This is the recommended constructor when germline data is available.

        Parameters
        ----------
        chrom : str
            Chromosome identifier.
        arm : str
            Chromosome arm.
        germline_result : GermlineFitResult
            Germline fit results.
        tumor_purity : float
            Estimated tumor purity (0, 1].
        spec : SomaticMixtureSpec, optional
            Model specification.

        Returns
        -------
        SomaticMixture
            Model with germline-informed prior.
        """
        prior = build_somatic_prior_from_germline(germline_result, tumor_purity, spec)
        return cls(chrom, arm, prior, spec)

    def fit(
        self,
        alt: np.ndarray,
        depth: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_restarts: int = 3,
        random_state: int = 42,
    ) -> "SomaticMixture":
        """
        Fit the 3-component mixture using MAP-EM.

        Parameters
        ----------
        alt : np.ndarray
            Alt allele read counts.
        depth : np.ndarray
            Total read depth.
        max_iter : int
            Maximum EM iterations per restart.
        tol : float
            Convergence tolerance for log-likelihood.
        n_restarts : int
            Number of random restarts.
        random_state : int
            Random seed.

        Returns
        -------
        SomaticMixture
            Self, for method chaining.
        """
        alt = np.asarray(alt, dtype=float)
        depth = np.asarray(depth, dtype=float)
        self._alt = alt
        self._depth = depth
        self.n_variants = len(alt)

        # Edge cases
        if self.n_variants == 0:
            self._set_empty_fit()
            return self

        if self.n_variants < 3:
            self._set_prior_fit()
            return self

        rng = np.random.default_rng(random_state)
        best_result = None

        for restart in range(n_restarts):
            result = self._fit_single_restart(alt, depth, max_iter, tol, rng, restart)

            if best_result is None or result["loglik"] > best_result["loglik"]:
                best_result = result

        # Store best
        self.mu = best_result["mu"]
        self.kappa = best_result["kappa"]
        self.pi = best_result["pi"]
        self.responsibilities = best_result["responsibilities"]
        self.loglik = best_result["loglik"]
        self.bic = best_result["bic"]
        self.converged = best_result["converged"]
        self.n_iterations = best_result["n_iterations"]

        return self

    def _fit_single_restart(
        self,
        alt: np.ndarray,
        depth: np.ndarray,
        max_iter: int,
        tol: float,
        rng: np.random.Generator,
        restart: int,
    ) -> dict:
        """Single MAP-EM optimization run."""
        N = len(alt)
        spec = self.spec
        prior = self.prior

        # Initialize
        if restart == 0:
            mu = smart_init_somatic(alt, depth, spec.bounds_mu)
            mu = 0.5 * mu + 0.5 * prior.mu_prior_mean
        else:
            mu = prior.mu_prior_mean + rng.uniform(-0.08, 0.08, 3)

        mu = np.clip(mu, spec.bounds_mu[:, 0], spec.bounds_mu[:, 1])
        kappa = prior.kappa_prior_mean.copy()
        pi = prior.pi_prior / prior.pi_prior.sum()

        if restart > 0:
            kappa = np.clip(
                kappa * rng.uniform(0.7, 1.3, 3),
                spec.bounds_kappa[0],
                spec.bounds_kappa[1],
            )

        prev_ll = -np.inf
        converged = False

        for iteration in range(max_iter):
            # =================================================================
            # E-step
            # =================================================================
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
                converged = True
                break
            prev_ll = ll

            # =================================================================
            # M-step (MAP)
            # =================================================================

            # Mixing proportions: Dirichlet posterior
            eff_counts = resps.sum(axis=0) + prior.pi_prior
            pi = eff_counts / eff_counts.sum()

            # Component parameters
            for k in range(3):
                # --- mu: Gaussian prior ---
                def neg_posterior_mu(m):
                    a, b = ab_from_mu_kappa(m, kappa[k])
                    ll_data = np.sum(resps[:, k] * beta_binom_logpmf(alt, depth, a, b))
                    ll_prior = (
                        -0.5
                        * ((m - prior.mu_prior_mean[k]) / prior.mu_prior_sigma[k]) ** 2
                    )
                    return -(ll_data + ll_prior)

                mu[k] = minimize_scalar(
                    neg_posterior_mu,
                    bounds=spec.bounds_mu[k],
                    method="bounded",
                ).x

                # --- kappa: log-normal prior ---
                def neg_posterior_kappa(kv):
                    a, b = ab_from_mu_kappa(mu[k], kv)
                    ll_data = np.sum(resps[:, k] * beta_binom_logpmf(alt, depth, a, b))
                    ll_prior = (
                        -0.5
                        * (
                            (np.log(kv) - np.log(prior.kappa_prior_mean[k]))
                            / prior.kappa_prior_log_sigma
                        )
                        ** 2
                    )
                    return -(ll_data + ll_prior)

                kappa[k] = minimize_scalar(
                    neg_posterior_kappa,
                    bounds=spec.bounds_kappa,
                    method="bounded",
                ).x

        # BIC
        n_params = 3 + 3 + 2  # mu + kappa + pi (simplex)
        bic = -2 * ll + n_params * np.log(N)

        return {
            "mu": mu,
            "kappa": kappa,
            "pi": pi,
            "responsibilities": resps,
            "loglik": ll,
            "bic": bic,
            "converged": converged,
            "n_iterations": iteration + 1,
        }

    def _set_empty_fit(self):
        """Set parameters for empty input."""
        self.mu = self.prior.mu_prior_mean.copy()
        self.kappa = self.prior.kappa_prior_mean.copy()
        self.pi = self.prior.pi_prior / self.prior.pi_prior.sum()
        self.responsibilities = np.array([]).reshape(0, 3)
        self.loglik = 0.0
        self.bic = 0.0
        self.converged = True
        self.n_iterations = 0

    def _set_prior_fit(self):
        """Set parameters for very small N."""
        self.mu = self.prior.mu_prior_mean.copy()
        self.kappa = self.prior.kappa_prior_mean.copy()
        self.pi = self.prior.pi_prior / self.prior.pi_prior.sum()

        if self._alt is not None and len(self._alt) > 0:
            self.responsibilities = self.predict_proba(self._alt, self._depth)
        else:
            self.responsibilities = np.array([]).reshape(0, 3)

        self.loglik = -np.inf
        self.bic = np.inf
        self.converged = True
        self.n_iterations = 0

    def get_result(self) -> SomaticFitResult:
        """
        Package results into SomaticFitResult.

        Returns
        -------
        SomaticFitResult
            Structured results.
        """
        if self.mu is None:
            raise ValueError("Must call fit() before get_result()")

        return SomaticFitResult(
            chrom=self.chrom,
            arm=self.arm,
            mu=self.mu.copy(),
            kappa=self.kappa.copy(),
            pi=self.pi.copy(),
            n_variants=self.n_variants,
            loglik=self.loglik,
            bic=self.bic,
            converged=self.converged,
            n_iterations=self.n_iterations,
            responsibilities=self.responsibilities.copy(),
            prior=self.prior,
        )

    def predict_proba(self, alt: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Compute component probabilities for data.

        Parameters
        ----------
        alt : np.ndarray
            Alt counts.
        depth : np.ndarray
            Total depth.

        Returns
        -------
        np.ndarray
            Responsibilities (N x 3).
        """
        if self.mu is None:
            raise ValueError("Must fit before prediction")

        alt = np.asarray(alt, dtype=float)
        depth = np.asarray(depth, dtype=float)
        N = len(alt)

        if N == 0:
            return np.array([]).reshape(0, 3)

        log_resps = np.zeros((N, 3))
        for k in range(3):
            a, b = ab_from_mu_kappa(self.mu[k], self.kappa[k])
            log_resps[:, k] = np.log(self.pi[k] + 1e-10) + beta_binom_logpmf(
                alt, depth, a, b
            )

        log_norm = logsumexp(log_resps, axis=1)
        return np.exp(log_resps - log_norm[:, None])

    def predict(
        self, alt: np.ndarray, depth: np.ndarray, threshold: float = 0.0
    ) -> np.ndarray:
        """
        Predict component labels.

        Parameters
        ----------
        alt : np.ndarray
            Alt counts.
        depth : np.ndarray
            Total depth.
        threshold : float
            Min responsibility for assignment (-1 if below).

        Returns
        -------
        np.ndarray
            Component indices (0, 1, 2) or -1.
        """
        proba = self.predict_proba(alt, depth)
        if len(proba) == 0:
            return np.array([], dtype=int)

        assignments = proba.argmax(axis=1).astype(int)
        if threshold > 0:
            assignments[proba.max(axis=1) < threshold] = -1

        return assignments

    def get_component_summary(self) -> dict:
        """
        Summary statistics per component.

        Returns
        -------
        dict
            Per-component statistics.
        """
        if self.responsibilities is None:
            raise ValueError("Must fit first")

        summary = {}
        for k, name in enumerate(self.COMPONENT_NAMES):
            mask = self.responsibilities.argmax(axis=1) == k
            n_assigned = mask.sum()
            mean_resp = (
                self.responsibilities[:, k].mean() if self.n_variants > 0 else 0
            )

            summary[name] = {
                "mu": self.mu[k],
                "kappa": self.kappa[k],
                "pi": self.pi[k],
                "n_assigned": int(n_assigned),
                "mean_responsibility": mean_resp,
            }

        return summary

    def __repr__(self) -> str:
        status = "fitted" if self.mu is not None else "not fitted"
        return (
            f"SomaticMixture(chrom={self.chrom!r}, arm={self.arm!r}, "
            f"n_variants={self.n_variants}, prior={self.prior.source}, "
            f"status={status})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def fit_somatic_mixture(
    chrom: str,
    arm: str,
    alt: np.ndarray,
    depth: np.ndarray,
    germline_result: Optional[GermlineFitResult] = None,
    tumor_purity: float = 1.0,
    spec: Optional[SomaticMixtureSpec] = None,
    **kwargs,
) -> SomaticFitResult:
    """
    Convenience function to fit somatic mixture.

    Parameters
    ----------
    chrom : str
        Chromosome.
    arm : str
        Chromosome arm.
    alt : np.ndarray
        Alt counts.
    depth : np.ndarray
        Total depth.
    germline_result : GermlineFitResult, optional
        Germline fit for prior construction.
    tumor_purity : float
        Tumor purity estimate.
    spec : SomaticMixtureSpec, optional
        Model specification.
    **kwargs
        Passed to fit().

    Returns
    -------
    SomaticFitResult
        Fit results.
    """
    if germline_result is not None:
        model = SomaticMixture.from_germline(
            chrom, arm, germline_result, tumor_purity, spec
        )
    else:
        model = SomaticMixture(chrom, arm, None, spec)

    model.fit(alt, depth, **kwargs)
    return model.get_result()


def fit_arm(
    chrom: str,
    arm: str,
    germline_het_alt: np.ndarray,
    germline_het_depth: np.ndarray,
    somatic_alt: np.ndarray,
    somatic_depth: np.ndarray,
    germline_hom_alt: np.ndarray | None = None,
    germline_hom_depth: np.ndarray | None = None,
    tumor_purity: float = 1.0,
    min_germline_het: int = 5,
) -> dict:
    """
    Complete pipeline: germline estimation → prior → somatic mixture.

    Parameters
    ----------
    chrom : str
        Chromosome.
    arm : str
        Chromosome arm.
    germline_het_alt : np.ndarray
        Germline het alt counts.
    germline_het_depth : np.ndarray
        Germline het depths.
    somatic_alt : np.ndarray
        Somatic alt counts.
    somatic_depth : np.ndarray
        Somatic depths.
    germline_hom_alt : np.ndarray, optional
        Germline hom ALT alt counts.
    germline_hom_depth : np.ndarray, optional
        Germline hom ALT depths.
    tumor_purity : float
        Tumor purity.
    min_germline_het : int
        Min het variants to use germline prior.

    Returns
    -------
    dict
        Contains 'germline', 'somatic', 'prior_source'.
    """
    from .germline import fit_germline

    # Stage 1: Germline estimation
    germline_result = None
    if len(germline_het_alt) >= min_germline_het:
        germline_result = fit_germline(
            chrom=chrom,
            arm=arm,
            het_alt=germline_het_alt,
            het_depth=germline_het_depth,
            hom_alt=germline_hom_alt,
            hom_depth=germline_hom_depth,
        )

    # Stage 2: Somatic mixture
    somatic_result = fit_somatic_mixture(
        chrom=chrom,
        arm=arm,
        alt=somatic_alt,
        depth=somatic_depth,
        germline_result=germline_result,
        tumor_purity=tumor_purity,
    )

    return {
        "germline": germline_result,
        "somatic": somatic_result,
        "prior_source": "germline" if germline_result is not None else "default",
    }