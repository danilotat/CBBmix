from dataclasses import dataclass, field
import numpy as np
from scipy.special import gammaln, betaln
from scipy.optimize import minimize


def beta_binom_logpmf(
    k: np.ndarray, n: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """
    Compute log probability mass function of Beta-Binomial distribution.

    The Beta-Binomial PMF is:
        P(k | n, α, β) = C(n, k) × B(k + α, n - k + β) / B(α, β)

    In log-space (for numerical stability):
        log P = log C(n, k) + log B(k + α, n - k + β) - log B(α, β)

    Parameters
    ----------
    k : np.ndarray
        Number of successes (alt read counts).
    n : np.ndarray
        Number of trials (total depth).
    alpha : float
        First shape parameter of the Beta prior (α > 0).
    beta : float
        Second shape parameter of the Beta prior (β > 0).

    Returns
    -------
    np.ndarray
        Log-probabilities for each observation.
    """
    alpha = np.maximum(alpha, 1e-9)
    beta = np.maximum(beta, 1e-9)

    return (
        gammaln(n + 1.0)
        - gammaln(k + 1.0)
        - gammaln(n - k + 1.0)
        + betaln(k + alpha, n - k + beta)
        - betaln(alpha, beta)
    )


def ab_from_mu_kappa(mu: float, kappa: float) -> tuple[float, float]:
    """
    Convert mean-precision parameterization to standard Beta shape parameters.

    Relationships:
        α = μ × κ
        β = (1 - μ) × κ
        Variance = μ(1-μ) / (κ + 1)

    Parameters
    ----------
    mu : float
        Mean of the Beta distribution, in (0, 1).
    kappa : float
        Concentration/precision parameter (κ > 0).

    Returns
    -------
    tuple[float, float]
        (alpha, beta) shape parameters.
    """
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)
    return mu * kappa, (1.0 - mu) * kappa


def fit_beta_binomial_mle(
    alt: np.ndarray,
    depth: np.ndarray,
    mu_bounds: tuple[float, float] = (0.01, 0.99),
    kappa_bounds: tuple[float, float] = (1.0, 1000.0),
) -> tuple[float, float, float]:
    """
    Fit a single Beta-Binomial distribution via MLE.

    Finds (μ, κ) that maximize:
        Σᵢ log P(altᵢ | depthᵢ, μ, κ)

    Parameters
    ----------
    alt : np.ndarray
        Alt allele counts.
    depth : np.ndarray
        Total depth.
    mu_bounds : tuple[float, float]
        Bounds for mean parameter.
    kappa_bounds : tuple[float, float]
        Bounds for concentration parameter.

    Returns
    -------
    tuple[float, float, float]
        (mu, kappa, loglik) - MLE estimates and log-likelihood.
    """
    alt = np.asarray(alt, dtype=float)
    depth = np.asarray(depth, dtype=float)

    if len(alt) == 0:
        return 0.5, 50.0, 0.0

    # Initialize from method of moments
    vaf = alt / (depth + 1e-9)
    mu_init = np.clip(np.mean(vaf), mu_bounds[0] + 0.01, mu_bounds[1] - 0.01)

    # Estimate kappa from variance
    var_vaf = np.var(vaf)
    if var_vaf > 0 and var_vaf < mu_init * (1 - mu_init):
        kappa_init = mu_init * (1 - mu_init) / var_vaf - 1
        kappa_init = np.clip(kappa_init, kappa_bounds[0], kappa_bounds[1])
    else:
        kappa_init = 50.0

    def neg_loglik(params):
        mu, log_kappa = params
        kappa = np.exp(log_kappa)
        a, b = ab_from_mu_kappa(mu, kappa)
        ll = np.sum(beta_binom_logpmf(alt, depth, a, b))
        return -ll

    result = minimize(
        neg_loglik,
        x0=[mu_init, np.log(kappa_init)],
        method="L-BFGS-B",
        bounds=[mu_bounds, (np.log(kappa_bounds[0]), np.log(kappa_bounds[1]))],
    )

    mu_mle = result.x[0]
    kappa_mle = np.exp(result.x[1])
    loglik = -result.fun

    return mu_mle, kappa_mle, loglik


@dataclass
class GermlineSpec:
    """
    Specification for germline parameter estimation.

    Since zygosity is known, no mixture model is needed.
    This defines bounds and thresholds.
    """

    # Expected mu for heterozygous (0.5 for balanced diploid)
    mu_het_expected: float = 0.5

    # Bounds for mu estimation
    mu_het_bounds: tuple[float, float] = (0.10, 0.90)
    mu_hom_bounds: tuple[float, float] = (0.85, 0.9999)

    # Bounds for kappa
    kappa_bounds: tuple[float, float] = (5.0, 1000.0)

    # LOH detection threshold (|mu_het - 0.5|)
    loh_imbalance_threshold: float = 0.10

    # Minimum variants for reliable estimation
    min_variants_het: int = 5
    min_variants_hom: int = 3


@dataclass
class SomaticMixtureSpec:
    """
    Specification for 3-component somatic mixture model.

    Components:
        0: Subclonal (VAF ~ 0.05-0.25)
        1: Clonal (VAF ~ 0.3-0.6)
        2: LOH/Amplified (VAF ~ 0.6-0.99)
    """

    # Bounds for mu: [subclonal, clonal, loh]
    bounds_mu: np.ndarray = field(
        default_factory=lambda: np.array([[0.02, 0.30], [0.25, 0.75], [0.50, 0.99]])
    )

    # Bounds for kappa
    bounds_kappa: tuple[float, float] = (5.0, 500.0)

    # Default values (used when no germline prior available)
    mu_default: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.50, 0.85]))
    kappa_default: np.ndarray = field(
        default_factory=lambda: np.array([50.0, 50.0, 50.0])
    )
    pi_default: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.50, 0.25]))

    # Default Dirichlet prior on mixing proportions
    pi_prior_default: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 4.0, 2.0])
    )

    # Default Gaussian priors on mu
    mu_prior_sigma_default: np.ndarray = field(
        default_factory=lambda: np.array([0.08, 0.12, 0.12])
    )

    # Default prior on kappa (log-normal scale)
    kappa_prior_log_sigma_default: float = 0.5

    def __post_init__(self):
        self.bounds_mu = np.asarray(self.bounds_mu)
        self.mu_default = np.asarray(self.mu_default)
        self.kappa_default = np.asarray(self.kappa_default)
        self.pi_default = np.asarray(self.pi_default)
        self.pi_prior_default = np.asarray(self.pi_prior_default)
        self.mu_prior_sigma_default = np.asarray(self.mu_prior_sigma_default)


@dataclass
class GermlineFitResult:
    """
    Results from fitting germline variants (with known zygosity).

    Since zygosity labels are known, we fit each class via direct MLE
    rather than a mixture model.
    """

    # Chromosome arm identifier
    chrom: str
    arm: str

    # Heterozygous class parameters (MLE)
    mu_het: float
    kappa_het: float
    n_het: int
    loglik_het: float

    # Homozygous ALT class parameters (MLE)
    mu_hom: float
    kappa_hom: float
    n_hom: int
    loglik_hom: float

    @property
    def n_variants(self) -> int:
        """Total number of germline variants."""
        return self.n_het + self.n_hom

    @property
    def loglik(self) -> float:
        """Total log-likelihood."""
        return self.loglik_het + self.loglik_hom

    @property
    def allelic_imbalance(self) -> float:
        """Absolute deviation of het mean from 0.5."""
        return abs(self.mu_het - 0.5)

    @property
    def has_loh_signal(self) -> bool:
        """
        Heuristic LOH detection based on het VAF distribution.

        Returns True if sufficient het variants and strong deviation from 0.5.
        """
        return self.n_het >= 10 and self.allelic_imbalance > 0.10

    @property
    def confidence(self) -> float:
        """
        Confidence weight based on number of het variants.
        Saturates at ~30 variants.
        """
        return np.clip(self.n_het / 30.0, 0.1, 1.0)

    @property
    def het_direction(self) -> int:
        """
        Direction of allelic imbalance.

        Returns:
            +1 if mu_het > 0.5 (ALT allele favored)
            -1 if mu_het < 0.5 (REF allele favored)
             0 if balanced
        """
        if self.allelic_imbalance < 0.05:
            return 0
        return 1 if self.mu_het > 0.5 else -1


@dataclass
class SomaticPrior:
    """
    Prior for somatic mixture model, derived from germline fit.

    Encapsulates all information transferred from germline to somatic
    via Empirical Bayes.
    """

    chrom: str
    arm: str

    # Priors on component means [subclonal, clonal, loh]
    mu_prior_mean: np.ndarray
    mu_prior_sigma: np.ndarray

    # Priors on dispersion
    kappa_prior_mean: np.ndarray
    kappa_prior_log_sigma: float

    # Dirichlet prior on mixing proportions
    pi_prior: np.ndarray

    # Metadata
    loh_detected: bool
    confidence: float
    source: str = "germline"  # "germline" or "default"

    def __post_init__(self):
        self.mu_prior_mean = np.asarray(self.mu_prior_mean)
        self.mu_prior_sigma = np.asarray(self.mu_prior_sigma)
        self.kappa_prior_mean = np.asarray(self.kappa_prior_mean)
        self.pi_prior = np.asarray(self.pi_prior)


@dataclass
class SomaticFitResult:
    """
    Results from fitting somatic 3-component mixture.
    """

    chrom: str
    arm: str

    # Component parameters [subclonal, clonal, loh]
    mu: np.ndarray
    kappa: np.ndarray
    pi: np.ndarray

    # Fit diagnostics
    n_variants: int
    loglik: float
    bic: float
    converged: bool
    n_iterations: int

    # Responsibilities (N x 3 array)
    responsibilities: np.ndarray = field(repr=False)

    # Prior used
    prior: SomaticPrior = field(repr=False)

    # Component indices
    SUBCLONAL: int = 0
    CLONAL: int = 1
    LOH: int = 2
    COMPONENT_NAMES: tuple = ("subclonal", "clonal", "loh")

    def get_assignments(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get hard component assignments.

        Parameters
        ----------
        threshold : float
            Minimum responsibility to assign (otherwise -1).

        Returns
        -------
        np.ndarray
            Component index (0, 1, 2) or -1 if uncertain.
        """
        if self.n_variants == 0:
            return np.array([], dtype=int)

        max_resp = self.responsibilities.max(axis=1)
        assignments = self.responsibilities.argmax(axis=1).astype(int)
        assignments[max_resp < threshold] = -1
        return assignments

    def get_component_indices(self, component: int) -> np.ndarray:
        """Get indices of variants assigned to a component."""
        if self.n_variants == 0:
            return np.array([], dtype=int)
        return np.where(self.responsibilities.argmax(axis=1) == component)[0]


def build_somatic_prior_from_germline(
    germline_fit: GermlineFitResult,
    tumor_purity: float = 1.0,
    spec: SomaticMixtureSpec | None = None,
) -> SomaticPrior:
    """
    Construct somatic prior from germline fit results.

    Key mappings:
    - μ_het → clonal component location (allelic balance)
    - κ_het → technical dispersion for all components
    - |μ_het - 0.5| → LOH detection / component weight

    Parameters
    ----------
    germline_fit : GermlineFitResult
        Fitted germline parameters for the chromosome arm.
    tumor_purity : float
        Estimated tumor purity (0, 1].
    spec : SomaticMixtureSpec, optional
        Base specification for fallback values.

    Returns
    -------
    SomaticPrior
        Prior specification for somatic model.
    """
    if spec is None:
        spec = SomaticMixtureSpec()

    confidence = germline_fit.confidence

    # --- Subclonal component ---
    # Germline tells us nothing about subclonal fraction
    mu_subclonal = 0.15 * tumor_purity
    mu_subclonal = np.clip(mu_subclonal, 0.05, 0.25)
    sigma_subclonal = 0.08

    # --- Clonal component ---
    # Germline μ_het directly informs expected clonal VAF
    if germline_fit.has_loh_signal:
        mu_clonal = germline_fit.mu_het
        sigma_clonal = 0.05  # Tighter
    else:
        mu_clonal = 0.5 * tumor_purity + 0.5 * (1 - tumor_purity)
        mu_clonal = np.clip(mu_clonal, 0.35, 0.65)
        sigma_clonal = 0.10

    # --- LOH component ---
    if germline_fit.has_loh_signal:
        if germline_fit.mu_het > 0.5:
            mu_loh = germline_fit.mu_het
        else:
            mu_loh = 1.0 - germline_fit.mu_het
        mu_loh = np.clip(mu_loh, 0.6, 0.95)
        sigma_loh = 0.08
    else:
        mu_loh = 0.85
        sigma_loh = 0.12

    # --- Dispersion (from germline technical noise) ---
    kappa_mean = np.full(3, germline_fit.kappa_het)
    kappa_log_sigma = 0.5

    # --- Mixing proportions (Dirichlet) ---
    if germline_fit.has_loh_signal:
        pi_prior = np.array([1.0, 3.0, 3.0])
    else:
        pi_prior = np.array([2.0, 4.0, 1.5])

    pi_prior = pi_prior * (1 + confidence)

    # --- Adjust sigma by confidence ---
    sigma_scale = 1.0 / (0.5 + 0.5 * confidence)

    return SomaticPrior(
        chrom=germline_fit.chrom,
        arm=germline_fit.arm,
        mu_prior_mean=np.array([mu_subclonal, mu_clonal, mu_loh]),
        mu_prior_sigma=np.array([sigma_subclonal, sigma_clonal, sigma_loh])
        * sigma_scale,
        kappa_prior_mean=kappa_mean,
        kappa_prior_log_sigma=kappa_log_sigma,
        pi_prior=pi_prior,
        loh_detected=germline_fit.has_loh_signal,
        confidence=confidence,
        source="germline",
    )


def build_default_somatic_prior(
    chrom: str,
    arm: str,
    spec: SomaticMixtureSpec | None = None,
) -> SomaticPrior:
    """
    Build default somatic prior when germline data is unavailable.
    """
    if spec is None:
        spec = SomaticMixtureSpec()

    return SomaticPrior(
        chrom=chrom,
        arm=arm,
        mu_prior_mean=spec.mu_default.copy(),
        mu_prior_sigma=spec.mu_prior_sigma_default.copy(),
        kappa_prior_mean=spec.kappa_default.copy(),
        kappa_prior_log_sigma=spec.kappa_prior_log_sigma_default,
        pi_prior=spec.pi_prior_default.copy(),
        loh_detected=False,
        confidence=0.0,
        source="default",
    )


def smart_init_somatic(
    alt: np.ndarray,
    depth: np.ndarray,
    bounds_mu: np.ndarray,
) -> np.ndarray:
    """
    Smart initialization for somatic 3-component mixture.
    """
    if len(alt) == 0:
        return np.array([0.15, 0.50, 0.85])

    vaf = alt / (depth + 1e-9)

    # Subclonal: VAF < 0.25
    mask_sub = vaf < 0.25
    mu_sub = np.mean(vaf[mask_sub]) if np.sum(mask_sub) > 5 else 0.15

    # Clonal: VAF in [0.35, 0.65]
    mask_cl = (vaf >= 0.35) & (vaf <= 0.65)
    mu_cl = np.median(vaf[mask_cl]) if np.sum(mask_cl) > 5 else 0.5

    # LOH: VAF > 0.65
    mask_loh = vaf > 0.65
    mu_loh = np.mean(vaf[mask_loh]) if np.sum(mask_loh) > 5 else 0.85

    # Clip to bounds
    mu_sub = np.clip(mu_sub, bounds_mu[0, 0], bounds_mu[0, 1])
    mu_cl = np.clip(mu_cl, bounds_mu[1, 0], bounds_mu[1, 1])
    mu_loh = np.clip(mu_loh, bounds_mu[2, 0], bounds_mu[2, 1])

    return np.array([mu_sub, mu_cl, mu_loh])
