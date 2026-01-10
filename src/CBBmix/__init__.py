"""
Beta-Binomial Mixture Models for Somatic Variant Analysis.
...
"""

from .utils import (
    # Distribution functions
    beta_binom_logpmf,
    ab_from_mu_kappa,
    fit_beta_binomial_mle,
    # Data classes
    GermlineSpec,
    SomaticMixtureSpec,
    GermlineFitResult,
    SomaticFitResult,
    SomaticPrior,
    # Prior construction
    build_somatic_prior_from_germline,
    build_default_somatic_prior,
)

from .germline import (
    GermlineEstimator,
    fit_germline,
    fit_germline_from_combined,
)

from .somatic import (
    SomaticMixture,
    fit_somatic_mixture,
    fit_arm,
)

from .vcf import (
    ChromosomeArmLookup,
    GermlineVariantCollector,
)

__all__ = [
    # Classes
    "GermlineEstimator",
    "SomaticMixture",
    "ChromosomeArmLookup",
    "GermlineVariantCollector",
    # Convenience functions
    "fit_germline",
    "fit_germline_from_combined",
    "fit_somatic_mixture",
    "fit_arm",
    # Data classes
    "GermlineSpec",
    "SomaticMixtureSpec",
    "GermlineFitResult",
    "SomaticFitResult",
    "SomaticPrior",
    # Utilities
    "beta_binom_logpmf",
    "ab_from_mu_kappa",
    "fit_beta_binomial_mle",
    "build_somatic_prior_from_germline",
    "build_default_somatic_prior",
]

__version__ = "0.2.0"
