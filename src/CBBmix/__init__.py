"""
CBBmix: 3-component Beta-Binomial mixture model for clonal structure analysis.

This package provides Bayesian inference tools for analyzing somatic variants
in cancer genomics data from RNA-seq.

Main components:
- GermlineModel: Estimates per-arm allelic imbalance (delta, kappa, psi)
- SomaticModel: Pitman-Yor process clustering by Cellular Prevalence
- GermlineVariantCollector / SomaticVariantCollector: VCF parsing utilities

Example usage:
    from CBBmix import (
        GermlineVariantCollector,
        SomaticVariantCollector,
        GermlineModel,
        SomaticModel,
        SomaticPriorConfig,
    )

    # Collect variants
    germ_collector = GermlineVariantCollector("sample.vcf.gz")
    som_collector = SomaticVariantCollector("sample.vcf.gz")

    # Fit germline model
    germ_model = GermlineModel(germ_collector)
    germ_model.fit()

    # Fit somatic model with germline priors
    som_model = SomaticModel(som_collector, germ_model)
    som_model.fit()
"""

from .vcf import (
    GermlineVariantCollector,
    SomaticVariantCollector,
    ChromosomeArmLookup,
)
from .germline import GermlineModel
from .somatic import SomaticModel, SomaticPriorConfig

__version__ = "0.1.0"

__all__ = [
    "GermlineVariantCollector",
    "SomaticVariantCollector",
    "ChromosomeArmLookup",
    "GermlineModel",
    "SomaticModel",
    "SomaticPriorConfig",
]