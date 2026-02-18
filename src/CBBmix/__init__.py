"""
CBBmix: 3-component Beta-Binomial mixture model for clonal structure analysis.

This package provides Bayesian inference tools for analyzing somatic variants
in cancer genomics data from RNA-seq.

Main components:
- GermlineModel: Estimates per-arm allelic imbalance (delta, kappa, psi)
- SegmentedGermlineModel: Horseshoe-fused segmentation for per-variant psi
- SomaticModel: Pitman-Yor process clustering by Cellular Prevalence
- GermlineVariantCollector / SomaticVariantCollector: VCF parsing utilities

Example usage (arm-level):
    from CBBmix import (
        GermlineVariantCollector,
        SomaticVariantCollector,
        GermlineModel,
        SomaticModel,
    )

    germ_collector = GermlineVariantCollector("sample.vcf.gz")
    som_collector = SomaticVariantCollector("sample.vcf.gz")

    germ_model = GermlineModel(germ_collector)
    germ_model.fit()

    som_model = SomaticModel(som_collector, germ_model)
    som_model.fit()

Example usage (segmented):
    from CBBmix import (
        GermlineVariantCollector,
        SomaticVariantCollector,
        SegmentedGermlineModel,
        SomaticModel,
    )

    germ_collector = GermlineVariantCollector("sample.vcf.gz")
    som_collector = SomaticVariantCollector("sample.vcf.gz")

    # Use segmented model for finer-grained psi estimation
    germ_model = SegmentedGermlineModel(germ_collector)
    germ_model.fit()

    # SomaticModel automatically detects and uses segment lookup
    som_model = SomaticModel(som_collector, germ_model)
    som_model.fit()
"""

from .vcf import (
    GermlineVariantCollector,
    SomaticVariantCollector,
    ChromosomeArmLookup,
)
from .germline import GermlineModel, SegmentedGermlineModel
from .somatic import SomaticModel, SomaticPriorConfig
from .utils import (
    SegmentLookup,
    SegmentInfo,
    SegmentResult,
    ChromosomeSegmentationResult,
    compute_scaled_distances,
    extract_segments_from_posterior,
    prune_and_merge_segments,
    build_somatic_prior_from_germline,
)
from .plotting import (
    VariantHandler,
    BAFPlotter,
)
from .baf import compute_baf, compute_baf_genome, HAS_BAF_EXTENSION

__version__ = "0.1.0"

__all__ = [
    # VCF utilities
    "GermlineVariantCollector",
    "SomaticVariantCollector",
    "ChromosomeArmLookup",
    # Germline models
    "GermlineModel",
    "SegmentedGermlineModel",
    # Somatic model
    "SomaticModel",
    "SomaticPriorConfig",
    # Segmentation utilities
    "SegmentLookup",
    "SegmentInfo",
    "SegmentResult",
    "ChromosomeSegmentationResult",
    "compute_scaled_distances",
    "extract_segments_from_posterior",
    "prune_and_merge_segments",
    "build_somatic_prior_from_germline",
    # Plotting
    "VariantHandler",
    "BAFPlotter",
    # BAF extension
    "compute_baf",
    "compute_baf_genome",
    "HAS_BAF_EXTENSION",
]