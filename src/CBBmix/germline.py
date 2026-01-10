"""
Germline parameter estimation with known zygosity.

Since zygosity (het vs hom) is known from the VCF, we don't need a mixture
model. Instead, we fit independent Beta-Binomials to each class via MLE.

This module estimates:
1. μ_het, κ_het: Mean and precision for heterozygous variants
2. μ_hom, κ_hom: Mean and precision for homozygous ALT variants

The key output (μ_het, κ_het) informs the somatic prior:
- μ_het deviation from 0.5 → LOH signal
- κ_het → arm-specific technical dispersion
"""

import numpy as np
from typing import Optional

from .utils import (
    fit_beta_binomial_mle,
    GermlineSpec,
    GermlineFitResult,
)


class GermlineEstimator:
    """
    Estimate germline Beta-Binomial parameters with known zygosity labels.

    Since we know which variants are heterozygous vs homozygous, we fit
    each class separately via MLE. No EM or mixture model needed.

    Parameters
    ----------
    chrom : str
        Chromosome identifier (e.g., 'chr1', '1').
    arm : str
        Chromosome arm ('p' or 'q').
    spec : GermlineSpec, optional
        Specification with bounds and thresholds.

    Attributes
    ----------
    mu_het, kappa_het : float
        MLE parameters for heterozygous variants.
    mu_hom, kappa_hom : float
        MLE parameters for homozygous ALT variants.
    """

    def __init__(
        self,
        chrom: str,
        arm: str,
        spec: Optional[GermlineSpec] = None,
    ):
        self.chrom = chrom
        self.arm = arm
        self.spec = spec if spec is not None else GermlineSpec()

        # Parameters (set after fitting)
        self.mu_het: Optional[float] = None
        self.kappa_het: Optional[float] = None
        self.loglik_het: float = 0.0
        self.n_het: int = 0

        self.mu_hom: Optional[float] = None
        self.kappa_hom: Optional[float] = None
        self.loglik_hom: float = 0.0
        self.n_hom: int = 0

    def fit(
        self,
        het_alt: np.ndarray,
        het_depth: np.ndarray,
        hom_alt: np.ndarray,
        hom_depth: np.ndarray,
    ) -> "GermlineEstimator":
        """
        Fit Beta-Binomial parameters for het and hom classes separately.

        Parameters
        ----------
        het_alt : np.ndarray
            Alt counts for heterozygous variants.
        het_depth : np.ndarray
            Total depth for heterozygous variants.
        hom_alt : np.ndarray
            Alt counts for homozygous ALT variants.
        hom_depth : np.ndarray
            Total depth for homozygous ALT variants.

        Returns
        -------
        GermlineEstimator
            Self, for method chaining.
        """
        het_alt = np.asarray(het_alt, dtype=float)
        het_depth = np.asarray(het_depth, dtype=float)
        hom_alt = np.asarray(hom_alt, dtype=float)
        hom_depth = np.asarray(hom_depth, dtype=float)

        self.n_het = len(het_alt)
        self.n_hom = len(hom_alt)

        # --- Fit heterozygous class ---
        if self.n_het >= self.spec.min_variants_het:
            self.mu_het, self.kappa_het, self.loglik_het = fit_beta_binomial_mle(
                het_alt,
                het_depth,
                mu_bounds=self.spec.mu_het_bounds,
                kappa_bounds=self.spec.kappa_bounds,
            )
        else:
            # Fallback to defaults with simple estimate
            if self.n_het > 0:
                vaf = het_alt / (het_depth + 1e-9)
                self.mu_het = np.clip(np.mean(vaf), 0.3, 0.7)
            else:
                self.mu_het = self.spec.mu_het_expected
            self.kappa_het = 50.0
            self.loglik_het = 0.0

        # --- Fit homozygous ALT class ---
        if self.n_hom >= self.spec.min_variants_hom:
            self.mu_hom, self.kappa_hom, self.loglik_hom = fit_beta_binomial_mle(
                hom_alt,
                hom_depth,
                mu_bounds=self.spec.mu_hom_bounds,
                kappa_bounds=self.spec.kappa_bounds,
            )
        else:
            # Fallback
            if self.n_hom > 0:
                vaf = hom_alt / (hom_depth + 1e-9)
                self.mu_hom = np.clip(np.mean(vaf), 0.9, 0.999)
            else:
                self.mu_hom = 0.98
            self.kappa_hom = 100.0
            self.loglik_hom = 0.0

        return self

    def fit_het_only(
        self,
        het_alt: np.ndarray,
        het_depth: np.ndarray,
    ) -> "GermlineEstimator":
        """
        Fit only the heterozygous class (when hom variants unavailable).

        Parameters
        ----------
        het_alt : np.ndarray
            Alt counts for heterozygous variants.
        het_depth : np.ndarray
            Total depth for heterozygous variants.

        Returns
        -------
        GermlineEstimator
            Self, for method chaining.
        """
        return self.fit(
            het_alt=het_alt,
            het_depth=het_depth,
            hom_alt=np.array([]),
            hom_depth=np.array([]),
        )

    def get_result(self) -> GermlineFitResult:
        """
        Package results into GermlineFitResult.

        Returns
        -------
        GermlineFitResult
            Structured results with all parameters and derived quantities.
        """
        if self.mu_het is None:
            raise ValueError("Must call fit() before get_result()")

        return GermlineFitResult(
            chrom=self.chrom,
            arm=self.arm,
            mu_het=self.mu_het,
            kappa_het=self.kappa_het,
            n_het=self.n_het,
            loglik_het=self.loglik_het,
            mu_hom=self.mu_hom,
            kappa_hom=self.kappa_hom,
            n_hom=self.n_hom,
            loglik_hom=self.loglik_hom,
        )

    @property
    def allelic_imbalance(self) -> float:
        """Absolute deviation of mu_het from 0.5."""
        if self.mu_het is None:
            return 0.0
        return abs(self.mu_het - 0.5)

    @property
    def has_loh_signal(self) -> bool:
        """Whether het distribution suggests LOH."""
        return (
            self.n_het >= 10
            and self.allelic_imbalance > self.spec.loh_imbalance_threshold
        )

    def __repr__(self) -> str:
        status = "fitted" if self.mu_het is not None else "not fitted"
        return (
            f"GermlineEstimator(chrom={self.chrom!r}, arm={self.arm!r}, "
            f"n_het={self.n_het}, n_hom={self.n_hom}, status={status})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def fit_germline(
    chrom: str,
    arm: str,
    het_alt: np.ndarray,
    het_depth: np.ndarray,
    hom_alt: np.ndarray | None = None,
    hom_depth: np.ndarray | None = None,
    spec: Optional[GermlineSpec] = None,
) -> GermlineFitResult:
    """
    Convenience function to fit germline parameters and return results.

    Parameters
    ----------
    chrom : str
        Chromosome identifier.
    arm : str
        Chromosome arm.
    het_alt : np.ndarray
        Alt counts for heterozygous variants.
    het_depth : np.ndarray
        Depths for heterozygous variants.
    hom_alt : np.ndarray, optional
        Alt counts for homozygous ALT variants.
    hom_depth : np.ndarray, optional
        Depths for homozygous ALT variants.
    spec : GermlineSpec, optional
        Specification.

    Returns
    -------
    GermlineFitResult
        Fit results.
    """
    estimator = GermlineEstimator(chrom, arm, spec)

    if hom_alt is None or hom_depth is None:
        hom_alt = np.array([])
        hom_depth = np.array([])

    estimator.fit(het_alt, het_depth, hom_alt, hom_depth)
    return estimator.get_result()


def fit_germline_from_combined(
    chrom: str,
    arm: str,
    alt: np.ndarray,
    depth: np.ndarray,
    is_het: np.ndarray,
    spec: Optional[GermlineSpec] = None,
) -> GermlineFitResult:
    """
    Fit germline from combined arrays with a boolean zygosity mask.

    Parameters
    ----------
    chrom : str
        Chromosome identifier.
    arm : str
        Chromosome arm.
    alt : np.ndarray
        Alt counts for all germline variants.
    depth : np.ndarray
        Depths for all germline variants.
    is_het : np.ndarray
        Boolean array: True for het, False for hom ALT.
    spec : GermlineSpec, optional
        Specification.

    Returns
    -------
    GermlineFitResult
        Fit results.
    """
    alt = np.asarray(alt, dtype=float)
    depth = np.asarray(depth, dtype=float)
    is_het = np.asarray(is_het, dtype=bool)

    het_mask = is_het
    hom_mask = ~is_het

    return fit_germline(
        chrom=chrom,
        arm=arm,
        het_alt=alt[het_mask],
        het_depth=depth[het_mask],
        hom_alt=alt[hom_mask],
        hom_depth=depth[hom_mask],
        spec=spec,
    )
