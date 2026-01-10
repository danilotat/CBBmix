import numpy as np
import pytest

from germline_estimator import (
    GermlineEstimator,
    fit_germline,
    fit_germline_from_combined,
)


class TestGermlineEstimator:
    """Minimal tests for GermlineEstimator."""

    def test_fit_basic(self):
        """Test basic fitting with synthetic data."""
        # Simulate het variants around VAF=0.5
        np.random.seed(42)
        n_het = 50
        het_depth = np.random.randint(30, 100, size=n_het)
        het_alt = np.random.binomial(het_depth, 0.5)

        # Simulate hom variants around VAF=0.98
        n_hom = 20
        hom_depth = np.random.randint(30, 100, size=n_hom)
        hom_alt = np.random.binomial(hom_depth, 0.98)

        estimator = GermlineEstimator(chrom="chr1", arm="p")
        estimator.fit(het_alt, het_depth, hom_alt, hom_depth)

        # Check parameters were set
        assert estimator.mu_het is not None
        assert estimator.kappa_het is not None
        assert estimator.mu_hom is not None
        assert estimator.kappa_hom is not None

        # Check reasonable ranges
        assert 0.3 < estimator.mu_het < 0.7
        assert 0.9 < estimator.mu_hom < 1.0
        assert estimator.n_het == n_het
        assert estimator.n_hom == n_hom

    def test_fit_het_only(self):
        """Test fitting with only heterozygous variants."""
        np.random.seed(42)
        n_het = 30
        het_depth = np.random.randint(30, 100, size=n_het)
        het_alt = np.random.binomial(het_depth, 0.5)

        estimator = GermlineEstimator(chrom="chr1", arm="q")
        estimator.fit_het_only(het_alt, het_depth)

        assert estimator.mu_het is not None
        assert estimator.n_hom == 0

    def test_fallback_with_few_variants(self):
        """Test fallback behavior with insufficient variants."""
        het_alt = np.array([15, 18])
        het_depth = np.array([30, 35])

        estimator = GermlineEstimator(chrom="chr2", arm="p")
        estimator.fit(het_alt, het_depth, np.array([]), np.array([]))

        # Should use fallback values
        assert estimator.mu_het is not None
        assert estimator.kappa_het == 50.0  # fallback kappa

    def test_get_result(self):
        """Test get_result returns proper structure."""
        np.random.seed(42)
        het_depth = np.random.randint(30, 100, size=30)
        het_alt = np.random.binomial(het_depth, 0.5)

        estimator = GermlineEstimator(chrom="chr1", arm="p")
        estimator.fit_het_only(het_alt, het_depth)
        result = estimator.get_result()

        assert result.chrom == "chr1"
        assert result.arm == "p"
        assert result.mu_het is not None
        assert result.n_het == 30

    def test_get_result_before_fit_raises(self):
        """Test that get_result raises if fit not called."""
        estimator = GermlineEstimator(chrom="chr1", arm="p")
        with pytest.raises(ValueError, match="Must call fit"):
            estimator.get_result()

    def test_allelic_imbalance(self):
        """Test allelic imbalance calculation."""
        estimator = GermlineEstimator(chrom="chr1", arm="p")
        estimator.mu_het = 0.4
        assert estimator.allelic_imbalance == pytest.approx(0.1)

    def test_has_loh_signal(self):
        """Test LOH signal detection."""
        estimator = GermlineEstimator(chrom="chr1", arm="p")
        estimator.mu_het = 0.35  # imbalance = 0.15
        estimator.n_het = 50

        assert estimator.has_loh_signal is True

        estimator.mu_het = 0.48  # imbalance = 0.02
        assert estimator.has_loh_signal is False


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_fit_germline(self):
        """Test fit_germline function."""
        np.random.seed(42)
        het_depth = np.random.randint(30, 100, size=30)
        het_alt = np.random.binomial(het_depth, 0.5)

        result = fit_germline(
            chrom="chr1",
            arm="p",
            het_alt=het_alt,
            het_depth=het_depth,
        )

        assert result.chrom == "chr1"
        assert result.mu_het is not None

    def test_fit_germline_from_combined(self):
        """Test fit_germline_from_combined function."""
        np.random.seed(42)

        # Combined arrays
        n_het, n_hom = 30, 15
        het_depth = np.random.randint(30, 100, size=n_het)
        het_alt = np.random.binomial(het_depth, 0.5)
        hom_depth = np.random.randint(30, 100, size=n_hom)
        hom_alt = np.random.binomial(hom_depth, 0.98)

        alt = np.concatenate([het_alt, hom_alt])
        depth = np.concatenate([het_depth, hom_depth])
        is_het = np.array([True] * n_het + [False] * n_hom)

        result = fit_germline_from_combined(
            chrom="chr1",
            arm="q",
            alt=alt,
            depth=depth,
            is_het=is_het,
        )

        assert result.n_het == n_het
        assert result.n_hom == n_hom
