import numpy as np
import pytest

from somatic_mixture import (
    SomaticMixture,
    fit_somatic_mixture,
    fit_arm,
)
from utils import (
    SomaticPrior,
    SomaticMixtureSpec,
    GermlineFitResult,
)


class TestSomaticMixture:
    """Minimal tests for SomaticMixture."""

    @pytest.fixture
    def default_prior(self):
        """Default prior for testing."""
        return SomaticPrior(
            mu_prior_mean=np.array([0.15, 0.45, 0.85]),
            mu_prior_sigma=np.array([0.05, 0.1, 0.1]),
            kappa_prior_mean=np.array([50.0, 50.0, 50.0]),
            kappa_prior_log_sigma=0.5,
            pi_prior=np.array([1.0, 2.0, 1.0]),
            source="test",
        )

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic 3-component mixture data."""
        np.random.seed(42)

        # Subclonal (VAF ~ 0.15)
        n_sub = 30
        depth_sub = np.random.randint(50, 150, size=n_sub)
        alt_sub = np.random.binomial(depth_sub, 0.15)

        # Clonal (VAF ~ 0.45)
        n_clonal = 50
        depth_clonal = np.random.randint(50, 150, size=n_clonal)
        alt_clonal = np.random.binomial(depth_clonal, 0.45)

        # LOH (VAF ~ 0.85)
        n_loh = 20
        depth_loh = np.random.randint(50, 150, size=n_loh)
        alt_loh = np.random.binomial(depth_loh, 0.85)

        alt = np.concatenate([alt_sub, alt_clonal, alt_loh])
        depth = np.concatenate([depth_sub, depth_clonal, depth_loh])
        labels = np.array([0] * n_sub + [1] * n_clonal + [2] * n_loh)

        return alt, depth, labels

    def test_fit_basic(self, synthetic_data, default_prior):
        """Test basic fitting."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth, max_iter=100, n_restarts=2)

        assert model.mu is not None
        assert model.kappa is not None
        assert model.pi is not None
        assert len(model.mu) == 3
        assert len(model.pi) == 3
        assert np.isclose(model.pi.sum(), 1.0)

        # Check component ordering (subclonal < clonal < LOH)
        assert model.mu[0] < model.mu[1] < model.mu[2]

    def test_fit_recovers_components(self, synthetic_data, default_prior):
        """Test that fitting recovers approximate component means."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth, max_iter=200, n_restarts=3)

        # Should recover means within reasonable tolerance
        assert 0.05 < model.mu[0] < 0.25  # subclonal
        assert 0.30 < model.mu[1] < 0.60  # clonal
        assert 0.70 < model.mu[2] < 0.95  # LOH

    def test_fit_empty_data(self, default_prior):
        """Test fitting with empty data."""
        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(np.array([]), np.array([]))

        assert model.mu is not None
        assert model.converged is True
        assert model.n_variants == 0

    def test_fit_few_variants(self, default_prior):
        """Test fitting with very few variants uses prior."""
        alt = np.array([10, 25])
        depth = np.array([50, 50])

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth)

        # Should fall back to prior
        np.testing.assert_array_almost_equal(model.mu, default_prior.mu_prior_mean)

    def test_from_germline(self):
        """Test construction from germline result."""
        germline = GermlineFitResult(
            chrom="chr1",
            arm="p",
            mu_het=0.48,
            kappa_het=60.0,
            n_het=100,
            loglik_het=-500.0,
            mu_hom=0.97,
            kappa_hom=100.0,
            n_hom=30,
            loglik_hom=-100.0,
        )

        model = SomaticMixture.from_germline(
            chrom="chr1",
            arm="p",
            germline_result=germline,
            tumor_purity=0.8,
        )

        assert model.prior is not None
        assert model.prior.source == "germline"

    def test_predict_proba(self, synthetic_data, default_prior):
        """Test probability prediction."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth)

        proba = model.predict_proba(alt, depth)

        assert proba.shape == (len(alt), 3)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_predict(self, synthetic_data, default_prior):
        """Test label prediction."""
        alt, depth, true_labels = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth, n_restarts=3)

        pred = model.predict(alt, depth)

        assert len(pred) == len(alt)
        assert set(pred).issubset({0, 1, 2})

        # Should get most labels correct (>70% accuracy)
        accuracy = (pred == true_labels).mean()
        assert accuracy > 0.7

    def test_predict_with_threshold(self, synthetic_data, default_prior):
        """Test prediction with confidence threshold."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth)

        pred = model.predict(alt, depth, threshold=0.9)

        # Some predictions should be -1 (uncertain)
        assert -1 in pred or (model.responsibilities.max(axis=1) >= 0.9).all()

    def test_get_result(self, synthetic_data, default_prior):
        """Test result packaging."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth)
        result = model.get_result()

        assert result.chrom == "chr1"
        assert result.arm == "p"
        assert len(result.mu) == 3
        assert result.n_variants == len(alt)

    def test_get_result_before_fit_raises(self, default_prior):
        """Test that get_result raises before fit."""
        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)

        with pytest.raises(ValueError, match="Must call fit"):
            model.get_result()

    def test_get_component_summary(self, synthetic_data, default_prior):
        """Test component summary."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth)

        summary = model.get_component_summary()

        assert "subclonal" in summary
        assert "clonal" in summary
        assert "loh" in summary
        assert "mu" in summary["clonal"]
        assert "n_assigned" in summary["clonal"]

    def test_convergence(self, synthetic_data, default_prior):
        """Test that model converges."""
        alt, depth, _ = synthetic_data

        model = SomaticMixture(chrom="chr1", arm="p", prior=default_prior)
        model.fit(alt, depth, max_iter=500, tol=1e-6)

        assert model.converged is True
        assert model.n_iterations < 500


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_fit_somatic_mixture(self):
        """Test fit_somatic_mixture function."""
        np.random.seed(42)
        depth = np.random.randint(50, 100, size=50)
        alt = np.random.binomial(depth, 0.4)

        result = fit_somatic_mixture(
            chrom="chr1",
            arm="p",
            alt=alt,
            depth=depth,
        )

        assert result.chrom == "chr1"
        assert result.mu is not None

    def test_fit_somatic_mixture_with_germline(self):
        """Test fit_somatic_mixture with germline prior."""
        np.random.seed(42)
        depth = np.random.randint(50, 100, size=50)
        alt = np.random.binomial(depth, 0.4)

        germline = GermlineFitResult(
            chrom="chr1",
            arm="p",
            mu_het=0.5,
            kappa_het=50.0,
            n_het=100,
            loglik_het=-500.0,
            mu_hom=0.98,
            kappa_hom=100.0,
            n_hom=30,
            loglik_hom=-100.0,
        )

        result = fit_somatic_mixture(
            chrom="chr1",
            arm="p",
            alt=alt,
            depth=depth,
            germline_result=germline,
            tumor_purity=0.9,
        )

        assert result.prior.source == "germline"

    def test_fit_arm(self):
        """Test complete pipeline via fit_arm."""
        np.random.seed(42)

        # Germline het data
        germ_het_depth = np.random.randint(30, 100, size=50)
        germ_het_alt = np.random.binomial(germ_het_depth, 0.5)

        # Somatic data
        som_depth = np.random.randint(50, 100, size=40)
        som_alt = np.random.binomial(som_depth, 0.4)

        result = fit_arm(
            chrom="chr1",
            arm="p",
            germline_het_alt=germ_het_alt,
            germline_het_depth=germ_het_depth,
            somatic_alt=som_alt,
            somatic_depth=som_depth,
        )

        assert "germline" in result
        assert "somatic" in result
        assert result["prior_source"] == "germline"
        assert result["germline"] is not None
        assert result["somatic"] is not None

    def test_fit_arm_no_germline(self):
        """Test fit_arm falls back when insufficient germline."""
        np.random.seed(42)

        # Too few germline variants
        germ_het_alt = np.array([15, 18])
        germ_het_depth = np.array([30, 35])

        som_depth = np.random.randint(50, 100, size=40)
        som_alt = np.random.binomial(som_depth, 0.4)

        result = fit_arm(
            chrom="chr1",
            arm="p",
            germline_het_alt=germ_het_alt,
            germline_het_depth=germ_het_depth,
            somatic_alt=som_alt,
            somatic_depth=som_depth,
            min_germline_het=10,
        )

        assert result["prior_source"] == "default"
        assert result["germline"] is None
