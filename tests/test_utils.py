import numpy as np
import pytest
from scipy.stats import betabinom

from CBBmix.utils import beta_binom_logpmf, ab_from_mu_kappa, smart_init_somatic

class TestBetaBinomLogpmf:
    """Tests for beta_binom_logpmf function."""

    def test_matches_scipy(self):
        """Verify implementation matches scipy's beta-binomial."""
        n, a = 100, 30
        alpha, beta = 10.0, 20.0

        result = beta_binom_logpmf(a, n, alpha, beta)
        expected = betabinom.logpmf(a, n, alpha, beta)

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_vectorized(self):
        """Test with array inputs."""
        n = np.array([100, 200, 150])
        a = np.array([30, 60, 75])

        result = beta_binom_logpmf(a, n, 10.0, 20.0)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

        # Spot check against scipy
        expected = betabinom.logpmf(a[0], n[0], 10.0, 20.0)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_edge_cases(self):
        """Test boundary cases."""
        # a = 0
        result_zero = beta_binom_logpmf(0, 100, 10.0, 20.0)
        assert np.isfinite(result_zero)

        # a = n
        result_full = beta_binom_logpmf(100, 100, 10.0, 20.0)
        assert np.isfinite(result_full)

        # Very small alpha/beta (clamped to 1e-9)
        result_small = beta_binom_logpmf(50, 100, 1e-12, 1e-12)
        assert np.isfinite(result_small)

    def test_symmetric_distribution(self):
        """Test symmetry when alpha equals beta."""
        n = 100
        alpha = beta = 50.0

        result_low = beta_binom_logpmf(30, n, alpha, beta)
        result_high = beta_binom_logpmf(70, n, alpha, beta)

        np.testing.assert_allclose(result_low, result_high, rtol=1e-10)

    def test_output_is_log_probability(self):
        """Verify output is in log scale (negative or zero)."""
        result = beta_binom_logpmf(50, 100, 10.0, 10.0)
        assert result <= 0


class TestAbFromMuKappa:
    """Tests for ab_from_mu_kappa function."""

    def test_basic_conversion(self):
        """Test mu/kappa to alpha/beta conversion."""
        mu, kappa = 0.5, 100.0
        alpha, beta = ab_from_mu_kappa(mu, kappa)

        assert alpha == 50.0
        assert beta == 50.0

    def test_reconstruction(self):
        """Verify mu and kappa can be reconstructed."""
        mu, kappa = 0.7, 200.0
        alpha, beta = ab_from_mu_kappa(mu, kappa)

        assert np.isclose(alpha / (alpha + beta), mu)
        assert np.isclose(alpha + beta, kappa)

    def test_clipping(self):
        """Test that extreme mu values are clipped."""
        # Near zero
        alpha, beta = ab_from_mu_kappa(1e-10, 100.0)
        assert alpha == pytest.approx(1e-6 * 100.0)

        # Near one
        alpha, beta = ab_from_mu_kappa(1 - 1e-10, 100.0)
        assert beta == pytest.approx(1e-6 * 100.0)

    def test_vectorized(self):
        """Test with array inputs."""
        mu = np.array([0.2, 0.5, 0.8])
        kappa = np.array([100.0, 200.0, 150.0])

        alpha, beta = ab_from_mu_kappa(mu, kappa)

        np.testing.assert_allclose(alpha, [20.0, 100.0, 120.0])
        np.testing.assert_allclose(beta, [80.0, 100.0, 30.0])

    def test_always_positive(self):
        """Ensure alpha and beta are always positive."""
        for mu in np.linspace(0, 1, 20):
            alpha, beta = ab_from_mu_kappa(mu, 100.0)
            assert alpha > 0 and beta > 0


class TestSmartInitParameters:
    """Tests for smart_init_parameters function."""

    def test_three_component_data(self):
        """Test initialization with clear clusters."""
        np.random.seed(42)
        alt = np.concatenate(
            [
                np.random.binomial(100, 0.1, 20),  # subclonal
                np.random.binomial(100, 0.5, 30),  # clonal
                np.random.binomial(100, 0.9, 15),  # high/LOH
            ]
        )
        depth = np.full_like(alt, 100.0)
        bounds_mu = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1.0]])

        result = smart_init_somatic(alt, depth, bounds_mu)

        assert result.shape == (3,)
        assert result[0] < result[1] < result[2]  # Ordered
        # Within bounds
        for i in range(3):
            assert bounds_mu[i, 0] <= result[i] <= bounds_mu[i, 1]

    def test_sparse_data_fallbacks(self):
        """Test fallback values when data is sparse."""
        # Only clonal-like variants
        alt = np.array([45, 50, 48, 52])
        depth = np.array([100, 100, 100, 100])
        bounds_mu = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1.0]])

        result = smart_init_somatic(alt, depth, bounds_mu)

        # Subclonal should use bounds mean
        assert result[0] == np.mean(bounds_mu[0])
        # Clonal should use 0.5
        assert result[1] == 0.5
        # High should use 0.9
        assert result[2] == 0.9

    def test_clipping_to_bounds(self):
        """Test that results are always within bounds."""
        alt = np.concatenate([np.full(10, 0), np.full(10, 50), np.full(10, 100)])
        depth = np.full(30, 100.0)
        bounds_mu = np.array([[0.05, 0.25], [0.35, 0.65], [0.85, 0.95]])

        result = smart_init_somatic(alt, depth, bounds_mu)

        for i in range(3):
            assert bounds_mu[i, 0] <= result[i] <= bounds_mu[i, 1]

    def test_median_for_clonal(self):
        """Test that clonal uses median for robustness."""
        # Clonal data with outlier
        alt = np.array([40, 45, 48, 50, 52, 55, 80])
        depth = np.full(7, 100.0)
        bounds_mu = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1.0]])

        result = smart_init_somatic(alt, depth, bounds_mu)

        vaf = alt / depth
        mask_cl = (vaf >= 0.35) & (vaf <= 0.65)
        expected = np.median(vaf[mask_cl])

        np.testing.assert_allclose(result[1], expected)

    def test_handles_zero_depth(self):
        """Test graceful handling of zero depth."""
        alt = np.array([0, 10, 20])
        depth = np.array([0, 100, 100])
        bounds_mu = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1.0]])

        result = smart_init_somatic(alt, depth, bounds_mu)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_returns_numpy_array(self):
        """Test return type and dtype."""
        alt = np.array([10, 50, 90])
        depth = np.array([100, 100, 100])
        bounds_mu = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1.0]])

        result = smart_init_somatic(alt, depth, bounds_mu)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestIntegration:
    """Integration tests combining functions."""

    def test_workflow(self):
        """Test typical workflow: init -> convert -> calculate likelihood."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 20)
        depth = np.full(20, 100.0)
        bounds_mu = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1.0]])

        # Initialize
        mu_init = smart_init_somatic(alt, depth, bounds_mu)

        # Convert and calculate
        for k in range(3):
            alpha, beta = ab_from_mu_kappa(mu_init[k], 100.0)
            logp = beta_binom_logpmf(alt, depth, alpha, beta)

            assert logp.shape == (20,)
            assert np.all(np.isfinite(logp))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
