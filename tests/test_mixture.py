import numpy as np
import pytest
from CBBmix.mixture import fit_robust_3_component_smart
from CBBmix.params import BetaBinomialMixtureSpec


class TestFitRobust3ComponentSmart:
    """Tests for fit_robust_3_component_smart function."""

    def test_basic_fit_returns_dict(self):
        """Test that function returns a result dictionary."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        assert isinstance(result, dict)
        assert "loglik" in result
        assert "bic" in result
        assert "mu" in result
        assert "kappa" in result
        assert "pi" in result
        assert "responsibilities" in result

    def test_three_component_data(self):
        """Test fitting clear three-component data."""
        np.random.seed(42)
        # Generate three clear clusters
        alt_sub = np.random.binomial(100, 0.1, 20)
        alt_cl = np.random.binomial(100, 0.5, 30)
        alt_high = np.random.binomial(100, 0.9, 20)

        alt = np.concatenate([alt_sub, alt_cl, alt_high])
        depth = np.full(70, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        # Check mu values are ordered and reasonable
        assert result["mu"].shape == (3,)
        assert result["mu"][0] < result["mu"][1] < result["mu"][2]
        assert 0 < result["mu"][0] < 0.3
        assert 0.3 < result["mu"][1] < 0.7
        assert 0.7 < result["mu"][2] < 1.0

    def test_output_shapes(self):
        """Test that output arrays have correct shapes."""
        np.random.seed(42)
        N = 50
        alt = np.random.binomial(100, 0.5, N)
        depth = np.full(N, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        assert result["mu"].shape == (3,)
        assert result["kappa"].shape == (3,)
        assert result["pi"].shape == (3,)
        assert result["responsibilities"].shape == (N, 3)

    def test_pi_sums_to_one(self):
        """Test that mixing proportions sum to 1."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        assert np.isclose(result["pi"].sum(), 1.0)

    def test_responsibilities_sum_to_one(self):
        """Test that responsibilities sum to 1 across components."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        row_sums = result["responsibilities"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_loglik_increases_or_stable(self):
        """Test that best fit has reasonable log-likelihood."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        # Log-likelihood should be finite and negative (log of probability)
        assert np.isfinite(result["loglik"])
        assert result["loglik"] < 0

    def test_bic_calculation(self):
        """Test that BIC is calculated correctly."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        N = 50
        k_params = 8  # 3 mu + 3 kappa + 2 pi
        expected_bic = -2 * result["loglik"] + k_params * np.log(N)

        np.testing.assert_allclose(result["bic"], expected_bic)

    def test_custom_spec(self):
        """Test using custom BetaBinomialMixtureSpec."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        custom_spec = BetaBinomialMixtureSpec(
            bounds_mu=[[0.05, 0.2], [0.4, 0.6], [0.8, 0.95]], weight_prior=2.0
        )

        result = fit_robust_3_component_smart(
            alt, depth, spec=custom_spec, random_state=42
        )

        # Check mu values respect custom bounds
        assert (
            custom_spec.bounds_mu[0, 0]
            <= result["mu"][0]
            <= custom_spec.bounds_mu[0, 1]
        )
        assert (
            custom_spec.bounds_mu[1, 0]
            <= result["mu"][1]
            <= custom_spec.bounds_mu[1, 1]
        )
        assert (
            custom_spec.bounds_mu[2, 0]
            <= result["mu"][2]
            <= custom_spec.bounds_mu[2, 1]
        )

    def test_max_iter_parameter(self):
        """Test that max_iter limits iterations."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        # Should not raise error with very small max_iter
        result = fit_robust_3_component_smart(alt, depth, max_iter=5, random_state=42)
        assert result is not None

    def test_tolerance_parameter(self):
        """Test that tolerance affects convergence."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        # Larger tolerance should converge faster
        result = fit_robust_3_component_smart(alt, depth, tol=1e-2, random_state=42)
        assert result is not None

    def test_n_restarts_parameter(self):
        """Test that n_restarts works."""
        np.random.seed(42)
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result_1 = fit_robust_3_component_smart(
            alt, depth, n_restarts=1, random_state=42
        )
        result_5 = fit_robust_3_component_smart(
            alt, depth, n_restarts=5, random_state=42
        )

        # More restarts should give at least as good or better likelihood
        assert result_5["loglik"] >= result_1["loglik"]

    def test_random_state_reproducibility(self):
        """Test that random_state makes results reproducible."""
        alt = np.random.binomial(100, 0.5, 50)
        depth = np.full(50, 100.0)

        result1 = fit_robust_3_component_smart(alt, depth, random_state=42)
        result2 = fit_robust_3_component_smart(alt, depth, random_state=42)

        np.testing.assert_array_equal(result1["mu"], result2["mu"])
        np.testing.assert_array_equal(result1["kappa"], result2["kappa"])

    def test_varying_depth(self):
        """Test with varying sequencing depth."""
        np.random.seed(42)
        depth = np.random.randint(50, 200, 50)
        alt = np.random.binomial(depth, 0.5)

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        assert result is not None
        assert np.all(np.isfinite(result["mu"]))

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        alt = np.array([10, 50, 90])
        depth = np.array([100, 100, 100])

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        assert result is not None
        assert result["responsibilities"].shape == (3, 3)

    def test_input_conversion(self):
        """Test that inputs are converted to numpy arrays."""
        alt = [10, 50, 90, 10, 50, 90]
        depth = [100, 100, 100, 100, 100, 100]

        result = fit_robust_3_component_smart(alt, depth, random_state=42)

        assert result is not None
        assert isinstance(result["mu"], np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
