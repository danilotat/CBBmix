import numpy as np
import pytest
from CBBmix.params import BetaBinomialMixtureSpec


class TestBetaBinomialMixtureSpec:
    """Tests for BetaBinomialMixtureSpec dataclass."""
    
    def test_default_initialization(self):
        """Test default values are set correctly."""
        spec = BetaBinomialMixtureSpec()
        
        # Shape checks
        assert spec.bounds_mu.shape == (3, 2)
        assert spec.kappa_init.shape == (3,)
        assert spec.pi_init.shape == (3,)
        
        # Value checks
        np.testing.assert_array_equal(spec.bounds_mu[0], [0, 0.25])
        np.testing.assert_array_equal(spec.bounds_mu[1], [0.35, 0.65])
        np.testing.assert_array_equal(spec.bounds_mu[2], [0.85, 1])
        assert spec.bounds_kappa == (5, 5000)
        assert np.isclose(spec.pi_init.sum(), 1.0)
        assert spec.weight_prior == 1.5
        assert spec.mu_prior_mean == 0.5
        assert spec.mu_prior_sigma == 0.1
    
    def test_custom_parameters(self):
        """Test setting custom parameters."""
        spec = BetaBinomialMixtureSpec(
            bounds_mu=[[0.1, 0.3], [0.4, 0.6], [0.7, 0.9]],
            bounds_kappa=(10, 2000),
            kappa_init=[30.0, 80.0, 40.0],
            pi_init=[0.5, 0.3, 0.2],
            weight_prior=2.0,
            mu_prior_mean=0.6,
            mu_prior_sigma=0.15
        )
        
        assert spec.bounds_mu[0, 0] == 0.1
        assert spec.bounds_kappa == (10, 2000)
        assert np.isclose(spec.pi_init.sum(), 1.0)  # normalized
        assert spec.weight_prior == 2.0
    
    def test_pi_normalization(self):
        """Test that pi_init is normalized to sum to 1."""
        spec = BetaBinomialMixtureSpec(pi_init=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(spec.pi_init, [1/6, 2/6, 3/6])
    
    def test_validation_bounds_mu_shape(self):
        """Test that invalid bounds_mu shape raises ValueError."""
        with pytest.raises(ValueError, match="bounds_mu must have shape"):
            BetaBinomialMixtureSpec(bounds_mu=[[0, 0.5], [0.5, 1]])
    
    def test_validation_kappa_init_shape(self):
        """Test that invalid kappa_init shape raises ValueError."""
        with pytest.raises(ValueError, match="kappa_init must have shape"):
            BetaBinomialMixtureSpec(kappa_init=[50.0, 100.0])
    
    def test_validation_pi_init_shape(self):
        """Test that invalid pi_init shape raises ValueError."""
        with pytest.raises(ValueError, match="pi_init must have shape"):
            BetaBinomialMixtureSpec(pi_init=[0.5, 0.5])
    
    def test_independent_copies(self):
        """Test that default factory creates independent arrays."""
        spec1 = BetaBinomialMixtureSpec()
        spec2 = BetaBinomialMixtureSpec()
        
        spec1.bounds_mu[0, 0] = 999.0
        assert spec2.bounds_mu[0, 0] != 999.0
    
    def test_array_conversion(self):
        """Test that lists are converted to numpy arrays with float dtype."""
        spec = BetaBinomialMixtureSpec(
            bounds_mu=[[0, 1], [1, 2], [2, 3]],
            kappa_init=[50, 100, 50],
            pi_init=[1, 1, 1]
        )
        
        assert isinstance(spec.bounds_mu, np.ndarray)
        assert spec.bounds_mu.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
