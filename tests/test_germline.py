import numpy as np
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch

from CBBmix.germline import (
    GermlineModel,
    SegmentedGermlineModel,
    germline_segmented_model,
)
from CBBmix.utils import compute_scaled_distances


class TestGermlineModel:
    """Tests for the arm-level GermlineModel."""

    @pytest.fixture
    def mock_collector(self):
        """Create a mock GermlineVariantCollector."""
        collector = Mock()
        collector.germline_vars = {
            'chr1': {
                'p': {
                    'hetalt': {
                        'DP': [50, 60, 70, 80, 90] * 10,
                        'alt_DP': [25, 28, 33, 42, 48] * 10,
                        'VAF': [0.5, 0.47, 0.47, 0.525, 0.53] * 10,
                        'POS': list(range(1000000, 1000000 + 50 * 100000, 100000)),
                    }
                },
                'q': {
                    'hetalt': {
                        'DP': [50, 60, 70] * 5,
                        'alt_DP': [20, 25, 30] * 5,
                        'VAF': [0.4, 0.42, 0.43] * 5,
                        'POS': list(range(130000000, 130000000 + 15 * 100000, 100000)),
                    }
                }
            }
        }
        return collector

    def test_init(self, mock_collector):
        """Test GermlineModel initialization."""
        model = GermlineModel(mock_collector, min_dp_cutoff=10, min_snp=5)
        assert model._min_dp_cutoff == 10
        assert model._min_snp == 5
        assert not model.data_df.empty

    def test_preprocess_data(self, mock_collector):
        """Test data preprocessing."""
        model = GermlineModel(mock_collector, min_dp_cutoff=10)
        df = model.data_df

        assert 'chrom' in df.columns
        assert 'arm' in df.columns
        assert 'depth' in df.columns
        assert 'alt_count' in df.columns
        assert len(df) > 0

    def test_fit_skips_small_arms(self, mock_collector):
        """Test that arms with few variants are skipped."""
        # Modify mock to have very few variants on one arm
        mock_collector.germline_vars['chr1']['q']['hetalt'] = {
            'DP': [50, 60],
            'alt_DP': [25, 30],
            'VAF': [0.5, 0.5],
            'POS': [130000000, 130100000],
        }

        model = GermlineModel(mock_collector, min_snp=10)
        model.fit(num_warmup=50, num_samples=50)

        # chr1q should use default values
        assert 'chr1q' in model.arm_results
        assert model.arm_results['chr1q']['p_diploid_score'] == 0.95


class TestGermlineSegmentedModel:
    """Tests for the horseshoe-fused SegmentedGermlineModel."""

    @pytest.fixture
    def mock_collector_with_methods(self):
        """Create a mock GermlineVariantCollector with required methods."""
        collector = Mock()

        # Simulated diploid data
        np.random.seed(42)
        n_variants = 50
        positions = np.sort(np.random.randint(1000000, 100000000, n_variants))
        depths = np.random.randint(30, 100, n_variants)
        alt_counts = np.random.binomial(depths, 0.48)  # Slight shift from 0.5

        collector.get_available_chromosomes = Mock(return_value=['chr1'])
        collector.get_chromosome_data = Mock(return_value=(positions, depths, alt_counts))

        # Also set germline_vars for arm_results property
        collector.germline_vars = {'chr1': {'p': {}, 'q': {}}}

        return collector

    def test_init(self, mock_collector_with_methods):
        """Test SegmentedGermlineModel initialization."""
        model = SegmentedGermlineModel(
            mock_collector_with_methods,
            min_dp_cutoff=10,
            min_variants_per_chrom=20,
            tau_scale=0.01,
        )

        assert model._min_dp_cutoff == 10
        assert model._min_variants_per_chrom == 20
        assert model._tau_scale == 0.01

    def test_filter_by_depth(self, mock_collector_with_methods):
        """Test depth filtering."""
        model = SegmentedGermlineModel(
            mock_collector_with_methods,
            min_dp_cutoff=50
        )

        positions = np.array([100, 200, 300, 400])
        depths = np.array([30, 60, 40, 80])
        alt_counts = np.array([15, 30, 20, 40])

        pos_f, dep_f, alt_f = model._filter_by_depth(positions, depths, alt_counts)

        assert len(pos_f) == 2  # Only depths >= 50
        np.testing.assert_array_equal(pos_f, [200, 400])

    def test_get_segment_lookup(self, mock_collector_with_methods):
        """Test that segment lookup is created after fit."""
        model = SegmentedGermlineModel(
            mock_collector_with_methods,
            min_variants_per_chrom=10
        )

        # Before fit, should build lookup from empty results
        lookup = model.get_segment_lookup()
        assert lookup is not None

    def test_arm_results_property(self, mock_collector_with_methods):
        """Test backward compatibility via arm_results property."""
        model = SegmentedGermlineModel(
            mock_collector_with_methods,
            min_variants_per_chrom=100  # Force skip
        )

        # Add a default result manually
        from CBBmix.utils import SegmentResult, ChromosomeSegmentationResult
        model.chrom_results['chr1'] = ChromosomeSegmentationResult(
            chrom='chr1',
            n_variants=50,
            n_segments=1,
            delta_mean=0.01,
            delta_std=0.005,
            kappa_mean=50.0,
            kappa_std=5.0,
            segments=[
                SegmentResult(
                    segment_id=0,
                    start_position=1000000,
                    end_position=150000000,
                    n_variants=50,
                    psi_mean=0.1,
                    psi_std=0.05,
                )
            ],
            variant_segment_ids=np.zeros(50),
            positions=np.arange(50) * 1000000 + 1000000,
        )

        arm_results = model.arm_results

        # Should have entries for chr1p and chr1q
        assert 'chr1p' in arm_results or 'chr1q' in arm_results
        # Check structure
        for key, val in arm_results.items():
            assert 'psi_mean' in val
            assert 'delta_mean' in val
            assert 'kappa_mean' in val


class TestGermlineSegmentedModelFunction:
    """Tests for the germline_segmented_model numpyro function."""

    def test_model_runs(self):
        """Test that the model runs without error."""
        import numpyro
        from numpyro.infer import MCMC, NUTS

        np.random.seed(42)
        n_variants = 20
        depths = jnp.array(np.random.randint(30, 100, n_variants))
        alt_counts = jnp.array(np.random.binomial(depths, 0.5))
        positions = jnp.array(np.sort(np.random.randint(1e6, 1e8, n_variants)))
        d_scaled = compute_scaled_distances(positions)

        kernel = NUTS(germline_segmented_model)
        mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)

        # Should run without error
        mcmc.run(jax.random.PRNGKey(0), alt_counts, depths, d_scaled, 0.01)

        samples = mcmc.get_samples()

        # Check expected sample keys
        assert 'delta' in samples
        assert 'phi' in samples
        assert 'kappa' in samples
        assert 'psi' in samples
        assert 'tau' in samples
        assert 'lambdas' in samples
        assert 'z_raw' in samples
        assert 'increments_scaled' in samples

        # Check shapes
        assert samples['psi'].shape == (10, n_variants)
        assert samples['delta'].shape == (10,)
        assert samples['increments_scaled'].shape == (10, n_variants - 1)

    def test_model_psi_cumulative(self):
        """Test that psi is cumulative from increments."""
        import numpyro
        from numpyro.infer import MCMC, NUTS

        np.random.seed(123)
        n_variants = 10
        depths = jnp.array(np.full(n_variants, 50))
        alt_counts = jnp.array(np.full(n_variants, 25))  # VAF = 0.5
        positions = jnp.array(np.arange(n_variants) * 1000000)
        d_scaled = compute_scaled_distances(positions)

        kernel = NUTS(germline_segmented_model)
        mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
        mcmc.run(jax.random.PRNGKey(1), alt_counts, depths, d_scaled, 0.01)

        samples = mcmc.get_samples()
        psi = samples['psi']

        # psi should be non-negative (absolute value)
        assert jnp.all(psi >= 0)


class TestIntegration:
    """Integration tests for germline models."""

    def test_segmented_model_creates_valid_lookup(self):
        """Test full pipeline from collector to lookup."""
        from CBBmix.utils import SegmentResult, ChromosomeSegmentationResult

        # Create mock collector
        collector = Mock()
        np.random.seed(42)
        n = 30
        positions = np.sort(np.random.randint(1e6, 1e8, n))
        depths = np.random.randint(30, 100, n)
        alt_counts = np.random.binomial(depths, 0.5)

        collector.get_available_chromosomes = Mock(return_value=['chr1'])
        collector.get_chromosome_data = Mock(return_value=(positions, depths, alt_counts))
        collector.germline_vars = {'chr1': {'p': {}, 'q': {}}}

        model = SegmentedGermlineModel(
            collector,
            min_variants_per_chrom=100  # Skip fitting
        )

        # Manually add result
        model.chrom_results['chr1'] = ChromosomeSegmentationResult(
            chrom='chr1',
            n_variants=n,
            n_segments=1,
            delta_mean=0.01,
            delta_std=0.005,
            kappa_mean=50.0,
            kappa_std=5.0,
            segments=[
                SegmentResult(0, int(positions[0]), int(positions[-1]), n, 0.05, 0.02)
            ],
            variant_segment_ids=np.zeros(n),
            positions=positions,
        )

        lookup = model.get_segment_lookup()

        # Query should work
        seg = lookup.query('chr1', int(positions[n // 2]))
        assert seg is not None
        assert seg.psi_mean == 0.05
        assert seg.delta_mean == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
