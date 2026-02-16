import numpy as np
import jax.numpy as jnp
import pytest
from unittest.mock import Mock

from CBBmix.somatic import SomaticModel, SomaticPriorConfig
from CBBmix.germline import GermlineModel, SegmentedGermlineModel
from CBBmix.utils import SegmentLookup, SegmentResult, ChromosomeSegmentationResult


class TestSomaticPriorConfig:
    """Tests for SomaticPriorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SomaticPriorConfig()

        assert config.alpha_py == 1.0
        assert config.theta_py == 0.1
        assert config.max_clusters == 10
        assert config.sigma_scale == 0.1
        assert config.rho_alpha == 1.0
        assert config.rho_beta == 1.0
        assert config.max_kappa == 200.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = SomaticPriorConfig(
            alpha_py=2.0,
            theta_py=0.2,
            max_clusters=5,
        )

        assert config.alpha_py == 2.0
        assert config.theta_py == 0.2
        assert config.max_clusters == 5


class TestSomaticModel:
    """Tests for SomaticModel."""

    @pytest.fixture
    def mock_somatic_collector(self):
        """Create a mock SomaticVariantCollector."""
        collector = Mock()
        np.random.seed(42)

        n_variants = 30
        depths = list(np.random.randint(50, 150, n_variants))
        alt_depths = list(np.random.binomial(depths, 0.3))
        vafs = [a / d for a, d in zip(alt_depths, depths)]
        positions = list(range(50000000, 50000000 + n_variants * 100000, 100000))

        collector.somatic_vars = {
            'chr1': {
                'p': {
                    'DP': depths,
                    'alt_DP': alt_depths,
                    'VAF': vafs,
                    'POS': positions,
                }
            }
        }
        return collector

    @pytest.fixture
    def mock_germline_model(self):
        """Create a mock GermlineModel."""
        model = Mock(spec=GermlineModel)
        model.arm_results = {
            'chr1p': {
                'delta_mean': 0.01,
                'kappa_mean': 50.0,
                'psi_mean': 0.1,
            },
            'chr1q': {
                'delta_mean': 0.01,
                'kappa_mean': 50.0,
                'psi_mean': 0.05,
            }
        }
        return model

    @pytest.fixture
    def mock_segmented_germline_model(self):
        """Create a mock SegmentedGermlineModel."""
        model = Mock(spec=SegmentedGermlineModel)

        # Setup arm_results for compatibility
        model.arm_results = {
            'chr1p': {
                'delta_mean': 0.01,
                'kappa_mean': 50.0,
                'psi_mean': 0.1,
            }
        }

        # Setup segment lookup
        lookup = SegmentLookup()
        segments = [
            SegmentResult(
                segment_id=0,
                start_position=0,
                end_position=100000000,
                n_variants=50,
                psi_mean=0.1,
                psi_std=0.02,
            ),
            SegmentResult(
                segment_id=1,
                start_position=100000001,
                end_position=200000000,
                n_variants=30,
                psi_mean=0.25,
                psi_std=0.05,
            ),
        ]
        result = ChromosomeSegmentationResult(
            chrom='chr1',
            n_variants=80,
            n_segments=2,
            delta_mean=0.01,
            delta_std=0.005,
            kappa_mean=50.0,
            kappa_std=5.0,
            segments=segments,
            variant_segment_ids=np.array([0]*50 + [1]*30),
            positions=np.arange(80) * 2000000,
        )
        lookup.add_chromosome('chr1', result)

        model.get_segment_lookup = Mock(return_value=lookup)

        return model

    def test_init_with_germline_model(self, mock_somatic_collector, mock_germline_model):
        """Test initialization with arm-level GermlineModel."""
        model = SomaticModel(
            mock_somatic_collector,
            mock_germline_model,
            min_dp_cutoff=10,
        )

        assert model._use_segments is False
        assert model.segment_lookup is None
        assert not model.data_df.empty

    def test_init_with_segmented_germline_model(
        self, mock_somatic_collector, mock_segmented_germline_model
    ):
        """Test initialization with SegmentedGermlineModel."""
        model = SomaticModel(
            mock_somatic_collector,
            mock_segmented_germline_model,
            min_dp_cutoff=10,
        )

        assert model._use_segments is True
        assert model.segment_lookup is not None
        assert not model.data_df.empty

    def test_preprocess_data(self, mock_somatic_collector, mock_germline_model):
        """Test data preprocessing with arm-level model."""
        model = SomaticModel(
            mock_somatic_collector,
            mock_germline_model,
            min_dp_cutoff=10,
        )

        df = model.data_df

        assert 'depth' in df.columns
        assert 'alt_count' in df.columns
        assert 'arm_delta' in df.columns
        assert 'arm_kappa' in df.columns
        assert 'arm_psi' in df.columns

        # Check values are assigned from germline
        assert df['arm_delta'].iloc[0] == 0.01
        assert df['arm_kappa'].iloc[0] == 50.0
        assert df['arm_psi'].iloc[0] == 0.1

    def test_preprocess_data_segmented(
        self, mock_somatic_collector, mock_segmented_germline_model
    ):
        """Test data preprocessing with segmented model."""
        model = SomaticModel(
            mock_somatic_collector,
            mock_segmented_germline_model,
            min_dp_cutoff=10,
        )

        df = model.data_df

        assert 'depth' in df.columns
        assert 'position' in df.columns
        assert 'arm_delta' in df.columns
        assert 'arm_kappa' in df.columns
        assert 'arm_psi' in df.columns

        # Values should come from segment lookup
        assert df['arm_delta'].iloc[0] == 0.01
        assert df['arm_kappa'].iloc[0] == 50.0
        # psi should match segment (first segment has psi=0.1)
        assert df['arm_psi'].iloc[0] == 0.1

    def test_depth_filtering(self, mock_somatic_collector, mock_germline_model):
        """Test that depth filtering is applied."""
        # Add some low-depth variants
        mock_somatic_collector.somatic_vars['chr1']['p']['DP'].extend([5, 8, 7])
        mock_somatic_collector.somatic_vars['chr1']['p']['alt_DP'].extend([2, 4, 3])
        mock_somatic_collector.somatic_vars['chr1']['p']['VAF'].extend([0.4, 0.5, 0.43])
        mock_somatic_collector.somatic_vars['chr1']['p']['POS'].extend([60000000, 60100000, 60200000])

        model = SomaticModel(
            mock_somatic_collector,
            mock_germline_model,
            min_dp_cutoff=10,
        )

        # Low-depth variants should be filtered
        assert all(model.data_df['depth'] >= 10)

    def test_fit_empty_data(self, mock_germline_model):
        """Test fitting with empty data."""
        empty_collector = Mock()
        empty_collector.somatic_vars = {}

        model = SomaticModel(
            empty_collector,
            mock_germline_model,
            min_dp_cutoff=10,
        )

        # Should not raise
        model.fit(num_warmup=10, num_samples=10)

        assert model.samples is None

    def test_kappa_capping(self, mock_somatic_collector, mock_germline_model):
        """Test that kappa is capped at max_kappa."""
        mock_germline_model.arm_results['chr1p']['kappa_mean'] = 500.0

        config = SomaticPriorConfig(max_kappa=100.0)
        model = SomaticModel(
            mock_somatic_collector,
            mock_germline_model,
            prior_config=config,
        )

        # Kappa should be capped
        assert all(model.data_df['arm_kappa'] <= 100.0)


class TestSomaticWithPrunedSegments:
    """Tests that somatic model correctly uses pruned/merged segments."""

    @pytest.fixture
    def pruned_segment_lookup(self):
        """Create a SegmentLookup from segments that look like post-pruning output.

        Simulates: original 4 segments, small ones merged, leaving 2 with a gap.
        Segment 0: positions ~10M-50M  (diploid, psi~0)
        Segment 1: positions ~60M-120M (imbalanced, psi~0.3)
        """
        lookup = SegmentLookup()
        segments = [
            SegmentResult(
                segment_id=0,
                start_position=10_000_000,
                end_position=50_000_000,
                n_variants=40,
                psi_mean=0.02,
                psi_std=0.01,
            ),
            SegmentResult(
                segment_id=1,
                start_position=60_000_000,
                end_position=120_000_000,
                n_variants=35,
                psi_mean=0.30,
                psi_std=0.04,
            ),
        ]
        result = ChromosomeSegmentationResult(
            chrom='chr1',
            n_variants=75,
            n_segments=2,
            delta_mean=0.008,
            delta_std=0.003,
            kappa_mean=55.0,
            kappa_std=4.0,
            segments=segments,
            variant_segment_ids=np.array([0]*40 + [1]*35),
            positions=np.concatenate([
                np.linspace(10e6, 50e6, 40, dtype=int),
                np.linspace(60e6, 120e6, 35, dtype=int),
            ]),
        )
        lookup.add_chromosome('chr1', result)
        return lookup, result

    def test_somatic_picks_up_pruned_psi(self, pruned_segment_lookup):
        """Somatic variants get correct psi from merged segments."""
        lookup, _ = pruned_segment_lookup

        # Variant in segment 0 (diploid region)
        seg = lookup.query('chr1', 30_000_000)
        assert seg is not None
        assert seg.psi_mean == pytest.approx(0.02)
        assert seg.delta_mean == pytest.approx(0.008)
        assert seg.kappa_mean == pytest.approx(55.0)

        # Variant in segment 1 (imbalanced region)
        seg = lookup.query('chr1', 90_000_000)
        assert seg is not None
        assert seg.psi_mean == pytest.approx(0.30)

    def test_somatic_variant_in_gap_between_merged_segments(self, pruned_segment_lookup):
        """Somatic variant in gap (55M) between merged segments maps to nearest."""
        lookup, _ = pruned_segment_lookup

        # Position 55M is between segment 0 (end=50M) and segment 1 (start=60M)
        seg = lookup.query('chr1', 55_000_000)
        assert seg is not None
        # bisect_right on [10M, 60M] for 55M -> index 1 -> idx=0 -> segment 0
        assert seg.segment_id == 0

    def test_somatic_preprocessing_with_pruned_segments(self, pruned_segment_lookup):
        """Full preprocessing flow with pruned segments assigns correct per-variant params."""
        lookup, _ = pruned_segment_lookup

        segmented_model = Mock(spec=SegmentedGermlineModel)
        segmented_model.arm_results = {
            'chr1p': {
                'delta_mean': 0.008,
                'kappa_mean': 55.0,
                'psi_mean': 0.02,
            }
        }
        segmented_model.get_segment_lookup = Mock(return_value=lookup)

        collector = Mock()
        np.random.seed(42)
        # 5 variants in segment 0 region, 5 in segment 1 region
        positions_seg0 = [20_000_000, 25_000_000, 30_000_000, 35_000_000, 40_000_000]
        positions_seg1 = [70_000_000, 80_000_000, 90_000_000, 100_000_000, 110_000_000]
        all_pos = positions_seg0 + positions_seg1
        n = len(all_pos)
        depths = list(np.random.randint(50, 100, n))
        alt = list(np.random.binomial(depths, 0.3))

        collector.somatic_vars = {
            'chr1': {
                'p': {
                    'DP': depths,
                    'alt_DP': alt,
                    'VAF': [a / d for a, d in zip(alt, depths)],
                    'POS': all_pos,
                }
            }
        }

        model = SomaticModel(collector, segmented_model, min_dp_cutoff=1)
        df = model.data_df

        assert len(df) == n
        # First 5 should have psi from segment 0
        np.testing.assert_allclose(df['arm_psi'].values[:5], 0.02)
        # Last 5 should have psi from segment 1
        np.testing.assert_allclose(df['arm_psi'].values[5:], 0.30)
        # All should share same delta and kappa
        np.testing.assert_allclose(df['arm_delta'].values, 0.008)
        np.testing.assert_allclose(df['arm_kappa'].values, 55.0)

    def test_batch_query_with_pruned_segments(self, pruned_segment_lookup):
        """query_batch returns correct arrays for mixed positions."""
        lookup, _ = pruned_segment_lookup

        chroms = np.array(['chr1', 'chr1', 'chr1', 'chr2'])
        positions = np.array([30_000_000, 90_000_000, 55_000_000, 100])

        psi, delta, kappa = lookup.query_batch(chroms, positions)

        np.testing.assert_allclose(psi[0], 0.02)   # segment 0
        np.testing.assert_allclose(psi[1], 0.30)   # segment 1
        np.testing.assert_allclose(psi[2], 0.02)   # gap -> segment 0
        np.testing.assert_allclose(psi[3], 0.0)    # unknown chrom
        np.testing.assert_allclose(delta[:3], 0.008)
        np.testing.assert_allclose(kappa[:3], 55.0)
        np.testing.assert_allclose(kappa[3], 10.0)  # default


class TestSomaticModelIntegration:
    """Integration tests for SomaticModel with both germline types."""

    @pytest.fixture
    def somatic_collector(self):
        """Create somatic collector with realistic data."""
        collector = Mock()
        np.random.seed(42)

        # Generate variants from two clusters
        n1, n2 = 20, 15
        depths = np.random.randint(50, 150, n1 + n2)

        # Cluster 1: CP ~ 0.3, cluster 2: CP ~ 0.6
        alt1 = np.random.binomial(depths[:n1], 0.3)
        alt2 = np.random.binomial(depths[n1:], 0.6)
        alt = np.concatenate([alt1, alt2])

        collector.somatic_vars = {
            'chr1': {
                'p': {
                    'DP': list(depths),
                    'alt_DP': list(alt),
                    'VAF': [a / d for a, d in zip(alt, depths)],
                    'POS': list(range(50000000, 50000000 + (n1 + n2) * 100000, 100000)),
                }
            }
        }
        return collector

    def test_backward_compatibility(self, somatic_collector):
        """Test that arm-level GermlineModel still works."""
        germline = Mock(spec=GermlineModel)
        germline.arm_results = {
            'chr1p': {
                'delta_mean': 0.01,
                'kappa_mean': 50.0,
                'psi_mean': 0.05,
            }
        }

        model = SomaticModel(somatic_collector, germline)

        assert model._use_segments is False
        assert len(model.data_df) > 0
        assert model.data_df['arm_psi'].iloc[0] == 0.05

    def test_segmented_integration(self, somatic_collector):
        """Test that SegmentedGermlineModel is detected and used."""
        segmented = Mock(spec=SegmentedGermlineModel)
        segmented.arm_results = {
            'chr1p': {
                'delta_mean': 0.01,
                'kappa_mean': 50.0,
                'psi_mean': 0.15,
            }
        }

        lookup = SegmentLookup()
        segments = [
            SegmentResult(0, 0, int(1e9), 100, 0.15, 0.03)
        ]
        result = ChromosomeSegmentationResult(
            'chr1', 100, 1, 0.01, 0.005, 50.0, 5.0,
            segments=segments,
            variant_segment_ids=np.zeros(100),
            positions=np.arange(100) * 1000000,
        )
        lookup.add_chromosome('chr1', result)
        segmented.get_segment_lookup = Mock(return_value=lookup)

        model = SomaticModel(somatic_collector, segmented)

        assert model._use_segments is True
        assert model.segment_lookup is not None
        # psi should come from segment lookup
        assert model.data_df['arm_psi'].iloc[0] == 0.15


class TestModelMethods:
    """Tests for SomaticModel methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing methods."""
        collector = Mock()
        np.random.seed(42)
        n = 20
        depths = list(np.random.randint(50, 100, n))
        alt = list(np.random.binomial(depths, 0.4))

        collector.somatic_vars = {
            'chr1': {
                'p': {
                    'DP': depths,
                    'alt_DP': alt,
                    'VAF': [a / d for a, d in zip(alt, depths)],
                    'POS': list(range(50000000, 50000000 + n * 100000, 100000)),
                }
            }
        }

        germline = Mock(spec=GermlineModel)
        germline.arm_results = {
            'chr1p': {
                'delta_mean': 0.01,
                'kappa_mean': 50.0,
                'psi_mean': 0.05,
            }
        }

        model = SomaticModel(collector, germline)
        return model

    def test_print_summary_before_fit(self, fitted_model):
        """Test that print_summary handles unfitted model."""
        # Should not raise
        fitted_model.print_summary()

    def test_get_cluster_assignments_before_fit(self, fitted_model):
        """Test that get_cluster_assignments returns None before fit."""
        result = fitted_model.get_cluster_assignments()
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
