import numpy as np
import jax.numpy as jnp
import pytest

from CBBmix.utils import (
    compute_scaled_distances,
    extract_segments_from_posterior,
    prune_and_merge_segments,
    SegmentInfo,
    SegmentResult,
    ChromosomeSegmentationResult,
    SegmentLookup,
    build_somatic_prior_from_germline,
)


class TestComputeScaledDistances:
    """Tests for compute_scaled_distances function."""

    def test_basic_computation(self):
        """Test basic distance scaling."""
        positions = jnp.array([100, 200, 400, 500])
        d_scaled = compute_scaled_distances(positions)

        assert d_scaled.shape == (3,)
        # Distances are 100, 200, 100 -> median is 100
        # Scaled: sqrt(100/100), sqrt(200/100), sqrt(100/100)
        # = 1.0, sqrt(2), 1.0
        np.testing.assert_allclose(d_scaled, [1.0, np.sqrt(2), 1.0], rtol=1e-5)

    def test_uniform_spacing(self):
        """Test with uniformly spaced positions."""
        positions = jnp.array([0, 100, 200, 300, 400])
        d_scaled = compute_scaled_distances(positions)

        # All distances are 100, median is 100
        # All scaled distances should be 1.0
        np.testing.assert_allclose(d_scaled, jnp.ones(4), rtol=1e-5)

    def test_single_position(self):
        """Test with single position returns empty array."""
        positions = jnp.array([100])
        d_scaled = compute_scaled_distances(positions)
        assert len(d_scaled) == 0

    def test_two_positions(self):
        """Test with two positions."""
        positions = jnp.array([100, 200])
        d_scaled = compute_scaled_distances(positions)
        assert d_scaled.shape == (1,)
        # Single distance, median is itself, so scaled = 1.0
        np.testing.assert_allclose(d_scaled, [1.0], rtol=1e-5)

    def test_variable_density(self):
        """Test with variable variant density."""
        # Sparse region followed by dense region
        positions = jnp.array([0, 1000, 1010, 1020, 1030])
        d_scaled = compute_scaled_distances(positions)

        assert d_scaled.shape == (4,)
        # First distance (1000) should be much larger than others (10)
        assert d_scaled[0] > d_scaled[1]


class TestExtractSegmentsFromPosterior:
    """Tests for extract_segments_from_posterior function."""

    def test_single_segment(self):
        """Test case where all increments are small (single segment)."""
        np.random.seed(42)
        positions = np.array([100, 200, 300, 400, 500])
        n_samples = 100
        n_increments = 4
        n_variants = 5

        # Small increments - no breakpoints
        increments_samples = np.random.normal(0, 0.01, (n_samples, n_increments))
        psi_samples = np.abs(np.cumsum(increments_samples, axis=1))
        psi_samples = np.column_stack([np.zeros(n_samples), psi_samples])

        segment_ids, segment_stats = extract_segments_from_posterior(
            positions, increments_samples, psi_samples
        )

        assert segment_ids.shape == (n_variants,)
        assert np.all(segment_ids == 0)  # All in same segment
        assert len(segment_stats) == 1
        assert segment_stats[0]['segment_id'] == 0
        assert segment_stats[0]['n_variants'] == 5

    def test_multiple_segments(self):
        """Test case with clear breakpoint."""
        np.random.seed(42)
        positions = np.array([100, 200, 300, 400, 500])
        n_samples = 100
        n_increments = 4

        # Large increment at position 2 (between 300 and 400)
        increments_samples = np.random.normal(0, 0.01, (n_samples, n_increments))
        increments_samples[:, 2] = 0.5  # Clear breakpoint

        psi_samples = np.abs(np.cumsum(increments_samples, axis=1))
        psi_samples = np.column_stack([np.zeros(n_samples), psi_samples])

        segment_ids, segment_stats = extract_segments_from_posterior(
            positions, increments_samples, psi_samples,
            increment_threshold=0.1
        )

        # Should have 2 segments
        assert len(segment_stats) == 2
        assert segment_ids[0] == 0  # First 3 variants in segment 0
        assert segment_ids[3] == 1  # Last 2 variants in segment 1

    def test_segment_stats_content(self):
        """Test that segment stats contain expected keys."""
        positions = np.array([100, 200, 300])
        increments_samples = np.zeros((50, 2))
        psi_samples = np.zeros((50, 3))

        _, segment_stats = extract_segments_from_posterior(
            positions, increments_samples, psi_samples
        )

        required_keys = [
            'segment_id', 'start_position', 'end_position',
            'n_variants', 'psi_mean', 'psi_std', 'psi_ci_low', 'psi_ci_high'
        ]
        for key in required_keys:
            assert key in segment_stats[0]


class TestSegmentLookup:
    """Tests for SegmentLookup class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample ChromosomeSegmentationResult."""
        segments = [
            SegmentResult(
                segment_id=0,
                start_position=0,
                end_position=100000000,
                n_variants=50,
                psi_mean=0.05,
                psi_std=0.02,
            ),
            SegmentResult(
                segment_id=1,
                start_position=100000001,
                end_position=200000000,
                n_variants=30,
                psi_mean=0.3,
                psi_std=0.05,
            ),
        ]
        return ChromosomeSegmentationResult(
            chrom='chr1',
            n_variants=80,
            n_segments=2,
            delta_mean=0.01,
            delta_std=0.005,
            kappa_mean=50.0,
            kappa_std=5.0,
            segments=segments,
            variant_segment_ids=np.array([0]*50 + [1]*30),
            positions=np.arange(80) * 2500000,
        )

    def test_add_and_query(self, sample_result):
        """Test adding chromosome and querying."""
        lookup = SegmentLookup()
        lookup.add_chromosome('chr1', sample_result)

        # Query first segment
        seg = lookup.query('chr1', 50000000)
        assert seg is not None
        assert seg.segment_id == 0
        assert seg.psi_mean == 0.05
        assert seg.delta_mean == 0.01
        assert seg.kappa_mean == 50.0

        # Query second segment
        seg = lookup.query('chr1', 150000000)
        assert seg is not None
        assert seg.segment_id == 1
        assert seg.psi_mean == 0.3

    def test_query_unknown_chromosome(self):
        """Test query returns None for unknown chromosome."""
        lookup = SegmentLookup()
        assert lookup.query('chr1', 100) is None

    def test_query_batch(self, sample_result):
        """Test batch query."""
        lookup = SegmentLookup()
        lookup.add_chromosome('chr1', sample_result)

        chroms = np.array(['chr1', 'chr1', 'chr2'])
        positions = np.array([50000000, 150000000, 100])

        psi, delta, kappa = lookup.query_batch(chroms, positions)

        assert len(psi) == 3
        np.testing.assert_allclose(psi[0], 0.05, rtol=1e-5)
        np.testing.assert_allclose(psi[1], 0.3, rtol=1e-5)
        np.testing.assert_allclose(psi[2], 0.0, rtol=1e-5)  # unknown chrom default

        np.testing.assert_allclose(delta[0], 0.01, rtol=1e-5)
        np.testing.assert_allclose(kappa[0], 50.0, rtol=1e-5)

    def test_get_chromosomes(self, sample_result):
        """Test getting list of chromosomes."""
        lookup = SegmentLookup()
        lookup.add_chromosome('chr1', sample_result)

        chroms = lookup.get_chromosomes()
        assert 'chr1' in chroms

    def test_get_chromosome_result(self, sample_result):
        """Test retrieving full result."""
        lookup = SegmentLookup()
        lookup.add_chromosome('chr1', sample_result)

        result = lookup.get_chromosome_result('chr1')
        assert result is not None
        assert result.chrom == 'chr1'
        assert result.n_segments == 2


class TestBuildSomaticPriorFromGermline:
    """Tests for build_somatic_prior_from_germline convenience function."""

    def test_basic_usage(self):
        """Test basic usage with SegmentLookup."""
        segments = [
            SegmentResult(
                segment_id=0, start_position=0, end_position=1e9,
                n_variants=100, psi_mean=0.1, psi_std=0.05
            )
        ]
        result = ChromosomeSegmentationResult(
            chrom='chr1', n_variants=100, n_segments=1,
            delta_mean=0.02, delta_std=0.01,
            kappa_mean=60.0, kappa_std=10.0,
            segments=segments,
            variant_segment_ids=np.zeros(100),
            positions=np.arange(100) * 1000000,
        )

        lookup = SegmentLookup()
        lookup.add_chromosome('chr1', result)

        chroms = np.array(['chr1', 'chr1'])
        positions = np.array([100, 50000000])

        psi, delta, kappa = build_somatic_prior_from_germline(
            lookup, chroms, positions
        )

        assert len(psi) == 2
        np.testing.assert_allclose(psi, [0.1, 0.1], rtol=1e-5)
        np.testing.assert_allclose(delta, [0.02, 0.02], rtol=1e-5)
        np.testing.assert_allclose(kappa, [60.0, 60.0], rtol=1e-5)


class TestDataclasses:
    """Tests for dataclass definitions."""

    def test_segment_info(self):
        """Test SegmentInfo dataclass."""
        seg = SegmentInfo(
            chrom='chr1', segment_id=0, start=0, end=1e8,
            psi_mean=0.1, psi_std=0.05, delta_mean=0.01, kappa_mean=50.0
        )
        assert seg.chrom == 'chr1'
        assert seg.psi_mean == 0.1

    def test_segment_result(self):
        """Test SegmentResult dataclass."""
        seg = SegmentResult(
            segment_id=0, start_position=0, end_position=1e8,
            n_variants=100, psi_mean=0.1, psi_std=0.05
        )
        assert seg.segment_id == 0
        assert seg.n_variants == 100

    def test_chromosome_segmentation_result(self):
        """Test ChromosomeSegmentationResult dataclass."""
        result = ChromosomeSegmentationResult(
            chrom='chr1', n_variants=100, n_segments=2,
            delta_mean=0.01, delta_std=0.005,
            kappa_mean=50.0, kappa_std=5.0,
        )
        assert result.chrom == 'chr1'
        assert result.n_segments == 2
        assert len(result.segments) == 0  # default empty


class TestPruneAndMergeSegments:
    """Tests for prune_and_merge_segments function."""

    def _make_segments_and_ids(self, specs):
        """Helper: specs is list of (start, end, n_variants, psi_mean).
        Returns segments, variant_segment_ids, positions."""
        segments = []
        all_positions = []
        all_ids = []
        for i, (start, end, n, psi) in enumerate(specs):
            segments.append(SegmentResult(
                segment_id=i,
                start_position=start,
                end_position=end,
                n_variants=n,
                psi_mean=psi,
                psi_std=0.01,
            ))
            pos = np.linspace(start, end, n, dtype=int)
            all_positions.append(pos)
            all_ids.append(np.full(n, i, dtype=np.int32))

        positions = np.concatenate(all_positions)
        variant_segment_ids = np.concatenate(all_ids)
        return segments, variant_segment_ids, positions

    def test_no_pruning_needed(self):
        """All segments >= min_variants: output unchanged."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 1000, 10, 0.1),
            (1001, 2000, 8, 0.2),
            (2001, 3000, 15, 0.3),
        ])
        centromere = 50000  # all on p arm

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        assert len(new_segs) == 3
        assert set(new_ids) == {0, 1, 2}

    def test_single_small_merged_into_neighbor(self):
        """Small segment between two large ones merges into closer psi."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 1000, 20, 0.1),
            (1001, 1100, 3, 0.12),
            (1101, 3000, 25, 0.5),
        ])
        centromere = 50000

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        assert len(new_segs) == 2
        # Small seg (psi=0.12) should merge into seg 0 (psi=0.1), not seg 2 (psi=0.5)
        assert new_segs[0].n_variants == 23
        assert new_segs[1].n_variants == 25

    def test_merge_by_psi_similarity(self):
        """Small segment merges into neighbor with most similar psi."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 1000, 20, 0.1),
            (1001, 1100, 2, 0.45),
            (1101, 3000, 25, 0.5),
        ])
        centromere = 50000

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        assert len(new_segs) == 2
        # Small seg (psi=0.45) should merge into seg 2 (psi=0.5), not seg 0 (psi=0.1)
        assert new_segs[0].n_variants == 20
        assert new_segs[1].n_variants == 27

    def test_arm_boundary_respected(self):
        """Small p-arm segment cannot merge into q-arm neighbor."""
        # centromere at 1500: seg0 on p, seg1 on q
        segments, ids, positions = self._make_segments_and_ids([
            (1000, 1400, 3, 0.1),   # p arm, small
            (1600, 3000, 20, 0.1),  # q arm, large
        ])
        centromere = 1500

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        # Cannot merge across arm boundary
        assert len(new_segs) == 2

    def test_chain_of_small_segments(self):
        """Multiple consecutive small segments all absorbed into the large one."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 100, 2, 0.1),
            (101, 200, 3, 0.12),
            (201, 300, 2, 0.11),
            (301, 3000, 30, 0.15),
        ])
        centromere = 50000

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        assert len(new_segs) == 1
        assert new_segs[0].n_variants == 37

    def test_empty_segments(self):
        """Empty list returns as-is."""
        new_segs, new_ids = prune_and_merge_segments(
            [], np.array([], dtype=np.int32), np.array([]), centromere_pos=50000,
        )
        assert len(new_segs) == 0
        assert len(new_ids) == 0

    def test_single_segment(self):
        """Single segment returns as-is regardless of size."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 100, 2, 0.1),
        ])

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere_pos=50000, min_variants=5,
        )

        assert len(new_segs) == 1
        assert new_segs[0].n_variants == 2

    def test_reindexing_contiguous(self):
        """After merge, segment IDs are contiguous 0..K-1."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 1000, 20, 0.1),
            (1001, 1100, 2, 0.12),
            (1101, 2000, 15, 0.5),
            (2001, 2100, 3, 0.48),
            (2101, 3000, 25, 0.3),
        ])
        centromere = 50000

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        # IDs should be 0, 1, ..., K-1
        expected_ids = set(range(len(new_segs)))
        assert set(s.segment_id for s in new_segs) == expected_ids
        assert set(new_ids) == expected_ids

    def test_psi_samples_recomputation(self):
        """With psi_samples, merged stats match raw recomputation."""
        np.random.seed(42)
        segments, ids, positions = self._make_segments_and_ids([
            (0, 1000, 20, 0.1),
            (1001, 1100, 3, 0.12),
            (1101, 3000, 25, 0.5),
        ])
        n_total = 20 + 3 + 25
        n_samples = 100
        # Create psi_samples with known structure
        psi_samples = np.random.rand(n_samples, n_total) * 0.1
        # Set segment 2 psi higher
        psi_samples[:, 23:] = np.random.rand(n_samples, 25) * 0.5 + 0.3
        centromere = 50000

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere,
            min_variants=5, psi_samples=psi_samples,
        )

        assert len(new_segs) == 2
        # Verify merged segment stats from raw posterior
        merged_mask = new_ids == 0
        expected_psi_per_sample = np.mean(psi_samples[:, merged_mask], axis=1)
        np.testing.assert_allclose(
            new_segs[0].psi_mean, np.mean(expected_psi_per_sample), rtol=1e-5,
        )
        np.testing.assert_allclose(
            new_segs[0].psi_std, np.std(expected_psi_per_sample), rtol=1e-5,
        )

    def test_variant_segment_ids_consistency(self):
        """After merge, each segment ID maps to correct variant count."""
        segments, ids, positions = self._make_segments_and_ids([
            (0, 1000, 20, 0.1),
            (1001, 1100, 3, 0.12),
            (1101, 3000, 25, 0.5),
        ])
        centromere = 50000

        new_segs, new_ids = prune_and_merge_segments(
            segments, ids, positions, centromere, min_variants=5,
        )

        for seg in new_segs:
            count = np.sum(new_ids == seg.segment_id)
            assert count == seg.n_variants, (
                f"Segment {seg.segment_id}: n_variants={seg.n_variants} "
                f"but found {count} in variant_segment_ids"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
