"""
Utility functions and data classes for CBBmix.

This module provides:
- Distance computation for horseshoe-fused segmentation
- Segment extraction from posterior samples
- SegmentLookup for O(log N) position-to-segment mapping
- Data classes for segment results
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import bisect


def compute_scaled_distances(positions: jnp.ndarray) -> jnp.ndarray:
    """
    Compute scaled distances between consecutive variants.

    The scaling uses sqrt(diff / median_distance) to normalize
    for variable variant density along the genome.

    Parameters
    ----------
    positions : jnp.ndarray
        Sorted genomic positions of shape (N,)

    Returns
    -------
    d_scaled : jnp.ndarray
        Scaled distances of shape (N-1,)
    """
    if len(positions) < 2:
        return jnp.array([])

    diffs = jnp.diff(positions).astype(jnp.float64)
    # Avoid division by zero
    median_dist = jnp.maximum(jnp.median(diffs), 1.0)
    d_scaled = jnp.sqrt(diffs / median_dist)

    return d_scaled


def extract_segments_from_posterior(
    positions: np.ndarray,
    increments_samples: np.ndarray,
    psi_samples: np.ndarray,
    ci_level: float = 0.95,
    increment_threshold: float = 0.1,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract segment boundaries from posterior increment samples.

    A segment boundary is placed where the posterior probability
    that |increment| > threshold exceeds (1 - ci_level).

    Parameters
    ----------
    positions : np.ndarray
        Sorted genomic positions of shape (N,)
    increments_samples : np.ndarray
        Posterior increment samples of shape (n_samples, N-1)
    psi_samples : np.ndarray
        Posterior psi samples of shape (n_samples, N)
    ci_level : float
        Credible interval level for determining significant increments
    increment_threshold : float
        Threshold for considering an increment as a breakpoint

    Returns
    -------
    segment_ids : np.ndarray
        Segment assignment for each variant of shape (N,)
    segment_stats : list of dict
        Per-segment statistics including psi mean/std, start/end positions
    """
    n_samples, n_increments = increments_samples.shape
    n_variants = len(positions)

    if n_increments != n_variants - 1:
        raise ValueError(
            f"Increment shape {increments_samples.shape} doesn't match "
            f"positions shape {positions.shape}"
        )

    # Compute probability that |increment| > threshold for each position
    prob_breakpoint = np.mean(
        np.abs(increments_samples) > increment_threshold, axis=0
    )

    # Find breakpoints where probability exceeds (1 - ci_level)
    breakpoint_mask = prob_breakpoint > (1 - ci_level)
    breakpoint_indices = np.where(breakpoint_mask)[0]

    # Build segment assignments
    segment_ids = np.zeros(n_variants, dtype=np.int32)
    current_segment = 0
    segment_starts = [0]

    for bp_idx in breakpoint_indices:
        # Breakpoint at index i means boundary between variant i and i+1
        current_segment += 1
        segment_ids[bp_idx + 1:] = current_segment
        segment_starts.append(bp_idx + 1)

    segment_starts.append(n_variants)
    n_segments = current_segment + 1

    # Compute per-segment statistics
    segment_stats = []
    for seg_id in range(n_segments):
        seg_mask = segment_ids == seg_id
        seg_positions = positions[seg_mask]
        seg_psi = psi_samples[:, seg_mask]

        # Mean psi across variants in segment, then across samples
        psi_per_sample = np.mean(seg_psi, axis=1)

        # P(diploid) = fraction of posterior samples where segment psi < 0.05
        p_diploid = float(np.mean(psi_per_sample < 0.05))

        segment_stats.append({
            'segment_id': seg_id,
            'start_position': int(seg_positions[0]),
            'end_position': int(seg_positions[-1]),
            'n_variants': int(np.sum(seg_mask)),
            'psi_mean': float(np.mean(psi_per_sample)),
            'psi_std': float(np.std(psi_per_sample)),
            'psi_ci_low': float(np.percentile(psi_per_sample, (1 - ci_level) / 2 * 100)),
            'psi_ci_high': float(np.percentile(psi_per_sample, (1 + ci_level) / 2 * 100)),
            'p_diploid': p_diploid,
        })

    return segment_ids, segment_stats


def prune_and_merge_segments(
    segments: List[SegmentResult],
    variant_segment_ids: np.ndarray,
    positions: np.ndarray,
    centromere_pos: int,
    min_variants: int = 5,
    psi_samples: Optional[np.ndarray] = None,
) -> Tuple[List[SegmentResult], np.ndarray]:
    """
    Prune small segments by merging them into adjacent neighbors.

    Small segments (fewer than min_variants) are iteratively absorbed
    into the adjacent segment with the most similar psi_mean, provided
    both segments lie on the same chromosome arm (p or q).

    Parameters
    ----------
    segments : List[SegmentResult]
        Segment results from extract_segments_from_posterior
    variant_segment_ids : np.ndarray
        Per-variant segment assignments of shape (N,)
    positions : np.ndarray
        Sorted genomic positions of shape (N,)
    centromere_pos : int
        Centromere position separating p and q arms
    min_variants : int
        Minimum number of variants for a segment to survive
    psi_samples : Optional[np.ndarray]
        Posterior psi samples of shape (n_samples, N) for recomputing
        merged segment statistics. If None, uses weighted average.

    Returns
    -------
    segments : List[SegmentResult]
        Pruned and merged segments
    variant_segment_ids : np.ndarray
        Updated per-variant segment assignments (contiguous 0..K-1)
    """
    if len(segments) <= 1:
        return segments, variant_segment_ids

    # Work on copies
    segments = list(segments)
    variant_segment_ids = variant_segment_ids.copy()

    def _arm_label(seg: SegmentResult) -> str:
        midpoint = (seg.start_position + seg.end_position) // 2
        return 'p' if midpoint < centromere_pos else 'q'

    def _merge_into(target_idx: int, small_idx: int) -> None:
        """Merge segments[small_idx] into segments[target_idx]."""
        target = segments[target_idx]
        small = segments[small_idx]

        # Reassign variant IDs
        old_id = small.segment_id
        new_id = target.segment_id
        variant_segment_ids[variant_segment_ids == old_id] = new_id

        # Recompute stats
        mask = variant_segment_ids == new_id
        merged_positions = positions[mask]

        if psi_samples is not None:
            psi_per_sample = np.mean(psi_samples[:, mask], axis=1)
            new_psi_mean = float(np.mean(psi_per_sample))
            new_psi_std = float(np.std(psi_per_sample))
            new_p_diploid = float(np.mean(psi_per_sample < 0.05))
        else:
            total_n = target.n_variants + small.n_variants
            new_psi_mean = (
                target.psi_mean * target.n_variants
                + small.psi_mean * small.n_variants
            ) / total_n
            new_psi_std = (
                target.psi_std * target.n_variants
                + small.psi_std * small.n_variants
            ) / total_n
            new_p_diploid = (
                target.p_diploid * target.n_variants
                + small.p_diploid * small.n_variants
            ) / total_n

        segments[target_idx] = SegmentResult(
            segment_id=new_id,
            start_position=int(merged_positions[0]),
            end_position=int(merged_positions[-1]),
            n_variants=int(np.sum(mask)),
            psi_mean=new_psi_mean,
            psi_std=new_psi_std,
            p_diploid=new_p_diploid,
        )
        segments.pop(small_idx)

    # Iterative greedy loop
    changed = True
    while changed:
        changed = False
        # Sort segments by start_position to maintain spatial order
        segments.sort(key=lambda s: s.start_position)

        for i, seg in enumerate(segments):
            if seg.n_variants >= min_variants:
                continue

            arm = _arm_label(seg)
            candidates = []

            # Previous neighbor
            if i > 0 and _arm_label(segments[i - 1]) == arm:
                candidates.append((i - 1, abs(segments[i - 1].psi_mean - seg.psi_mean)))

            # Next neighbor
            if i < len(segments) - 1 and _arm_label(segments[i + 1]) == arm:
                candidates.append((i + 1, abs(segments[i + 1].psi_mean - seg.psi_mean)))

            if not candidates:
                continue

            # Pick neighbor with most similar psi; tie-break by larger n_variants
            best = min(
                candidates,
                key=lambda c: (c[1], -segments[c[0]].n_variants),
            )
            target_idx = best[0]

            # Adjust index if target is after small (removing small shifts indices)
            if target_idx > i:
                _merge_into(target_idx, i)
            else:
                _merge_into(target_idx, i)

            changed = True
            break  # Restart loop after mutation

    # Re-index segment IDs contiguously
    segments.sort(key=lambda s: s.start_position)
    old_to_new = {}
    for new_id, seg in enumerate(segments):
        old_to_new[seg.segment_id] = new_id
        segments[new_id] = SegmentResult(
            segment_id=new_id,
            start_position=seg.start_position,
            end_position=seg.end_position,
            n_variants=seg.n_variants,
            psi_mean=seg.psi_mean,
            psi_std=seg.psi_std,
            p_diploid=seg.p_diploid,
        )

    new_ids = np.empty_like(variant_segment_ids)
    for old_id, new_id in old_to_new.items():
        new_ids[variant_segment_ids == old_id] = new_id
    variant_segment_ids = new_ids

    return segments, variant_segment_ids


@dataclass
class SegmentInfo:
    """Information about a single genomic segment."""
    chrom: str
    segment_id: int
    start: int
    end: int
    psi_mean: float
    psi_std: float
    delta_mean: float
    kappa_mean: float
    n_variants: int = 0
    p_diploid: float = 0.0


@dataclass
class SegmentResult:
    """Result for a single segment within a chromosome."""
    segment_id: int
    start_position: int
    end_position: int
    n_variants: int
    psi_mean: float
    psi_std: float
    p_diploid: float = 0.0


@dataclass
class ChromosomeSegmentationResult:
    """Result of segmentation for a single chromosome."""
    chrom: str
    n_variants: int
    n_segments: int
    delta_mean: float
    delta_std: float
    kappa_mean: float
    kappa_std: float
    segments: List[SegmentResult] = field(default_factory=list)
    variant_segment_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    # Store full posterior samples for downstream use
    psi_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_samples: np.ndarray = field(default_factory=lambda: np.array([]))


class SegmentLookup:
    """
    O(log N) lookup for mapping genomic positions to segments.

    This class provides efficient position-to-segment mapping
    for use by the somatic model.
    """

    def __init__(self):
        self._chrom_data: Dict[str, Dict] = {}

    def add_chromosome(
        self,
        chrom: str,
        result: ChromosomeSegmentationResult,
    ) -> None:
        """
        Add segmentation result for a chromosome.

        Parameters
        ----------
        chrom : str
            Chromosome name
        result : ChromosomeSegmentationResult
            Segmentation result containing segments and positions
        """
        # Build segment boundary list for binary search
        boundaries = []
        segment_infos = []

        for seg in result.segments:
            boundaries.append(seg.start_position)
            segment_infos.append(SegmentInfo(
                chrom=chrom,
                segment_id=seg.segment_id,
                start=seg.start_position,
                end=seg.end_position,
                psi_mean=seg.psi_mean,
                psi_std=seg.psi_std,
                delta_mean=result.delta_mean,
                kappa_mean=result.kappa_mean,
                n_variants=seg.n_variants,
                p_diploid=seg.p_diploid,
            ))

        self._chrom_data[chrom] = {
            'boundaries': boundaries,
            'segments': segment_infos,
            'result': result,
        }

    def query(self, chrom: str, position: int) -> Optional[SegmentInfo]:
        """
        Query segment for a single position.

        Parameters
        ----------
        chrom : str
            Chromosome name
        position : int
            Genomic position

        Returns
        -------
        SegmentInfo or None
            Segment information if found, None otherwise
        """
        if chrom not in self._chrom_data:
            return None

        data = self._chrom_data[chrom]
        boundaries = data['boundaries']
        segments = data['segments']

        if not boundaries:
            return None

        # Binary search for the segment containing this position
        idx = bisect.bisect_right(boundaries, position) - 1

        if idx < 0:
            # Position before first segment - use first segment
            idx = 0
        elif idx >= len(segments):
            # Position after last segment - use last segment
            idx = len(segments) - 1

        seg = segments[idx]

        # Check if position is within segment bounds (with some tolerance)
        if position < seg.start or position > seg.end:
            # Position falls in a gap - return nearest segment
            pass

        return seg

    def query_batch(
        self,
        chroms: np.ndarray,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Query segments for multiple positions.

        Parameters
        ----------
        chroms : np.ndarray
            Chromosome names
        positions : np.ndarray
            Genomic positions

        Returns
        -------
        psi : np.ndarray
            Segment psi values
        delta : np.ndarray
            Chromosome delta values
        kappa : np.ndarray
            Chromosome kappa values
        """
        n = len(chroms)
        psi = np.zeros(n, dtype=np.float64)
        delta = np.zeros(n, dtype=np.float64)
        kappa = np.full(n, 10.0, dtype=np.float64)  # default kappa

        for i, (c, p) in enumerate(zip(chroms, positions)):
            seg = self.query(c, p)
            if seg is not None:
                psi[i] = seg.psi_mean
                delta[i] = seg.delta_mean
                kappa[i] = seg.kappa_mean
            else:
                # Use defaults for missing segments
                psi[i] = 0.0
                delta[i] = 0.0
                kappa[i] = 10.0

        return psi, delta, kappa

    def get_chromosomes(self) -> List[str]:
        """Return list of chromosomes with segmentation data."""
        return list(self._chrom_data.keys())

    def get_chromosome_result(self, chrom: str) -> Optional[ChromosomeSegmentationResult]:
        """Get the full segmentation result for a chromosome."""
        if chrom in self._chrom_data:
            return self._chrom_data[chrom]['result']
        return None


def build_somatic_prior_from_germline(
    segment_lookup: SegmentLookup,
    chroms: np.ndarray,
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build somatic priors from germline segmentation results.

    This is a convenience function that wraps SegmentLookup.query_batch.

    Parameters
    ----------
    segment_lookup : SegmentLookup
        Lookup table with germline segmentation results
    chroms : np.ndarray
        Chromosome names for somatic variants
    positions : np.ndarray
        Genomic positions for somatic variants

    Returns
    -------
    psi : np.ndarray
        Per-variant psi values from germline segments
    delta : np.ndarray
        Per-variant delta values (chromosome-level)
    kappa : np.ndarray
        Per-variant kappa values (chromosome-level)
    """
    return segment_lookup.query_batch(chroms, positions)
