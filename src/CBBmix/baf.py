"""Python wrapper for the C++ BAF extension.

Provides ``compute_baf`` and ``compute_baf_genome`` with a graceful fallback
when the compiled extension is unavailable (e.g. htslib not installed).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    from CBBmix._baf import compute_baf as _compute_baf

    HAS_BAF_EXTENSION = True
except ImportError:
    HAS_BAF_EXTENSION = False


def compute_baf(
    bam_path: str,
    ref_path: str,
    region: str,
    min_depth: int = 10,
    min_mapq: int = 20,
    min_baseq: int = 20,
    min_baf: float = 0.2,
    max_baf: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute B-Allele Frequencies from a BAM file for a genomic region.

    Parameters
    ----------
    bam_path : str
        Path to an indexed BAM file.
    ref_path : str
        Path to a reference FASTA with a .fai index.
    region : str
        Genomic region, e.g. ``"chr1"`` or ``"chr1:1000000-2000000"``.
    min_depth : int
        Minimum read depth to consider a position (default: 10).
    min_mapq : int
        Minimum mapping quality filter (default: 20).
    min_baseq : int
        Minimum base quality filter (default: 20).
    min_baf : float
        Minimum BAF threshold to report (default: 0.2).
    max_baf : float
        Maximum BAF threshold to report (default: 0.7).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(positions, bafs)`` — int32 genomic positions and float32 BAF values.

    Raises
    ------
    RuntimeError
        If the C++ extension is not available or files cannot be opened.
    """
    if not HAS_BAF_EXTENSION:
        raise RuntimeError(
            "BAF extension not available. Install htslib and rebuild: "
            "pip install -e ."
        )
    return _compute_baf(
        bam_path, ref_path, region, min_depth, min_mapq, min_baseq, min_baf, max_baf
    )


CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def compute_baf_genome(
    bam_path: str,
    ref_path: str,
    chromosomes: Optional[list] = None,
    min_depth: int = 10,
    min_mapq: int = 20,
    min_baseq: int = 20,
    min_baf: float = 0.2,
    max_baf: float = 0.7,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute BAF across all chromosomes (or a subset).

    Parameters
    ----------
    bam_path : str
        Path to an indexed BAM file.
    ref_path : str
        Path to a reference FASTA with a .fai index.
    chromosomes : list[str] | None
        Chromosomes to process. Defaults to chr1-22, chrX, chrY.
    min_depth : int
        Minimum read depth (default: 10).
    min_mapq : int
        Minimum mapping quality (default: 20).
    min_baseq : int
        Minimum base quality (default: 20).
    min_baf : float
        Minimum BAF to report (default: 0.2).
    max_baf : float
        Maximum BAF to report (default: 0.7).

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Mapping of chromosome name to ``(positions, bafs)``.
    """
    if chromosomes is None:
        chromosomes = CHROMOSOMES

    results = {}
    for chrom in chromosomes:
        positions, bafs = compute_baf(
            bam_path, ref_path, chrom, min_depth, min_mapq, min_baseq, min_baf, max_baf
        )
        results[chrom] = (positions, bafs)
    return results
