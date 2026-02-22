#!/usr/bin/env python
"""Compute per-gene BAF from a BAM + FASTA + GTF, then fit the Gene-Clustered HMM."""

import argparse
import re
import numpy as np
from CBBmix.baf import compute_baf
from CBBmix.hmm import GeneClusteredHMM
from CBBmix.genepool import GTF_record


def parse_region(region: str):
    """Parse 'chr1' or 'chr1:100-2000' into (chrom, start, end)."""
    m = re.match(r"^([^:]+):(\d+)-(\d+)$", region)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return region, None, None


def collect_genes_in_region(gtf_path, chrom, start, end):
    """Parse GTF for gene features overlapping the target region, sorted by start."""
    genes = []
    with open(gtf_path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9:
                continue
            rec = GTF_record(*fields)
            if rec.feature_type != "gene":
                continue
            if rec.chromosome != chrom:
                continue
            if start is not None and rec.end < start:
                continue
            if end is not None and rec.start > end:
                continue
            name = rec.attributes.get("gene_name", rec.attributes.get("gene_id", f"gene_{rec.start}"))
            genes.append((name, rec.chromosome, rec.start, rec.end))
    genes.sort(key=lambda g: g[2])
    return genes


def build_gene_arrays(genes, bam, ref, min_depth):
    """
    Compute BAF per gene and build the flat arrays needed by GeneClusteredHMM.

    Returns (positions, depth, alt_depth, gene_indices, gene_centers, gene_names)
    or None if too few genes have het sites.
    """
    positions_list = []
    depth_list = []
    alt_list = []
    gene_indices_list = []
    gene_centers = []
    gene_names = []
    g_counter = 0

    for name, chrom, start, end in genes:
        region = f"{chrom}:{start}-{end}"
        pos, dep, alt = compute_baf(bam, ref, region, min_depth=min_depth)
        if len(pos) == 0:
            continue
        positions_list.append(pos)
        depth_list.append(dep)
        alt_list.append(alt)
        gene_indices_list.append(np.full(len(pos), g_counter, dtype=np.int32))
        gene_centers.append((start + end) / 2.0)
        gene_names.append(name)
        g_counter += 1

    if g_counter < 3:
        return None

    return (
        np.concatenate(positions_list),
        np.concatenate(depth_list),
        np.concatenate(alt_list),
        np.concatenate(gene_indices_list),
        np.array(gene_centers, dtype=np.float64),
        gene_names,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bam", help="Indexed BAM file")
    parser.add_argument("ref", help="Reference FASTA with .fai index")
    parser.add_argument("gtf", help="GTF annotation file (uncompressed)")
    parser.add_argument("region", help='Genomic region, e.g. "chr1" or "chr1:1-1000000"')
    parser.add_argument("--min-depth", type=int, default=10)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=2)
    args = parser.parse_args()

    chrom, start, end = parse_region(args.region)

    # 1. Collect genes from GTF
    genes = collect_genes_in_region(args.gtf, chrom, start, end)
    print(f"Region {args.region}: {len(genes)} genes found in GTF")

    if len(genes) < 3:
        print("Too few genes – aborting.")
        return

    # 2. Compute per-gene BAF and build arrays
    result = build_gene_arrays(genes, args.bam, args.ref, args.min_depth)
    if result is None:
        print("Too few genes with het sites – aborting.")
        return

    positions, depth, alt_depth, gene_indices, gene_centers, gene_names = result
    n_snps = len(positions)
    n_genes = len(gene_centers)
    print(f"  {n_snps} het SNPs across {n_genes} genes with data")

    # 3. Fit GeneClusteredHMM
    hmm = GeneClusteredHMM(
        n_states=args.n_states,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    hmm.fit(positions, depth, alt_depth, gene_indices, gene_centers)
    hmm.summary()

    # 4. Decode per-gene states
    gene_states = hmm.decode(positions, depth, alt_depth, gene_indices, gene_centers)
    params = hmm.get_posterior_params()

    print("\nPosterior mu per state:", params["mu"])
    print("Posterior kappa per state:", params["kappa"])
    print("\nState counts:", dict(zip(*np.unique(gene_states, return_counts=True))))

    # 5. Map back to SNP-level for inspection
    snp_states = gene_states[gene_indices]
    print(f"\nSNP-level state assignments: {len(snp_states)} sites")

    # 6. Per-gene summary
    print(f"\n{'Gene':<20s} {'SNPs':>5s} {'State':>5s}")
    print("-" * 32)
    snp_counts = np.bincount(gene_indices, minlength=n_genes)
    for i, name in enumerate(gene_names):
        print(f"{name:<20s} {snp_counts[i]:5d} {gene_states[i]:5d}")


if __name__ == "__main__":
    main()
