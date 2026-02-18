#!/usr/bin/env python
"""Compute BAF from a BAM + FASTA, then fit the Beta-Binomial HMM."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from CBBmix.baf import compute_baf
from CBBmix.hmm import BetaBinomialHMM
from CBBmix.plotting import plot_baf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bam", help="Indexed BAM file")
    parser.add_argument("ref", help="Reference FASTA with .fai index")
    parser.add_argument("region", help='Genomic region, e.g. "chr1" or "chr1:1-1000000"')
    parser.add_argument("--min-depth", type=int, default=10)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=2)
    args = parser.parse_args()

    # 1. Compute BAF
    positions, depth, alt_depth = compute_baf(
        args.bam, args.ref, args.region, min_depth=args.min_depth,
    )
    print(f"Region {args.region}: {len(positions)} het sites found")
    fig, ax = plt.subplots(figsize=(10, 1))
    ax = plot_baf(positions, depth, alt_depth)
    fig.tight_layout()
    fig.savefig(f"{args.region}_baf.png", dpi=300, bbox_inches='tight')

    if len(positions) < 10:
        print("Too few sites – skipping HMM fit.")
        return

    # 2. Fit HMM
    hmm = BetaBinomialHMM(
        n_states=args.n_states,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    hmm.fit(positions, depth, alt_depth)
    hmm.summary()

    # 3. Decode states
    states = hmm.decode(positions, depth, alt_depth)
    params = hmm.get_posterior_params()

    print("\nPosterior mu per state:", params["mu"])
    print("Posterior kappa per state:", params["kappa"])
    print("\nState counts:", dict(zip(*np.unique(states, return_counts=True))))


if __name__ == "__main__":
    main()
