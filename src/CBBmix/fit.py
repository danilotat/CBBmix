#!/usr/bin/env python3
"""
CBBmix Full Inference Pipeline

This script runs the complete CBBmix pipeline:
1. Parse VCF for germline and somatic variants
2. Fit germline model per chromosome arm (estimate delta, kappa, psi)
3. Fit somatic Pitman-Yor mixture model (cluster by Cellular Prevalence)
4. Output results to files

Usage:
    python fit.py --vcf input.vcf.gz --output-dir results/
    python fit.py --vcf input.vcf.gz --output-dir results/ --num-samples 2000
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import jax

from .vcf import GermlineVariantCollector, SomaticVariantCollector
from .germline import GermlineModel
from .somatic import SomaticModel, SomaticPriorConfig

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure logging to file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO

    log_file = output_dir / "cbbmix.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CBBmix: 3-component Beta-Binomial mixture model for clonal structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--vcf",
        type=str,
        required=True,
        help="Path to input VCF file (must have hetProb and somProb INFO fields)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files",
    )

    # Variant filtering
    parser.add_argument(
        "--min-depth",
        type=int,
        default=10,
        help="Minimum read depth for variants",
    )
    parser.add_argument(
        "--af-low",
        type=float,
        default=0.25,
        help="Lower AF threshold for germline heterozygotes",
    )
    parser.add_argument(
        "--af-high",
        type=float,
        default=0.75,
        help="Upper AF threshold for germline heterozygotes",
    )
    parser.add_argument(
        "--min-snp-per-arm",
        type=int,
        default=10,
        help="Minimum SNPs per arm for germline fitting",
    )

    # MCMC parameters
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=500,
        help="Number of MCMC warmup iterations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of MCMC posterior samples",
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=1,
        help="Number of MCMC chains (somatic model only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Somatic model priors
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=10,
        help="Maximum number of somatic clusters (truncation level K)",
    )
    parser.add_argument(
        "--alpha-py",
        type=float,
        default=1.0,
        help="Pitman-Yor concentration parameter",
    )
    parser.add_argument(
        "--theta-py",
        type=float,
        default=0.1,
        help="Pitman-Yor discount parameter (0=Dirichlet Process)",
    )
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=0.1,
        help="Scale for per-variant ASE noise",
    )

    # Inference options
    parser.add_argument(
        "--skip-somatic",
        action="store_true",
        help="Skip somatic model fitting (germline only)",
    )
    parser.add_argument(
        "--use-full-model",
        action="store_true",
        help="Use non-marginalized somatic model (slower but samples z, h)",
    )

    # Misc
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def save_germline_results(
    germline_model: GermlineModel,
    output_dir: Path,
) -> None:
    """Save germline model results to files."""
    # Save arm-level summary
    records = []
    for arm_key, results in germline_model.arm_results.items():
        records.append({
            "arm": arm_key,
            "chrom": arm_key[:-1],
            "arm_type": arm_key[-1],
            **results,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "germline_arms.csv", index=False)
    logging.info(f"Saved germline arm results to {output_dir / 'germline_arms.csv'}")

    # Save raw variant data
    germline_model.data_df.to_csv(
        output_dir / "germline_variants.csv", index=False
    )
    logging.info(f"Saved germline variants to {output_dir / 'germline_variants.csv'}")


def save_somatic_results(
    somatic_model: SomaticModel,
    output_dir: Path,
) -> None:
    """Save somatic model results to files."""
    # Save clustering summary
    if somatic_model.clustering_results is not None:
        somatic_model.clustering_results.to_csv(
            output_dir / "somatic_clusters.csv", index=False
        )
        logging.info(f"Saved cluster summary to {output_dir / 'somatic_clusters.csv'}")

    # Save variant data with cluster assignments
    df = somatic_model.data_df.copy()
    assignments = somatic_model.get_cluster_assignments()
    if assignments is not None:
        df["cluster"] = assignments

        # Add cluster CP (rho) for each variant
        if somatic_model.clustering_results is not None:
            cluster_rho = dict(zip(
                somatic_model.clustering_results["cluster_id"],
                somatic_model.clustering_results["rho_mean"],
            ))
            df["cluster_rho"] = df["cluster"].map(cluster_rho)

    df.to_csv(output_dir / "somatic_variants.csv", index=False)
    logging.info(f"Saved somatic variants to {output_dir / 'somatic_variants.csv'}")

    # Save posterior samples (compressed)
    if somatic_model.samples is not None:
        np.savez_compressed(
            output_dir / "somatic_posterior.npz",
            **{k: np.array(v) for k, v in somatic_model.samples.items()},
        )
        logging.info(f"Saved posterior samples to {output_dir / 'somatic_posterior.npz'}")


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Save run configuration for reproducibility."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "vcf": str(args.vcf),
        "parameters": {
            "min_depth": args.min_depth,
            "af_low": args.af_low,
            "af_high": args.af_high,
            "min_snp_per_arm": args.min_snp_per_arm,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "seed": args.seed,
            "max_clusters": args.max_clusters,
            "alpha_py": args.alpha_py,
            "theta_py": args.theta_py,
            "sigma_scale": args.sigma_scale,
        },
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def run_pipeline(args: argparse.Namespace) -> int:
    """
    Execute the full CBBmix inference pipeline.

    Returns exit code (0 for success, 1 for failure).
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir, args.verbose)
    save_run_config(args, output_dir)

    logging.info("=" * 60)
    logging.info("CBBmix Inference Pipeline")
    logging.info("=" * 60)
    logging.info(f"Input VCF: {args.vcf}")
    logging.info(f"Output directory: {output_dir}")

    # =========================================================
    # Step 1: Collect variants from VCF
    # =========================================================
    logging.info("\n[Step 1/4] Collecting variants from VCF...")

    try:
        germline_collector = GermlineVariantCollector(
            args.vcf,
            af_thresholds=[args.af_low, args.af_high],
        )
        n_germ_arms = sum(
            len(arms) for arms in germline_collector.germline_vars.values()
        )
        logging.info(f"  Germline: found variants across {n_germ_arms} chromosome arms")
    except Exception as e:
        logging.error(f"Failed to collect germline variants: {e}")
        return 1

    if not args.skip_somatic:
        try:
            somatic_collector = SomaticVariantCollector(args.vcf)
            n_som_vars = sum(
                sum(len(arm_data.get("DP", [])) for arm_data in arms.values())
                for arms in somatic_collector.somatic_vars.values()
            )
            logging.info(f"  Somatic: found {n_som_vars} variants")
        except Exception as e:
            logging.error(f"Failed to collect somatic variants: {e}")
            return 1

    # =========================================================
    # Step 2: Fit germline model
    # =========================================================
    logging.info("\n[Step 2/4] Fitting germline model (per chromosome arm)...")

    try:
        germline_model = GermlineModel(
            germline_collector,
            min_dp_cutoff=args.min_depth,
            min_snp=args.min_snp_per_arm,
        )

        logging.info(f"  Variants after depth filter: {len(germline_model.data_df)}")

        germline_model.fit(
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
        )

        save_germline_results(germline_model, output_dir)

        # Print summary
        logging.info("\n  Germline arm summary:")
        for arm_key, res in germline_model.arm_results.items():
            logging.info(
                f"    {arm_key}: psi={res['psi_mean']:.4f}±{res['psi_std']:.4f}, "
                f"delta={res['delta_mean']:.4f}, kappa={res['kappa_mean']:.1f}, "
                f"p_diploid={res['p_diploid_score']:.2f}"
            )
    except Exception as e:
        logging.error(f"Germline model fitting failed: {e}")
        raise

    if args.skip_somatic:
        logging.info("\n[Skipping somatic model as requested]")
        logging.info("\nPipeline complete (germline only).")
        return 0

    # =========================================================
    # Step 3: Fit somatic model
    # =========================================================
    logging.info("\n[Step 3/4] Fitting somatic Pitman-Yor mixture model...")

    try:
        prior_config = SomaticPriorConfig(
            alpha_py=args.alpha_py,
            theta_py=args.theta_py,
            max_clusters=args.max_clusters,
            sigma_scale=args.sigma_scale,
        )

        somatic_model = SomaticModel(
            somatic_collector,
            germline_model,
            prior_config=prior_config,
            min_dp_cutoff=args.min_depth,
        )

        logging.info(f"  Variants after depth filter: {len(somatic_model.data_df)}")

        if len(somatic_model.data_df) == 0:
            logging.warning("  No somatic variants found. Skipping somatic fitting.")
        else:
            somatic_model.fit(
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                seed=args.seed,
                use_marginalized=not args.use_full_model,
            )

            save_somatic_results(somatic_model, output_dir)

    except Exception as e:
        logging.error(f"Somatic model fitting failed: {e}")
        raise

    # =========================================================
    # Step 4: Summary
    # =========================================================
    logging.info("\n[Step 4/4] Pipeline complete!")
    logging.info(f"\nOutput files saved to: {output_dir}")
    logging.info("  - config.json: run configuration")
    logging.info("  - cbbmix.log: full log")
    logging.info("  - germline_arms.csv: per-arm germline parameters")
    logging.info("  - germline_variants.csv: germline variant data")
    if not args.skip_somatic and len(somatic_model.data_df) > 0:
        logging.info("  - somatic_clusters.csv: identified clonal populations")
        logging.info("  - somatic_variants.csv: variants with cluster assignments")
        logging.info("  - somatic_posterior.npz: posterior samples")

    logging.info("\n" + "=" * 60)

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user.")
        return 130
    except Exception as e:
        logging.exception(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
