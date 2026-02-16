import numpy as np
import pandas as pd
import jax
import logging
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from dataclasses import dataclass
from numpyro.infer import MCMC, NUTS, init_to_value
from typing import Optional, Dict, List, Union
from .vcf import GermlineVariantCollector
from .utils import (
    compute_scaled_distances,
    extract_segments_from_posterior,
    prune_and_merge_segments,
    SegmentLookup,
    SegmentResult,
    ChromosomeSegmentationResult,
)

jax.config.update("jax_enable_x64", True)

@dataclass
class GermlineArmParameters:
    p_diploid_score: float = 0.95
    psi_mean: float = 0.0
    psi_std: float = 0.1
    delta_mean: float = 0.005
    delta_std: float = 0.005
    kappa_mean: float = 10.0
    kappa_std: float = 2.0

class GermlineModel:
    """
    This class holds a Beta-Binomial arm-level mixture model fitted using
    germline variants across chromosome arms.

    For each arm, heterozygous germline variants are modeled under two hypotheses:
    - H_0 (balanced): BAF centered around 0.5
    - H_1 (imbalanced): BAF follows a symmetric mixture around (0.5 − δ, 0.5 + δ)

    The inference estimates the probability that each arm is diploid.
    """

    def __init__(self, germline_collector_data: GermlineVariantCollector,
                 arm_priors: Optional[GermlineArmParameters] = None, 
                 min_dp_cutoff=10, min_snp=10):
        self._min_dp_cutoff = min_dp_cutoff
        self._min_snp = min_snp
        self.raw_data = germline_collector_data
        self.data_df = self.preprocess_data()
        self.arm_results = {}
        self._default_germ_priors = GermlineArmParameters if not arm_priors else arm_priors
        
    @property
    def arm_diploidity(self) -> pd.DataFrame:
        if len(self.arm_results) == 0:
            raise ValueError(f"The model should be fitted before returning arm diploidity probabilities.")
        else:
            arms, p, kappa_mean = [], [], []
            for arm, vals in self.arm_results.items():
                arms.append(arm)
                p.append(vals.p_diploid_score)
                kappa_mean.append(vals.kappa_mean)
            return pd.DataFrame({
                'arm': arms,
                'p_diploid': p,
                'mean_precision': kappa_mean
            })

    def preprocess_data(self):
        records = []
        for chrom, arms in self.raw_data.germline_vars.items():
            for arm in arms:
                arm_data = self.raw_data.germline_vars[chrom][arm]
                if not arm_data:
                    logging.warning(
                        f"No heterozygous variants found for arm {arm} of chromosome {chrom}"
                    )
                    continue
                
                dps = arm_data['DP']
                alt_dps = arm_data['alt_DP']
                vafs = arm_data['VAF']
                positions = arm_data['POS']
                for d, ad, v, pos in zip(dps, alt_dps, vafs, positions):
                    records.append({
                        'chrom': chrom,
                        'arm': arm,
                        'depth': int(d),
                        'alt_count': int(ad),
                        'vaf': float(v),
                        'pos': int(pos),
                    })

        df = pd.DataFrame(records)
        df = df[df['depth'] >= self._min_dp_cutoff]
        return df
    

    def _model_single_arm(self, depth, alt_count):
        n_variants = depth.shape[0]

        # 1. PRIORS
        
        # Relaxed Reference Bias: Allow up to ~5-10% bias
        delta = numpyro.sample("delta", dist.HalfNormal(0.05))

        # Better Overdispersion Prior: 
        # Use LogNormal to avoid phi near 0 (which causes infinite precision)
        # This encourages phi to be around 0.1 to 1.0, keeping kappa reasonable (10 to 100 range)
        phi = numpyro.sample("phi", dist.LogNormal(loc=jnp.log(0.1), scale=1.0))
        kappa = numpyro.deterministic("kappa", 1.0 / phi + 1.0)

        # 2. HYPOTHESIS MIXTURE (The Fix for Diploidy)
        
        # Prior probability of the arm being diploid (e.g., 50% or 95%)
        # We infer this, or we can fix it if we want a strict hypothesis test.
        rho_diploid = numpyro.sample("rho_diploid", dist.Beta(2, 2))
        
        # Magnitude of imbalance IF it exists (The Slab)
        # We name it raw because it only applies if H1 is true
        psi_imbalanced = numpyro.sample("psi_slab", dist.HalfNormal(0.5))

        # 3. LIKELIHOOD CALCULATION
        
        # Base Logits (Ref bias + Noise)
        # Note: I removed the StudentT per-variant noise (sigma) 
        # because BetaBinomial (kappa) already handles overdispersion. 
        # Having both creates identifiability issues and slows convergence.
        eta_base = -delta 

        # --- Pathway A: H0 (Diploid) ---
        # psi = 0
        mu_diploid = jax.nn.sigmoid(eta_base)
        conc = kappa - 1
        
        log_prob_h0 = dist.BetaBinomial(
            concentration1=mu_diploid * conc,
            concentration0=(1 - mu_diploid) * conc,
            total_count=depth
        ).log_prob(alt_count).sum()

        # --- Pathway B: H1 (Imbalanced/Aneuploid) ---
        # psi = psi_imbalanced
        eta_down = eta_base - psi_imbalanced
        eta_up   = eta_base + psi_imbalanced
        
        mu_down = jax.nn.sigmoid(eta_down)
        mu_up   = jax.nn.sigmoid(eta_up)
        
        # Clip for stability
        mu_down = jnp.clip(mu_down, 1e-6, 1 - 1e-6)
        mu_up   = jnp.clip(mu_up, 1e-6, 1 - 1e-6)

        # Mixture of Up/Down shifts (Unknown phasing)
        lp_down = dist.BetaBinomial(
            concentration1=mu_down * conc, 
            concentration0=(1 - mu_down) * conc, 
            total_count=depth
        ).log_prob(alt_count)
        
        lp_up = dist.BetaBinomial(
            concentration1=mu_up * conc, 
            concentration0=(1 - mu_up) * conc, 
            total_count=depth
        ).log_prob(alt_count)
        
        # Sum over variants for the H1 likelihood
        log_prob_h1 = (jnp.logaddexp(lp_down, lp_up) - jnp.log(2.0)).sum()

        # 4. FINAL MARGINALIZATION
        # We mix H0 and H1 weights in log-space
        # P(Data) = rho * P(Data|H0) + (1-rho) * P(Data|H1)
        
        total_log_prob = jnp.logaddexp(
            jnp.log(rho_diploid) + log_prob_h0,
            jnp.log(1.0 - rho_diploid) + log_prob_h1
        )
        
        numpyro.factor("obs_mixture", total_log_prob)
        
        # Track the actual posterior probability of diploidy for this arm
        # This is calculated via Bayes rule: P(H0 | Data)
        log_posterior_odds = (jnp.log(rho_diploid) + log_prob_h0) - total_log_prob
        numpyro.deterministic("p_diploid_posterior", jnp.exp(log_posterior_odds))

    def fit(self, num_warmup=500, num_samples=1000, **kwargs):
        """
        Fit the model independently for each chromosome arm.
        """
        logging.info("Starting Germline Fit (Arm-by-Arm)...")
        unique_arms = self.data_df[['chrom', 'arm']].drop_duplicates()
        for _, row in unique_arms.iterrows():
            chrom, arm = row['chrom'], row['arm']
            arm_key = f"{chrom}{arm}"
            subset = self.data_df[
                (self.data_df['chrom'] == chrom) &
                (self.data_df['arm'] == arm)
            ]
            if len(subset) <= self._min_snp:
                logging.warning(
                    f"Skipping {arm_key}: not enough variants ({len(subset)})"
                )
                self.arm_results[arm_key] = self._default_germ_priors
                continue

            logging.info(f"Fitting {arm_key} ({len(subset)} variants)...")
            logging.info(f"Across arm {arm_key} of {chrom} we have a mean VAF of {np.mean(subset['vaf'])} ")
            depth = jnp.array(subset['depth'].values)
            alt = jnp.array(subset['alt_count'].values)
            kernel = NUTS(self._model_single_arm)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, **kwargs)
            mcmc.run(jax.random.PRNGKey(42), depth, alt)
            samples = mcmc.get_samples()
            p_diploid_post = samples['p_diploid_posterior']
            psi_post = samples['psi_slab']
            delta_post = samples['delta']
            kappa_post = samples['kappa']

            self.arm_results[arm_key] = GermlineArmParameters(
                p_diploid_score=float(jnp.mean(p_diploid_post)),
                psi_mean=float(jnp.mean(psi_post)),
                psi_std=float(jnp.std(psi_post)),
                delta_mean=float(jnp.mean(delta_post)),
                kappa_mean=float(jnp.mean(kappa_post)),
                kappa_std=float(jnp.std(kappa_post)),
            )
        logging.info("Germline Fit Complete.")
    
    def fit(self, num_warmup=500, num_samples=1000, **kwargs):
        logging.info("Starting Germline Fit (Arm-by-Arm)...")
        init_values = {
            "kappa": 50.0,
            "delta": 0.001,
        }
        psi_grid = jnp.linspace(0.025, 0.50, 30)
        psi_prior = dist.TruncatedNormal(low=0.02, loc=0.1, scale=0.2)
        log_prior_psi = psi_prior.log_prob(psi_grid)

        unique_arms = self.data_df[['chrom', 'arm']].drop_duplicates()
        for _, row in unique_arms.iterrows():
            chrom, arm = row['chrom'], row['arm']
            arm_key = f"{chrom}{arm}"
            subset = self.data_df[
                (self.data_df['chrom'] == chrom) &
                (self.data_df['arm'] == arm)
            ]
            if len(subset) <= self._min_snp:
                logging.warning(f"Skipping {arm_key}: not enough variants ({len(subset)})")
                self.arm_results[arm_key] = self._default_germ_priors
                continue

            logging.info(f"Fitting {arm_key} ({len(subset)} variants)...")
            depth = jnp.array(subset['depth'].values)
            alt = jnp.array(subset['alt_count'].values)

            kernel = NUTS(self._model_single_arm, init_strategy=init_to_value(values=init_values))
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, **kwargs)
            mcmc.run(jax.random.PRNGKey(42), depth, alt)
            samples = mcmc.get_samples()

            p_diploid_post = samples['p_diploid_posterior']
            delta_post = samples['delta']
            kappa_post = samples['kappa']

            # Recover psi posterior: for each MCMC draw, compute
            # p(psi_j | delta_i, kappa_i, data) over the grid, then take the expectation
            def _psi_posterior_mean(delta_i, kappa_i):
                eta = -delta_i
                conc = jnp.maximum(kappa_i - 1.0, 0.1)

                def _h1_loglik(psi):
                    mu_d = jnp.clip(jax.nn.sigmoid(eta - psi), 1e-4, 1 - 1e-4)
                    mu_u = jnp.clip(jax.nn.sigmoid(eta + psi), 1e-4, 1 - 1e-4)
                    lp_d = dist.BetaBinomial(mu_d * conc, (1 - mu_d) * conc, total_count=depth).log_prob(alt)
                    lp_u = dist.BetaBinomial(mu_u * conc, (1 - mu_u) * conc, total_count=depth).log_prob(alt)
                    return (jnp.logaddexp(lp_d, lp_u) - jnp.log(2.0)).sum()

                log_liks = jax.vmap(_h1_loglik)(psi_grid)
                log_weights = log_liks + log_prior_psi
                log_weights = log_weights - jax.nn.logsumexp(log_weights)  # normalize
                weights = jnp.exp(log_weights)
                return jnp.sum(weights * psi_grid)

            psi_means = jax.vmap(_psi_posterior_mean)(delta_post, kappa_post)

            self.arm_results[arm_key] = GermlineArmParameters(
                p_diploid_score=float(jnp.mean(p_diploid_post)),
                psi_mean=float(jnp.mean(psi_means)),
                psi_std=float(jnp.std(psi_means)),
                delta_mean=float(jnp.mean(delta_post)),
                delta_std=float(jnp.std(delta_post)),
                kappa_mean=float(jnp.mean(kappa_post)),
                kappa_std=float(jnp.std(kappa_post)),
            )

        logging.info("Germline Fit Complete.")



def germline_segmented_model(alt_counts, total_counts, d_scaled=None, tau_scale=0.01):
    """
    Horseshoe-fused segmentation model for per-chromosome germline variants.

    This model replaces the arm-level psi with per-variant psi_i, imposing
    piecewise constancy via a horseshoe prior on positional increments.

    Parameters
    ----------
    alt_counts : jnp.ndarray
        Alternate allele counts of shape (N,)
    total_counts : jnp.ndarray
        Total read depths of shape (N,)
    d_scaled : jnp.ndarray
        Scaled inter-variant distances of shape (N-1,)
    tau_scale : float
        Scale for the global shrinkage parameter tau

    Model specification:
    - delta ~ HalfNormal(0.01)  # global reference bias
    - phi ~ Exp(1) -> kappa = 1/phi + 1  # overdispersion
    - tau ~ HalfCauchy(tau_scale)  # global shrinkage
    - lambdas ~ HalfCauchy(1)^(N-1)  # local shrinkage per increment
    - z_raw ~ Normal(0,1)^(N-1)  # non-centered increments
    - psi_0 ~ Normal(0, 0.5)  # anchor for first variant
    - sigma ~ Normal(0, 0.1)^N  # per-variant ASE

    Derived:
    - increments = z_raw * tau * lambdas * d_scaled
    - psi = |cumsum([psi_0] + increments)|  # absolute for imbalance magnitude

    Likelihood: 50/50 mixture of BetaBinomials for unknown phasing
    """
    n_variants = alt_counts.shape[0]
    n_increments = n_variants - 1

    # Global reference bias (small, shifts AF down from 0.5)
    delta = numpyro.sample("delta", dist.HalfNormal(0.01))

    # Overdispersion via phi -> kappa reparameterization
    phi = numpyro.sample("phi", dist.Exponential(1.0))
    kappa = numpyro.deterministic("kappa", 1.0 / phi + 1.0)

    # Horseshoe prior on increments
    # Global shrinkage - controls overall sparsity
    tau = numpyro.sample("tau", dist.HalfCauchy(tau_scale))

    # Local shrinkage per increment (non-centered for better sampling)
    with numpyro.plate("increments", n_increments):
        lambdas = numpyro.sample("lambdas", dist.HalfCauchy(1.0))
        z_raw = numpyro.sample("z_raw", dist.Normal(0.0, 1.0))

    # Compute increments with distance scaling
    # Larger distances allow larger jumps
    if d_scaled is None:
        increments = z_raw * tau * lambdas
    else:
        increments = z_raw * tau * lambdas * d_scaled
    numpyro.deterministic("increments_scaled", increments)

    # Anchor for cumulative sum (first variant's psi before taking absolute)
    psi_0 = numpyro.sample("psi_0", dist.Normal(0.0, 0.5))

    # Build cumulative psi (raw, can be positive or negative)
    psi_raw = jnp.concatenate([jnp.array([psi_0]), psi_0 + jnp.cumsum(increments)])

    # Take absolute value - we model magnitude of imbalance
    psi = numpyro.deterministic("psi", jnp.abs(psi_raw))

    #NOTE: simplification by removing sigma
    # # Per-variant allele-specific expression noise
    # with numpyro.plate("variants", n_variants):
    #     sigma = numpyro.sample("sigma", dist.Normal(0.0, 0.1))

    # # Compute log-odds (eta) for each variant
    # # Base: diploid het has logit(0.5)=0, shifted by -delta (ref bias) + sigma (ASE)
    eta_base = -delta # + sigma
    eta_down = eta_base - psi
    eta_up = eta_base + psi

    # Convert to mean via inverse logit (sigmoid)
    mu_down = jax.nn.sigmoid(eta_down)
    mu_up = jax.nn.sigmoid(eta_up)

    # Clip for numerical stability
    mu_down = jnp.clip(mu_down, 1e-6, 1 - 1e-6)
    mu_up = jnp.clip(mu_up, 1e-6, 1 - 1e-6)

    # Beta-Binomial parameterization: alpha = mu*(kappa-1), beta = (1-mu)*(kappa-1)
    conc = kappa - 1

    # Log-likelihood for down-shifted component
    log_prob_down = dist.BetaBinomial(
        concentration1=mu_down * conc,
        concentration0=(1 - mu_down) * conc,
        total_count=total_counts
    ).log_prob(alt_counts)

    # Log-likelihood for up-shifted component
    log_prob_up = dist.BetaBinomial(
        concentration1=mu_up * conc,
        concentration0=(1 - mu_up) * conc,
        total_count=total_counts
    ).log_prob(alt_counts)

    # 50/50 mixture (unknown haplotype phasing)
    log_mix_prob = jnp.logaddexp(log_prob_down, log_prob_up) - jnp.log(2.0)

    # Observation likelihood
    numpyro.factor("obs", log_mix_prob.sum())


class SegmentedGermlineModel:
    """
    Horseshoe-fused segmentation model for germline variants.

    This model fits per-chromosome (not per-arm) and estimates:
    - Global reference bias (delta)
    - Global overdispersion (kappa)
    - Per-variant psi with horseshoe-fused sparsity

    Post-inference, segments are extracted by thresholding increments.

    Parameters
    ----------
    germline_collector : GermlineVariantCollector
        Collected germline variants from VCF
    min_dp_cutoff : int
        Minimum depth filter for variants
    min_variants_per_chrom : int
        Minimum variants required to fit a chromosome
    tau_scale : float
        Scale for global shrinkage (smaller = more sparse)
    ci_level : float
        Credible interval level for segment extraction
    increment_threshold : float
        Threshold for considering an increment as a breakpoint
    """

    def __init__(
        self,
        germline_collector: GermlineVariantCollector,
        min_dp_cutoff: int = 10,
        min_variants_per_chrom: int = 20,
        tau_scale: float = 0.01,
        ci_level: float = 0.95,
        increment_threshold: float = 0.1,
        min_variants_per_segment: int = 5,
    ):
        self.raw_data = germline_collector
        self._min_dp_cutoff = min_dp_cutoff
        self._min_variants_per_chrom = min_variants_per_chrom
        self._tau_scale = tau_scale
        self._ci_level = ci_level
        self._increment_threshold = increment_threshold
        self._min_variants_per_segment = min_variants_per_segment

        # Results storage
        self.chrom_results: Dict[str, ChromosomeSegmentationResult] = {}
        self._segment_lookup: Optional[SegmentLookup] = None

    def _filter_by_depth(
        self,
        positions: np.ndarray,
        depths: np.ndarray,
        alt_counts: np.ndarray,
    ) -> tuple:
        """Filter variants by minimum depth."""
        mask = depths >= self._min_dp_cutoff
        return positions[mask], depths[mask], alt_counts[mask]

    def _fit_chromosome(
        self,
        chrom: str,
        positions: np.ndarray,
        depths: np.ndarray,
        alt_counts: np.ndarray,
        num_warmup: int,
        num_samples: int,
        num_chains: int,
        max_tree_depth: int,
        target_accept_prob: float,
        rng_key: jax.random.PRNGKey,
        centromere_pos: int = 0,
    ) -> ChromosomeSegmentationResult:
        """Fit the segmented model for a single chromosome."""
        n_variants = len(positions)
        logging.info(f"Fitting {chrom} ({n_variants} variants)...")

        # Compute scaled distances
        d_scaled = compute_scaled_distances(jnp.array(positions))

        # Prepare JAX arrays
        alt_jax = jnp.array(alt_counts)
        depth_jax = jnp.array(depths)

        kernel = NUTS(
            germline_segmented_model,
            max_tree_depth=max_tree_depth,
            target_accept_prob=target_accept_prob,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(rng_key, alt_jax, depth_jax, d_scaled=None, tau_scale=self._tau_scale)
        samples = mcmc.get_samples()

        # Extract segments from posterior
        increments_samples = np.array(samples['increments_scaled'])
        psi_samples = np.array(samples['psi'])

        segment_ids, segment_stats = extract_segments_from_posterior(
            positions,
            increments_samples,
            psi_samples,
            ci_level=self._ci_level,
            increment_threshold=self._increment_threshold,
        )

        # Build segment results
        segments = [
            SegmentResult(
                segment_id=s['segment_id'],
                start_position=s['start_position'],
                end_position=s['end_position'],
                n_variants=s['n_variants'],
                psi_mean=s['psi_mean'],
                psi_std=s['psi_std'],
                p_diploid=s['p_diploid'],
            )
            for s in segment_stats
        ]

        # Prune small segments by merging into neighbors
        segments, segment_ids = prune_and_merge_segments(
            segments=segments,
            variant_segment_ids=segment_ids,
            positions=positions,
            centromere_pos=centromere_pos,
            min_variants=self._min_variants_per_segment,
            psi_samples=psi_samples,
        )

        # Compute global parameter posteriors
        delta_samples = np.array(samples['delta'])
        kappa_samples = np.array(samples['kappa'])

        result = ChromosomeSegmentationResult(
            chrom=chrom,
            n_variants=n_variants,
            n_segments=len(segments),
            delta_mean=float(np.mean(delta_samples)),
            delta_std=float(np.std(delta_samples)),
            kappa_mean=float(np.mean(kappa_samples)),
            kappa_std=float(np.std(kappa_samples)),
            segments=segments,
            variant_segment_ids=segment_ids,
            positions=positions,
            psi_samples=psi_samples,
            delta_samples=delta_samples,
            kappa_samples=kappa_samples,
        )

        return result

    def fit(
        self,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        max_tree_depth: int = 12,
        target_accept_prob: float = 0.9,
    ) -> None:
        """
        Fit the segmented model for all chromosomes.

        Parameters
        ----------
        num_warmup : int
            Number of warmup iterations for NUTS
        num_samples : int
            Number of posterior samples
        num_chains : int
            Number of MCMC chains
        max_tree_depth : int
            Maximum tree depth for NUTS
        target_accept_prob : float
            Target acceptance probability for NUTS
        """
        logging.info("Starting Segmented Germline Fit...")

        from .vcf import ChromosomeArmLookup, _CHROMOSOME_ARMS
        arm_lookup = ChromosomeArmLookup(_CHROMOSOME_ARMS)

        chromosomes = self.raw_data.get_available_chromosomes()
        rng_key = jax.random.PRNGKey(42)

        for chrom in chromosomes:
            positions, depths, alt_counts = self.raw_data.get_chromosome_data(chrom)

            if len(positions) == 0:
                logging.warning(f"No variants for {chrom}")
                continue

            # Filter by depth
            positions, depths, alt_counts = self._filter_by_depth(
                positions, depths, alt_counts
            )

            if len(positions) < self._min_variants_per_chrom:
                logging.warning(
                    f"Skipping {chrom}: not enough variants ({len(positions)})"
                )
                # Create default result
                self.chrom_results[chrom] = self._create_default_result(chrom)
                continue

            # Split RNG key for this chromosome
            rng_key, subkey = jax.random.split(rng_key)

            result = self._fit_chromosome(
                chrom=chrom,
                positions=positions,
                depths=depths,
                alt_counts=alt_counts,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                max_tree_depth=max_tree_depth,
                target_accept_prob=target_accept_prob,
                rng_key=subkey,
                centromere_pos=arm_lookup.centromeres.get(chrom, 0),
            )

            self.chrom_results[chrom] = result
            logging.info(
                f"{chrom}: {result.n_segments} segments, "
                f"delta={result.delta_mean:.4f}, kappa={result.kappa_mean:.1f}"
            )

        # Build segment lookup
        self._build_segment_lookup()
        logging.info("Segmented Germline Fit Complete.")

    def _create_default_result(self, chrom: str) -> ChromosomeSegmentationResult:
        """Create default result for chromosomes with insufficient variants."""
        return ChromosomeSegmentationResult(
            chrom=chrom,
            n_variants=0,
            n_segments=1,
            delta_mean=0.005,
            delta_std=0.005,
            kappa_mean=10.0,
            kappa_std=2.0,
            segments=[
                SegmentResult(
                    segment_id=0,
                    start_position=0,
                    end_position=int(1e9),
                    n_variants=0,
                    psi_mean=0.0,
                    psi_std=0.1,
                    p_diploid=0.95,
                )
            ],
            variant_segment_ids=np.array([]),
            positions=np.array([]),
        )

    def _build_segment_lookup(self) -> None:
        """Build the segment lookup table from fitted results."""
        self._segment_lookup = SegmentLookup()
        for chrom, result in self.chrom_results.items():
            self._segment_lookup.add_chromosome(chrom, result)

    def get_segment_lookup(self) -> SegmentLookup:
        """
        Get the segment lookup table for somatic model integration.

        Returns
        -------
        SegmentLookup
            Lookup table for position-to-segment mapping
        """
        if self._segment_lookup is None:
            self._build_segment_lookup()
        return self._segment_lookup

    @property
    def arm_results(self) -> Dict[str, dict]:
        """
        Backward compatibility: aggregate to arm-level results.

        Returns dictionary compatible with GermlineModel.arm_results format.
        """
        from .vcf import ChromosomeArmLookup, _CHROMOSOME_ARMS

        arm_lookup = ChromosomeArmLookup(_CHROMOSOME_ARMS)
        arm_results = {}

        for chrom, result in self.chrom_results.items():
            if result.n_variants == 0:
                continue

            # Group segments by arm
            arm_psi = {'p': [], 'q': []}
            for seg in result.segments:
                mid_pos = (seg.start_position + seg.end_position) // 2
                arm = arm_lookup.query(chrom, mid_pos)
                if arm:
                    arm_psi[arm].append(seg.psi_mean)

            for arm in ['p', 'q']:
                arm_key = f"{chrom}{arm}"
                if arm_psi[arm]:
                    mean_psi = float(np.mean(arm_psi[arm]))
                    arm_results[arm_key] = {
                        'p_diploid_score': float(1.0 if mean_psi < 0.05 else 0.5),
                        'psi_mean': mean_psi,
                        'psi_std': float(np.std(arm_psi[arm])) if len(arm_psi[arm]) > 1 else 0.1,
                        'delta_mean': result.delta_mean,
                        'delta_std': result.delta_std,
                        'kappa_mean': result.kappa_mean,
                        'kappa_std': result.kappa_std,
                    }
                else:
                    arm_results[arm_key] = {
                        'p_diploid_score': 0.95,
                        'psi_mean': 0.0,
                        'psi_std': 0.1,
                        'delta_mean': result.delta_mean,
                        'delta_std': result.delta_std,
                        'kappa_mean': result.kappa_mean,
                        'kappa_std': result.kappa_std,
                    }

        return arm_results


if __name__ == '__main__':
    germVars = GermlineVariantCollector(
        "/Users/danilo/Research/Tools/CBBmix/data/vcf/HCC1395_BREAST_final.vcf.gz",
        af_thresholds=[.35, .65]
    )
    germmodel = GermlineModel(
        germVars
    )
    # germmodel = SegmentedGermlineModel(
    #     germVars, min_variants_per_chrom=10
    # )
    germmodel.fit(
        num_warmup=500,
        num_samples=3000,
    )
    print(germmodel.arm_diploidity)
    # for armkey in germmodel.arm_results:
    #     chrom, arm = armkey[:-1], armkey[-1]
    #     res = germmodel.arm_results[armkey]
    #     vaf_data = germVars.germline_vars[chrom][arm]['hetalt']['VAF']
    #     print(f"{armkey}: VAF={np.mean(vaf_data):.3f}±{np.std(vaf_data):.3f}, "
    #           f"psi={res['psi_mean']:.4f}±{res['psi_std']:.4f}, "
    #           f"delta={res['delta_mean']:.4f}, kappa={res['kappa_mean']:.1f}, "
    #           f"p_diploid={res['p_diploid_score']:.2f}")
    


