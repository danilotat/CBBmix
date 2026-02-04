import numpy as np
import pandas as pd
import jax
import logging
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from collections import defaultdict
from typing import Optional
from vcf import GermlineVariantCollector
# from utils import (
#     fit_beta_binomial_mle,
#     GermlineSpec,
#     GermlineFitResult,
# )

jax.config.update("jax_enable_x64", True)

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
                 min_dp_cutoff=10, min_snp=10):
        self._min_dp_cutoff = min_dp_cutoff
        self._min_snp = min_snp
        self.raw_data = germline_collector_data
        self.data_df = self.preprocess_data()
        self.arm_results = {}

    def preprocess_data(self):
        records = []
        for chrom, arms in self.raw_data.germline_vars.items():
            for arm in arms:
                try:
                    dps, alt_dps, vafs = (
                        self.raw_data.germline_vars[chrom][arm]['hetalt'].values()
                    )
                    for d, ad, v in zip(dps, alt_dps, vafs):
                        records.append({
                            'chrom': chrom,
                            'arm': arm,
                            'depth': int(d),
                            'alt_count': int(ad),
                            'vaf': float(v)
                        })
                except Exception:
                    logging.warning(
                        f"No heterozygous variants found for arm {arm} of chromosome {chrom}"
                    )

        df = pd.DataFrame(records)
        df = df[df['depth'] >= self._min_dp_cutoff]
        return df

    def _model_single_arm(self, depth, alt_count):
        """
        Germline model for detecting allelic imbalance per chromosome arm.

        Model specification (see CLAUDE.md):
        - delta: reference bias, affects alt allele negatively (HalfNormal(0.01))
        - phi/kappa: overdispersion (phi ~ Exp(1), kappa = 1/phi + 1)
        - psi: segment shift magnitude (HalfNormal(0.1)) - signal of interest
        - sigma_i: per-variant ASE noise (Normal(0, 0.1))

        For a diploid heterozygote (expected AF=0.5), base logit is 0.
        eta_base = -delta + sigma (ref bias shifts down, ASE is random)

        CNA creates bimodal distribution (unknown phasing):
        eta_down = eta_base - psi
        eta_up = eta_base + psi

        Likelihood is 50/50 mixture of Beta-Binomials.
        """
        n_variants = depth.shape[0]

        # Global reference bias (small, shifts AF down from 0.5)
        delta = numpyro.sample("delta", dist.HalfNormal(0.01))

        # Overdispersion via phi -> kappa reparameterization
        phi = numpyro.sample("phi", dist.Exponential(1.0))
        kappa = numpyro.deterministic("kappa", 1.0 / phi + 1.0)

        # Segment shift magnitude (psi=0 means diploid, psi>0 means imbalanced)
        psi = numpyro.sample("psi", dist.HalfNormal(0.5))

        # Per-variant allele-specific expression noise
        with numpyro.plate("variants", n_variants):
            sigma = numpyro.sample("sigma", dist.Normal(0, 0.1))

        # Compute log-odds (eta) for each variant
        # Base: diploid het has logit(0.5)=0, shifted by -delta (ref bias) + sigma (ASE)
        eta_base = -delta + sigma
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
            total_count=depth
        ).log_prob(alt_count)

        # Log-likelihood for up-shifted component
        log_prob_up = dist.BetaBinomial(
            concentration1=mu_up * conc,
            concentration0=(1 - mu_up) * conc,
            total_count=depth
        ).log_prob(alt_count)

        # 50/50 mixture (unknown haplotype phasing)
        log_mix_prob = jnp.logaddexp(log_prob_down, log_prob_up) - jnp.log(2.0)

        # Observation likelihood
        numpyro.factor("obs", log_mix_prob.sum())


    def fit(self, num_warmup=500, num_samples=1000):
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
                self.arm_results[arm_key] = {
                    'p_diploid_score': 0.95,
                    'psi_mean': 0.0,
                    'psi_std': 0.1,
                    'delta_mean': 0.005,
                    'delta_std': 0.005,
                    'kappa_mean': 10.0,
                    'kappa_std': 2.0,
                }
                continue

            logging.info(f"Fitting {arm_key} ({len(subset)} variants)...")
            logging.info(f"Across arm {arm_key} of {chrom} we have a mean VAF of {np.mean(subset['vaf'])} ")
            depth = jnp.array(subset['depth'].values)
            alt = jnp.array(subset['alt_count'].values)

            kernel = NUTS(self._model_single_arm)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=1)
            mcmc.run(jax.random.PRNGKey(42), depth, alt)
            
            samples = mcmc.get_samples()

            # Extract posterior samples
            psi_post = samples['psi']
            delta_post = samples['delta']
            kappa_post = samples['kappa']

            # Diploid probability: psi near 0 indicates balanced/diploid
            is_diploid_prob = jnp.mean(psi_post < 0.05)

            self.arm_results[arm_key] = {
                'p_diploid_score': float(is_diploid_prob),
                'psi_mean': float(jnp.mean(psi_post)),
                'psi_std': float(jnp.std(psi_post)),
                'delta_mean': float(jnp.mean(delta_post)),
                'delta_std': float(jnp.std(delta_post)),
                'kappa_mean': float(jnp.mean(kappa_post)),
                'kappa_std': float(jnp.std(kappa_post)),
            }

        logging.info("Germline Fit Complete.")


if __name__ == '__main__':
    germVars = GermlineVariantCollector(
        "/Users/danilo/Research/Tools/CBBmix/data/vcf/ipiPD1_26_PRE_final_passonly.vcf.gz",
        af_thresholds=[.25, .75]
    )
    germmodel = GermlineModel(
        germVars
    )
    germmodel.fit()
    for armkey in germmodel.arm_results:
        chrom, arm = armkey[:-1], armkey[-1]
        res = germmodel.arm_results[armkey]
        vaf_data = germVars.germline_vars[chrom][arm]['hetalt']['VAF']
        print(f"{armkey}: VAF={np.mean(vaf_data):.3f}±{np.std(vaf_data):.3f}, "
              f"psi={res['psi_mean']:.4f}±{res['psi_std']:.4f}, "
              f"delta={res['delta_mean']:.4f}, kappa={res['kappa_mean']:.1f}, "
              f"p_diploid={res['p_diploid_score']:.2f}")
    


