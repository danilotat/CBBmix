"""
Somatic Pitman-Yor Process mixture model for clustering somatic variants.

This module implements a Bayesian nonparametric mixture model to cluster
somatic variants by their Cellular Prevalence (CP). The model integrates
germline-estimated parameters (delta, kappa, psi) to account for:
- Reference bias (delta)
- Overdispersion (kappa)
- Segment-level allelic imbalance (psi)

The Pitman-Yor Process provides a flexible prior over cluster assignments,
allowing for power-law behavior in cluster sizes.

Model specification (see CLAUDE.md for full mathematical details):
- Cluster weights via truncated stick-breaking
- Cellular Prevalence (rho_k) per cluster
- Per-variant haplotype assignment (h_i) and ASE noise (sigma_i)
- Beta-Binomial observation model
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.handlers import seed, trace
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from germline import GermlineModel
from vcf import SomaticVariantCollector

jax.config.update("jax_enable_x64", True)


MAX_KAPPA = 200.0


@dataclass
class SomaticPriorConfig:
    """Configuration for somatic model priors.

    Attributes:
        alpha_py: Pitman-Yor concentration parameter (controls cluster count)
        theta_py: Pitman-Yor discount parameter (controls power-law behavior)
        max_clusters: Truncation level K for stick-breaking
        sigma_scale: Scale for per-variant ASE noise (default 0.1)
        rho_alpha: Beta prior alpha for Cellular Prevalence (default 1.0)
        rho_beta: Beta prior beta for Cellular Prevalence (default 1.0)
        max_kappa: Maximum allowed kappa from germline (default 200.0)
    """
    alpha_py: float = 1.0
    theta_py: float = 0.1
    max_clusters: int = 10
    sigma_scale: float = 0.1
    rho_alpha: float = 1.0
    rho_beta: float = 1.0
    max_kappa: float = MAX_KAPPA


class SomaticModel:
    """
    Pitman-Yor Process mixture model for clustering somatic variants.

    This model clusters somatic variants based on their Cellular Prevalence (CP),
    integrating germline-estimated parameters to correct for reference bias,
    overdispersion, and segment-level allelic imbalance.

    The model accounts for unknown haplotype phasing by introducing a latent
    variable h_i for each variant, representing whether the mutation is on
    the major or minor allele.

    Parameters
    ----------
    somatic_collector_data : SomaticVariantCollector
        Collected somatic variants from VCF
    germline_model : GermlineModel
        Fitted germline model with posterior samples
    prior_config : SomaticPriorConfig, optional
        Configuration for model priors
    min_dp_cutoff : int
        Minimum depth filter for variants (default 10)
    use_germline_samples : bool
        If True, sample from germline posterior during inference.
        If False, use point estimates (posterior means).
    """

    def __init__(
        self,
        somatic_collector_data: SomaticVariantCollector,
        germline_model: GermlineModel,
        prior_config: Optional[SomaticPriorConfig] = None,
        min_dp_cutoff: int = 10,
        use_germline_samples: bool = False,
    ):
        self.raw_data = somatic_collector_data
        self.germline_model = germline_model
        self.germline_results = germline_model.arm_results
        self.prior_config = prior_config or SomaticPriorConfig()
        self._min_dp_cutoff = min_dp_cutoff
        self._use_germline_samples = use_germline_samples

        # Preprocess data to link variants with arm-level germline parameters
        self.data_df = self._preprocess_data()

        # Store inference results
        self.mcmc = None
        self.samples = None
        self.clustering_results = None

    def _preprocess_data(self) -> pd.DataFrame:
        """
        Extract somatic variants and map them to germline arm-level parameters.

        Returns DataFrame with columns:
        - chrom, arm: chromosome arm identifier
        - depth, alt_count, vaf: variant read data
        - arm_delta, arm_kappa, arm_psi: germline parameters for this arm
        """
        records = []

        for chrom, arms in self.raw_data.somatic_vars.items():
            for arm in arms:
                try:
                    arm_data = self.raw_data.somatic_vars[chrom][arm]
                    dps = arm_data['DP']
                    alt_dps = arm_data['alt_DP']
                    vafs = arm_data['VAF']

                    arm_key = f"{chrom}{arm}"

                    # Retrieve germline posteriors for this arm
                    if arm_key in self.germline_results:
                        arm_stats = self.germline_results[arm_key]
                        g_delta = arm_stats.get('delta_mean', 0.0)
                        g_kappa = arm_stats.get('kappa_mean', 10.0)
                        g_psi = arm_stats.get('psi_mean', 0.0)
                    else:
                        logging.warning(
                            f"Arm {arm_key} not found in germline results. "
                            "Using diploid defaults."
                        )
                        g_delta = 0.0
                        g_kappa = 10.0
                        g_psi = 0.0

                    # Cap kappa to prevent numerical issues
                    g_kappa = min(g_kappa, self.prior_config.max_kappa)

                    for d, ad, v in zip(dps, alt_dps, vafs):
                        records.append({
                            'chrom': chrom,
                            'arm': arm,
                            'arm_key': arm_key,
                            'depth': int(d),
                            'alt_count': int(ad),
                            'vaf': float(v),
                            'arm_delta': float(g_delta),
                            'arm_kappa': float(g_kappa),
                            'arm_psi': float(g_psi),
                        })
                except Exception as e:
                    logging.debug(f"No somatic variants for {chrom}{arm}: {e}")

        df = pd.DataFrame(records)
        if not df.empty:
            df = df[df['depth'] >= self._min_dp_cutoff]

        return df

    def _pitman_yor_model(
        self,
        depth: jnp.ndarray,
        alt_count: jnp.ndarray,
        arm_delta: jnp.ndarray,
        arm_kappa: jnp.ndarray,
        arm_psi: jnp.ndarray,
    ):
        """
        Numpyro model for Pitman-Yor Process somatic variant clustering.

        Model Structure (see CLAUDE.md):

        Global Priors:
            (alpha_PY, theta_PY) - Fixed hyperparameters
            Omega_germ ~ P_germline (Empirical Posterior)

        Cluster Parameters (k=1...K):
            nu_k ~ Beta(1-theta, alpha + k*theta)
            rho_k ~ Beta(rho_alpha, rho_beta)

        Local Variant Latents (i=1...N):
            z_i ~ Categorical(w)
            sigma_i ~ Normal(0, sigma_scale)
            h_i ~ Bernoulli(0.5)

        Deterministic Link:
            mu_i = rho_{z_i} * logit^{-1}(-delta + sigma_i + (2*h_i - 1)*psi_{s_i})

        Observation:
            d_alt,i ~ BetaBinom(d_i, mu_i, kappa)
        """
        config = self.prior_config
        K = config.max_clusters
        n_variants = depth.shape[0]

        # ============================================
        # 1. Pitman-Yor Stick-Breaking Construction
        # ============================================
        # nu_k ~ Beta(1 - theta, alpha + k*theta) for k=1,...,K-1
        # nu_K = 1 (to ensure weights sum to 1)

        with numpyro.plate("sticks", K - 1):
            k_indices = jnp.arange(1, K)  # k = 1, 2, ..., K-1
            nu = numpyro.sample(
                "nu",
                dist.Beta(
                    1.0 - config.theta_py,
                    config.alpha_py + k_indices * config.theta_py
                )
            )

        # Append nu_K = 1 for truncation
        nu_full = jnp.concatenate([nu, jnp.array([1.0])])

        # Compute mixing weights via stick-breaking
        # w_k = nu_k * prod_{j<k}(1 - nu_j)
        one_minus_nu = 1.0 - nu_full
        cumprod_one_minus_nu = jnp.concatenate([
            jnp.array([1.0]),
            jnp.cumprod(one_minus_nu[:-1])
        ])
        weights = nu_full * cumprod_one_minus_nu
        weights = numpyro.deterministic("weights", weights)

        # ============================================
        # 2. Cluster-Level Parameters: Cellular Prevalence
        # ============================================
        with numpyro.plate("clusters", K):
            rho = numpyro.sample(
                "rho",
                dist.Beta(config.rho_alpha, config.rho_beta)
            )

        # ============================================
        # 3. Variant-Level Latent Variables
        # ============================================
        with numpyro.plate("variants", n_variants):
            # Cluster assignment
            z = numpyro.sample("z", dist.Categorical(weights))

            # Per-variant ASE noise
            sigma = numpyro.sample(
                "sigma",
                dist.Normal(0.0, config.sigma_scale)
            )

            # Haplotype assignment (0 = minor allele, 1 = major allele)
            h = numpyro.sample("h", dist.Bernoulli(0.5))

        # ============================================
        # 4. Compute Expected VAF
        # ============================================
        # Get cellular prevalence for each variant's assigned cluster
        rho_i = rho[z]

        # Compute allelic proportions based on haplotype
        # eta_base = -delta + sigma
        # If h=1 (major): eta = eta_base + psi
        # If h=0 (minor): eta = eta_base - psi
        eta_base = -arm_delta + sigma

        # (2*h - 1) maps h=0 -> -1, h=1 -> +1
        haplotype_sign = 2.0 * h - 1.0
        eta = eta_base + haplotype_sign * arm_psi

        # Allelic proportion via inverse logit
        pi = jax.nn.sigmoid(eta)

        # Expected somatic VAF = CP * allelic proportion
        mu = rho_i * pi

        # Numerical stability
        mu = jnp.clip(mu, 1e-6, 1.0 - 1e-6)

        # ============================================
        # 5. Observation Model: Beta-Binomial
        # ============================================
        # Parameterization: alpha = mu*(kappa-1), beta = (1-mu)*(kappa-1)
        conc = arm_kappa - 1.0
        conc = jnp.maximum(conc, 1e-6)  # Ensure positive

        with numpyro.plate("obs", n_variants):
            numpyro.sample(
                "y",
                dist.BetaBinomial(
                    concentration1=mu * conc,
                    concentration0=(1.0 - mu) * conc,
                    total_count=depth
                ),
                obs=alt_count
            )

    def _pitman_yor_model_marginalized(
        self,
        depth: jnp.ndarray,
        alt_count: jnp.ndarray,
        arm_delta: jnp.ndarray,
        arm_kappa: jnp.ndarray,
        arm_psi: jnp.ndarray,
    ):
        """
        Marginalized version of the PYP model for more efficient inference.

        This version marginalizes out the discrete latent variables (z, h)
        analytically, which can improve MCMC mixing.
        """
        config = self.prior_config
        K = config.max_clusters
        n_variants = depth.shape[0]

        # ============================================
        # 1. Pitman-Yor Stick-Breaking
        # ============================================
        with numpyro.plate("sticks", K - 1):
            k_indices = jnp.arange(1, K)
            nu = numpyro.sample(
                "nu",
                dist.Beta(
                    1.0 - config.theta_py,
                    config.alpha_py + k_indices * config.theta_py
                )
            )

        nu_full = jnp.concatenate([nu, jnp.array([1.0])])
        one_minus_nu = 1.0 - nu_full
        cumprod_one_minus_nu = jnp.concatenate([
            jnp.array([1.0]),
            jnp.cumprod(one_minus_nu[:-1])
        ])
        weights = nu_full * cumprod_one_minus_nu
        weights = numpyro.deterministic("weights", weights)
        log_weights = jnp.log(weights + 1e-10)

        # ============================================
        # 2. Cluster Cellular Prevalences
        # ============================================
        with numpyro.plate("clusters", K):
            rho = numpyro.sample(
                "rho",
                dist.Beta(config.rho_alpha, config.rho_beta)
            )

        # ============================================
        # 3. Per-Variant ASE Noise
        # ============================================
        with numpyro.plate("variants", n_variants):
            sigma = numpyro.sample(
                "sigma",
                dist.Normal(0.0, config.sigma_scale)
            )

        # ============================================
        # 4. Compute Likelihoods (Marginalized)
        # ============================================
        # Expand dimensions for broadcasting: (N, K, 2) for variants x clusters x haplotypes

        # Shape: (N, 1)
        delta_exp = arm_delta[:, None]
        psi_exp = arm_psi[:, None]
        kappa_exp = arm_kappa[:, None]
        sigma_exp = sigma[:, None]
        depth_exp = depth[:, None]
        alt_exp = alt_count[:, None]

        # Shape: (1, K)
        rho_exp = rho[None, :]

        # Compute eta for both haplotypes
        eta_base = -delta_exp + sigma_exp  # (N, 1)

        # Major allele (h=1): eta_base + psi
        eta_major = eta_base + psi_exp  # (N, 1)
        # Minor allele (h=0): eta_base - psi
        eta_minor = eta_base - psi_exp  # (N, 1)

        # Allelic proportions
        pi_major = jax.nn.sigmoid(eta_major)  # (N, 1)
        pi_minor = jax.nn.sigmoid(eta_minor)  # (N, 1)

        # Expected VAF for each cluster and haplotype
        # mu = rho * pi, shape: (N, K)
        mu_major = rho_exp * pi_major
        mu_minor = rho_exp * pi_minor

        # Clip for stability
        mu_major = jnp.clip(mu_major, 1e-6, 1.0 - 1e-6)
        mu_minor = jnp.clip(mu_minor, 1e-6, 1.0 - 1e-6)

        # Concentration parameter
        conc = jnp.maximum(kappa_exp - 1.0, 1e-6)  # (N, 1)

        # Log-likelihoods for each cluster and haplotype
        # Shape: (N, K)
        log_prob_major = dist.BetaBinomial(
            concentration1=mu_major * conc,
            concentration0=(1.0 - mu_major) * conc,
            total_count=depth_exp
        ).log_prob(alt_exp)

        log_prob_minor = dist.BetaBinomial(
            concentration1=mu_minor * conc,
            concentration0=(1.0 - mu_minor) * conc,
            total_count=depth_exp
        ).log_prob(alt_exp)

        # Marginalize over haplotype: log(0.5 * exp(major) + 0.5 * exp(minor))
        # = log(0.5) + logsumexp(major, minor)
        log_prob_cluster = jnp.log(0.5) + jnp.logaddexp(
            log_prob_major, log_prob_minor
        )  # (N, K)

        # Marginalize over clusters: log(sum_k w_k * p(y|k))
        # = logsumexp(log_w_k + log_p(y|k))
        log_mixture_prob = jax.scipy.special.logsumexp(
            log_weights + log_prob_cluster, axis=-1
        )  # (N,)

        # Factor the total log-likelihood
        numpyro.factor("obs", log_mixture_prob.sum())

    def fit(
        self,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        seed: int = 42,
        use_marginalized: bool = True,
    ) -> None:
        """
        Fit the somatic clustering model using MCMC.

        Parameters
        ----------
        num_warmup : int
            Number of warmup/burn-in iterations
        num_samples : int
            Number of posterior samples to draw
        num_chains : int
            Number of MCMC chains
        seed : int
            Random seed for reproducibility
        use_marginalized : bool
            If True, use marginalized model (more efficient).
            If False, use full model with discrete latent variables.
        """
        if self.data_df.empty:
            logging.warning("No somatic variants after filtering. Skipping fit.")
            return

        n_variants = len(self.data_df)
        logging.info(f"Fitting somatic model on {n_variants} variants...")
        logging.info(f"Max clusters: {self.prior_config.max_clusters}")

        # Prepare JAX arrays
        depth = jnp.array(self.data_df['depth'].values)
        alt_count = jnp.array(self.data_df['alt_count'].values)
        arm_delta = jnp.array(self.data_df['arm_delta'].values)
        arm_kappa = jnp.array(self.data_df['arm_kappa'].values)
        arm_psi = jnp.array(self.data_df['arm_psi'].values)

        # Select model
        model_fn = (
            self._pitman_yor_model_marginalized
            if use_marginalized
            else self._pitman_yor_model
        )

        # Run MCMC
        kernel = NUTS(model_fn)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=True,
        )

        self.mcmc.run(
            jax.random.PRNGKey(seed),
            depth,
            alt_count,
            arm_delta,
            arm_kappa,
            arm_psi,
        )

        self.samples = self.mcmc.get_samples()
        self._summarize_results()

        logging.info("Somatic model fitting complete.")

    def _summarize_results(self) -> None:
        """Summarize posterior to identify major clonal populations."""
        if self.samples is None:
            return

        # Compute posterior mean weights
        weights_samples = self.samples['weights']  # (n_samples, K)
        rho_samples = self.samples['rho']  # (n_samples, K)

        mean_weights = jnp.mean(weights_samples, axis=0)
        mean_rho = jnp.mean(rho_samples, axis=0)
        std_rho = jnp.std(rho_samples, axis=0)

        # Build summary for clusters with weight > 5%
        summary_records = []
        for k in range(self.prior_config.max_clusters):
            if mean_weights[k] > 0.05:
                summary_records.append({
                    'cluster_id': k,
                    'weight': float(mean_weights[k]),
                    'rho_mean': float(mean_rho[k]),
                    'rho_std': float(std_rho[k]),
                })

        self.clustering_results = pd.DataFrame(summary_records)
        if not self.clustering_results.empty:
            self.clustering_results = self.clustering_results.sort_values(
                'rho_mean', ascending=False
            )

        print("\n--- Identified Somatic Clones (Cellular Prevalence) ---")
        print(self.clustering_results.to_string(index=False))

    def get_cluster_assignments(self) -> Optional[np.ndarray]:
        """
        Compute posterior cluster assignments for each variant.

        Returns
        -------
        assignments : np.ndarray of shape (n_variants,)
            Most likely cluster assignment for each variant, or None if not fitted.
        """
        if self.samples is None:
            logging.warning("Model not fitted. Call fit() first.")
            return None

        if 'z' in self.samples:
            # Mode of discrete z samples
            z_samples = self.samples['z']  # (n_samples, n_variants)
            z_mode = jax.scipy.stats.mode(z_samples, axis=0).mode
            return np.array(z_mode)
        else:
            # For marginalized model, compute assignment probabilities
            logging.info("Computing cluster assignments from marginalized model...")
            return self._compute_assignments_marginalized()

    def _compute_assignments_marginalized(self) -> np.ndarray:
        """Compute cluster assignments from marginalized model posteriors."""
        depth = jnp.array(self.data_df['depth'].values)
        alt_count = jnp.array(self.data_df['alt_count'].values)
        arm_delta = jnp.array(self.data_df['arm_delta'].values)
        arm_kappa = jnp.array(self.data_df['arm_kappa'].values)
        arm_psi = jnp.array(self.data_df['arm_psi'].values)

        # Use posterior means
        weights = jnp.mean(self.samples['weights'], axis=0)
        rho = jnp.mean(self.samples['rho'], axis=0)
        sigma = jnp.mean(self.samples['sigma'], axis=0)

        K = self.prior_config.max_clusters
        n_variants = len(depth)

        # Compute log-posteriors for each cluster
        log_weights = jnp.log(weights + 1e-10)

        # Expand dims
        delta_exp = arm_delta[:, None]
        psi_exp = arm_psi[:, None]
        kappa_exp = arm_kappa[:, None]
        sigma_exp = sigma[:, None]
        depth_exp = depth[:, None]
        alt_exp = alt_count[:, None]
        rho_exp = rho[None, :]

        eta_base = -delta_exp + sigma_exp
        eta_major = eta_base + psi_exp
        eta_minor = eta_base - psi_exp

        pi_major = jax.nn.sigmoid(eta_major)
        pi_minor = jax.nn.sigmoid(eta_minor)

        mu_major = jnp.clip(rho_exp * pi_major, 1e-6, 1.0 - 1e-6)
        mu_minor = jnp.clip(rho_exp * pi_minor, 1e-6, 1.0 - 1e-6)

        conc = jnp.maximum(kappa_exp - 1.0, 1e-6)

        log_prob_major = dist.BetaBinomial(
            concentration1=mu_major * conc,
            concentration0=(1.0 - mu_major) * conc,
            total_count=depth_exp
        ).log_prob(alt_exp)

        log_prob_minor = dist.BetaBinomial(
            concentration1=mu_minor * conc,
            concentration0=(1.0 - mu_minor) * conc,
            total_count=depth_exp
        ).log_prob(alt_exp)

        log_prob_cluster = jnp.log(0.5) + jnp.logaddexp(
            log_prob_major, log_prob_minor
        )

        # Posterior cluster probabilities (unnormalized log)
        log_posterior = log_weights + log_prob_cluster

        # Argmax for MAP assignment
        assignments = jnp.argmax(log_posterior, axis=-1)

        return np.array(assignments)

    def print_summary(self) -> None:
        """Print MCMC summary statistics."""
        if self.mcmc is not None:
            self.mcmc.print_summary()
        else:
            logging.warning("Model not fitted. Call fit() first.")


if __name__ == '__main__':
    from vcf import GermlineVariantCollector, SomaticVariantCollector

    # Example usage
    vcf_path = "/Users/danilo/Research/Tools/CBBmix/data/vcf/ipiPD1_26_PRE_final_passonly.vcf.gz"

    # 1. Collect variants
    germ_collector = GermlineVariantCollector(vcf_path, af_thresholds=[0.25, 0.75])
    som_collector = SomaticVariantCollector(vcf_path)

    # 2. Fit germline model
    germ_model = GermlineModel(germ_collector)
    germ_model.fit(num_warmup=200, num_samples=500)

    # 3. Fit somatic model with germline priors
    prior_config = SomaticPriorConfig(
        alpha_py=1.0,
        theta_py=0.1,
        max_clusters=10,
        sigma_scale=0.1,
    )

    som_model = SomaticModel(
        som_collector,
        germ_model,
        prior_config=prior_config,
    )

    som_model.fit(num_warmup=200, num_samples=500)
    som_model.print_summary()

    # Get cluster assignments
    assignments = som_model.get_cluster_assignments()
    if assignments is not None:
        print(f"\nCluster assignments: {np.unique(assignments, return_counts=True)}")
