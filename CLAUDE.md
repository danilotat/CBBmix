# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CBBmix is a 3-component Beta-Binomial mixture model for clustering clonal structure of somatic variants in cancer genomics. It uses Bayesian inference (JAX/Numpyro) for germline parameter estimation and a Pitman-Yor Process mixture for somatic variant clustering.

## Commands

```bash
pip install -e ".[dev]"          # Install for development
pytest tests/                     # Run all tests
pytest tests/test_utils.py -v     # Run single test file
black . && ruff check .           # Lint
```

## Architecture

### Data Flow

VCF → GermlineVariantCollector (hetalt, AF 0.25-0.75) → GermlineEstimator (MCMC) → per-arm delta/kappa → build_somatic_prior_from_germline() → SomaticMixture (Pitman-Yor) → cluster assignments

### Modules (`src/CBBmix/`)

- **vcf.py**: VCF parsing, variant collection, `ChromosomeArmLookup` with hardcoded GRCh38 arm boundaries
- **germline.py**: Bayesian model detecting allelic imbalance/LOH per chromosome arm
- **somatic.py**: 3-component mixture (subclonal/clonal/LOH) using Pitman-Yor process
- **utils.py**: Beta-Binomial statistics, MLE fitting, data classes, prior construction

### Germline definition

The germline model is used to infer the chromosome arm baseline to allele imbalance. Each arm is then proposed to be a mixture of betabinomials, fitted on the single variants.
Given a variant on position $i$, with total depth $d$:

$$
d_\text{alt} \sim BetaBinom(d, \alpha, \beta)
$$

However, given that we're in RNA-seq, the shift from the expected 0.5 for diploid heterozygosity is given mostly by two components

- $\delta$ : bias towards reference sequence
- $\sigma$ : allele specific expression

The reference bias affects positively the ref allele, while it has negative effects on the alt allele. We would assume then that given $\sigma = 0$, the AF for an heterozygous variant will be $\text{AF}_{\text{RNA}}= \text{AF}_\text{true} - \delta$

For this reason, we modeled $\delta \sim HalfNorm(0.01)$. The allele specific expression, instead, could go in both directions, thus affect the overall count. Again its impact is key and spreaded, and is i.i.d in all the cells with the same behavior. Thus $\sigma \sim Norm(0, 0.1)$

For a diploid heterozygote (expected 0.5), the base log-odds is $\text{logit}(0.5) = 0$. We modeled it as:

$$
\begin{aligned}
\eta &= 0 - \delta' + \sigma' \\
\mu &= \text{logit}^{-1}(\eta) = \frac{1}{1 + e^{-\eta}}
\end{aligned}
$$


We reparametrize 

$$
\begin{aligned}
\alpha &= \text{logit}^{-1}(-\delta + \sigma) \cdot (\kappa - 1) \\
\beta &= (1 - \text{logit}^{-1}(-\delta + \sigma)) \cdot (\kappa - 1)
\end{aligned}
$$

where 

$$
\begin{aligned}
\kappa &= \frac{1}{\phi} + 1 \\
\phi &\sim Exponential(1)
\end{aligned}
$$

When a segment has a CNA (like a duplication AAB), the allele ratio shifts away from 0.5. However, without knowing the haplotype phasing (which allele is on the duplicated chromosome), for any given variant $i$, the shift could be up (towards the alternative) or down (towards the reference).

Therefore, a CNA segment doesn't look like a shifted bell curve; it looks like a **split** (bimodal) distribution.

### The Proposed Component: Segment Magnitude $\psi_s$

We introduce a segment-specific parameter $\psi_s \ge 0$, representing the **magnitude of the imbalance** for segment $s$.

*   **If $\psi_s \approx 0$**: The segment is Diploid (balanced).
*   **If $\psi_s > 0$**: The segment has an alteration.

Since we don't know the phase, we model the likelihood of each variant as a **50/50 mixture** of shifting up or shifting down.

### The Mathematical Formulation

For a variant $i$ belonging to segment $s$:

$$
\begin{aligned}
\eta_{base, i} &= -\delta + \sigma_i \\
\eta_{down, i} &= \eta_{base, i} - \psi_s \\
\eta_{up, i} &= \eta_{base, i} + \psi_s
\end{aligned}
$$

The likelihood for the data $d_{\text{alt}}$ is a mixture of two Beta-Binomials:

$$
\begin{aligned}
P(d_{\text{alt}} | d, \dots) = \frac{1}{2} \cdot \text{BetaBinom}(d_{\text{alt}} | d, \mu(\eta_{down}), \kappa) + \frac{1}{2} \cdot \text{BetaBinom}(d_{\text{alt}} | d, \mu(\eta_{up}), \kappa)
\end{aligned}
$$

Where $\mu(\eta) = \text{logit}^{-1}(\eta)$.

*   **Global Bias:** $\delta \sim \text{HalfNormal}(0.01)$
*   **Global Overdispersion:** $\phi \sim \text{Exponential}(1)$ $\rightarrow$ $\kappa = \phi^{-1} + 1$
*   **Local ASE:** $\sigma_i \sim \text{Normal}(0, 0.1)$ (Random noise per variant)
*   **Segment Shift:** $\psi_s \sim \text{HalfNormal}(0.5)$ (The signal of interest)


### Somatic Definition

The somatic model clusters variants based on their Cellular Prevalence (CP). Unlike the germline model, where the allele ratio is fixed by the copy number state, the somatic allele ratio is a function of both the **segment imbalance** ($\psi_s$) and the **fraction of tumor cells** ($\rho$) carrying the mutation.

We assume the somatic variants inherit the noise characteristics ($\kappa$) and reference bias ($\delta$) of the germline model, but possess unique, unmeasured local stochasticity ($\sigma_i$).

### The Pitman-Yor Process (Truncated Stick-Breaking)

We define a truncation level $K$. The mixing weights $\boldsymbol{w} = (w_1, \dots, w_K)$ are generated via the stick-breaking construction:

$$
\begin{aligned}
\nu_k &\sim \text{Beta}(1 - \theta, \alpha_{\text{PY}} + k\theta) \quad \text{for } k=1,\dots,K-1 \\
\nu_K &= 1 \\
w_k &= \nu_k \prod_{j=1}^{k-1} (1 - \nu_j)
\end{aligned}
$$

The atoms of the process are the Cellular Prevalences $\rho_k$, representing the fraction of cells carrying the variants in cluster $k$:

$$
\rho_k \sim \text{Beta}(1, 1) \quad \text{(Uniform Prior on CP)}
$$

For each variant $i$, we assign a cluster label $z_i$:

$$
z_i \sim \text{Categorical}(\boldsymbol{w})
$$

### The Likelihood and Phasing

Since we do not know the haplotype, a somatic mutation on an imbalanced segment (where $\psi_s > 0$) effectively has two possible "expression baselines": it is either on the **Major** (over-expressed) allele or the **Minor** (under-expressed) allele.

For a specific variant $i$ in segment $s$ assigned to cluster $z_i=k$, we first draw the specific local noise $\sigma_i$ (since it is unknown for new variants):

$$
\sigma_i \sim \text{Normal}(0, 0.1)
$$

We calculate the potential allelic proportions of the underlying transcript:

$$
\begin{aligned}
\eta_{\text{base}, i} &= -\delta + \sigma_i \\
\pi_{\text{maj}, i} &= \text{logit}^{-1}(\eta_{\text{base}, i} + \psi_s) \\
\pi_{\text{min}, i} &= \text{logit}^{-1}(\eta_{\text{base}, i} - \psi_s)
\end{aligned}
$$

The expected somatic VAF ($\mu$) is the product of the Cellular Prevalence ($\rho_k$) and the transcript proportion ($\pi$). We introduce a latent phasing variable $h_i \in \{0, 1\}$ for each variant:

$$
h_i \sim \text{Bernoulli}(0.5)
$$

$$
\mu_{i} = 
\begin{cases} 
\rho_k \cdot \pi_{\text{maj}, i} & \text{if } h_i = 1 \text{ (Major Allele)} \\
\rho_k \cdot \pi_{\text{min}, i} & \text{if } h_i = 0 \text{ (Minor Allele)}
\end{cases}
$$

Finally, the observation model utilizes the germline precision $\kappa$:

$$
d_{\text{alt}, i} \sim \text{BetaBinom}(d_i, \mu_{i}, \kappa)
$$

### Integration of Germline Posteriors

To rigorously incorporate the uncertainty from your previous step, we do not fix the germline parameters. Let $\Omega_{\text{germ}} = \{ \delta, \phi, \psi_{1:S} \}$ be the set of germline parameters.

We denote the set of $N$ posterior samples from your germline model as $\Omega^{(1)}, \dots, \Omega^{(N)}$.

During the inference of the somatic model using NUTS, at each iteration $t$:

1.  Sample an index $j \sim \text{Uniform}(1, \dots, N)$.
2.  Retrieve the germline state: $\delta^{(j)}, \kappa^{(j)} = (1/\phi^{(j)}) + 1, \psi_{s}^{(j)}$.
3.  Update the somatic latent variables ($\sigma_i, z_i, \rho_k, h_i$) conditioned on these specific germline values.

### Summary of Hierarchical Structure

$$
\begin{aligned}
& \text{Global Priors:} \\
& \quad (\alpha_{\text{PY}}, \theta_{\text{PY}}) \quad \text{Fixed Hyperparameters} \\
& \quad \Omega_{\text{germ}} \sim \hat{P}_{\text{germline}} \quad \text{(Empirical Posterior)} \\
\\
& \text{Cluster Parameters (for } k=1\dots K \text{):} \\
& \quad \nu_k \sim \text{Beta}(1-\theta, \alpha + k\theta) \\
& \quad \rho_k \sim \text{Beta}(1, 1) \\
\\
& \text{Local Variant Latents (for } i=1\dots N \text{):} \\
& \quad z_i \sim \text{Categorical}(\boldsymbol{w}) \\
& \quad \sigma_i \sim \text{Normal}(0, 0.1) \\
& \quad h_i \sim \text{Bernoulli}(0.5) \\
\\
& \text{Deterministic Link:} \\
& \quad \mu_{i} = \rho_{z_i} \cdot \text{logit}^{-1}(-\delta + \sigma_i + (2h_i - 1)\psi_{s_i}) \\
\\
& \text{Observation:} \\
& \quad d_{\text{alt}, i} \sim \text{BetaBinom}(d_i, \mu_{i}, \kappa)
\end{aligned}
$$