import numpy as np
from scipy.special import gammaln, betaln

def beta_binom_logpmf(a, n, alpha, beta):
    """
    Computes the Log Probability Mass Function (PMF) of the Beta-Binomial distribution.
    
    PMF is:
    
        P(a | n, α, β) = C(n, a) * B(a + α, n - a + β) / B(α, β)
    
    where:
      - C(n, a) is the binomial coefficient: n! / (a! * (n-a)!)
      - B(x, y) is the Beta function: Γ(x)Γ(y) / Γ(x+y)
      
    In Log-space (to prevent underflow), this becomes:
    
        log P = log[C(n, a)] + log[B(a + α, n - a + β)] - log[B(α, β)]
    
    Expanding the binomial coefficient using Log-Gamma (gammaln):

        log[C(n, a)] = gammaln(n+1) - gammaln(a+1) - gammaln(n-a+1)
        
    """
    # Ensure alpha and beta are strictly positive to avoid domain errors in betaln
    alpha = np.maximum(alpha, 1e-9)
    beta = np.maximum(beta, 1e-9)
    return (
        # log C(n, a)
        gammaln(n + 1.0) - gammaln(a + 1.0) - gammaln(n - a + 1.0)
        # + log B(a + α, n - a + β)
        + betaln(a + alpha, n - a + beta) 
        # - log B(α, β)
        - betaln(alpha, beta)
    )

def ab_from_mu_kappa(mu, kappa):
    """
    Converts Mean (mu) and Precision (kappa) parameters to the standard 
    Beta shape parameters (alpha, beta).
    
    Mathematical Relationships:
    ---------------------------
    The standard Beta distribution implies:
        mu (Mean)     = alpha / (alpha + beta)
        kappa (Count) = alpha + beta
        
    Solving for alpha and beta:
        alpha = mu * kappa
        beta  = (1 - mu) * kappa
        
    This parametrization separates location (mu) from dispersion (kappa).
    
    Parameters:
    -----------
    mu : float
        Expected VAF (Variant Allele Frequency), bounded (0, 1).
    kappa : float
        Precision (concentration) parameter. Higher kappa => lower variance.
        Variance = mu(1-mu) / (kappa + 1)
    """
    # Clip mu to (epsilon, 1-epsilon) because alpha, beta must be > 0
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)
    
    return mu * kappa, (1.0 - mu) * kappa

def smart_init_parameters(alt, depth, bounds_mu: np.ndarray):
    """
    Initializes means based on actual data density rather than random guesses.
    """
    vaf = alt / (depth + 1e-9)
    # Assuming subclonal is below .25
    mask_sub = vaf < 0.25
    # if we've at least 5 variants, set to the mean, else back to prior       
    if np.sum(mask_sub) > 5:
        mu_sub = np.mean(vaf[mask_sub])
    else:
        mu_sub = np.mean(bounds_mu[0])
    # Clonal Init:
    mask_cl = (vaf >= bounds_mu[1][0]) & (vaf <= bounds_mu[1][1])
    if np.sum(mask_cl) > 5:
        # here we use median, as the clonal could be way more messy
        mu_cl = np.median(vaf[mask_cl])
    else:
        mu_cl = .5
    # High/LOH Init
    mask_high = vaf > bounds_mu[2][1]
    if np.sum(mask_high) > 5:
        mu_high = np.mean(vaf[mask_high])
    else:
        # here we force it to be way higher.
        mu_high = 0.9
        
    # Clip to bounds
    mu_sub = np.clip(mu_sub, bounds_mu[0][0], bounds_mu[0][1])
    mu_cl  = np.clip(mu_cl,  bounds_mu[1][0], bounds_mu[1][1])
    mu_high= np.clip(mu_high, bounds_mu[2][0], bounds_mu[2][1])
    
    return np.array([mu_sub, mu_cl, mu_high])