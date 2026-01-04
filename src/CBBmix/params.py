from dataclasses import dataclass, field
import numpy as np


_DEFAULT_MU_BOUNDS = np.array([[0, 0.25], [0.35, 0.65], [0.85, 1]])
_DEFAULT_KAPPA_BOUNDS = (5, 5000)
_DEFAULT_KAPPA_INIT = np.array([50.0, 100.0, 50.0])
_DEFAULT_PI_INIT = np.array([0.33, 0.34, 0.33])
_DEFAULT_WEIGHT_PRIOR = 1.5
_DEFAULT_MU_PRIOR_MEAN = 0.5
_DEFAULT_MU_PRIOR_SIGMA = 0.1


@dataclass
class BetaBinomialMixtureSpec:
    """
    Complete specification of a 3-component Beta-Binomial mixture model.
    Owns *all* statistical assumptions and defaults.
    """

    # Structural
    bounds_mu: np.ndarray = field(default_factory=lambda: _DEFAULT_MU_BOUNDS.copy())
    bounds_kappa: tuple = _DEFAULT_KAPPA_BOUNDS

    # Initialization
    kappa_init: np.ndarray = field(default_factory=lambda: _DEFAULT_KAPPA_INIT.copy())
    pi_init: np.ndarray = field(default_factory=lambda: _DEFAULT_PI_INIT.copy())

    # Priors / regularization
    weight_prior: float = _DEFAULT_WEIGHT_PRIOR
    mu_prior_mean: float = _DEFAULT_MU_PRIOR_MEAN
    mu_prior_sigma: float = _DEFAULT_MU_PRIOR_SIGMA

    def __post_init__(self):
        self.bounds_mu = np.asarray(self.bounds_mu, dtype=float)
        self.kappa_init = np.asarray(self.kappa_init, dtype=float)
        self.pi_init = np.asarray(self.pi_init, dtype=float)
        if self.bounds_mu.shape != (3, 2):
            raise ValueError("bounds_mu must have shape (3, 2)")
        if self.kappa_init.shape != (3,):
            raise ValueError("kappa_init must have shape (3,)")
        if self.pi_init.shape != (3,):
            raise ValueError("pi_init must have shape (3,)")
        self.pi_init = self.pi_init / self.pi_init.sum()
