import numpy as np
import scipy.linalg

class NdimGaussian:
    def __init__(self, dimensionality, eta=None, lam=None):
        self.dim = dimensionality

        self.eta = eta if (eta is not None and len(eta) == self.dim) else np.zeros(self.dim)
        self.lam = lam if (lam is not None and lam.shape == (self.dim, self.dim)) else np.eye(self.dim) * 1e-12

        self.c = None
        self.lower = None

    def mu(self):
        try:
            self.c, self.lower = scipy.linalg.cho_factor(self.lam, lower=False, check_finite=False
                )
            return scipy.linalg.cho_solve((self.c, self.lower), self.eta)

        except Exception:
            # fallback: pseudo-inverse
            return np.linalg.solve(self.lam, self.eta)

