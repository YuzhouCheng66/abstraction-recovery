import numpy as np
import scipy

class NdimGaussian:
    def __init__(self, dimensionality, eta=None, lam=None):
        self.dim = dimensionality

        if eta is not None and len(eta) == self.dim:
            self.eta = eta
        else:
            self.eta = np.zeros(self.dim)

        if lam is not None and lam.shape == (self.dim, self.dim):
            self.lam = lam
        else:
            self.lam = np.eye(self.dim)*1e-12

        self.c = None
        self.lower = None
    def mu(self):
        if self.lower:
            return scipy.linalg.cho_solve((self.c, self.lower), self.eta)            # solve Lam mu = eta
        else:
            self.c, self.lower = scipy.linalg.cho_factor(self.lam, lower=False, check_finite=False)
            return scipy.linalg.cho_solve((self.c, self.lower), self.eta)            # solve Lam mu = eta

    def Sigma(self):
        if self.lower:
            return scipy.linalg.cho_solve((self.c, self.lower), np.eye(self.lam.shape[0]))  # solve Lam Sigma = I
        else:
            self.c, self.lower = scipy.linalg.cho_factor(self.lam, lower=False, check_finite=False)
            return scipy.linalg.cho_solve((self.c, self.lower), np.eye(self.lam.shape[0]))  # solve Lam Sigma = I