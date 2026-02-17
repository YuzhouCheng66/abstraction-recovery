import numpy as np
import manifpy


class NdimGaussian:
    def __init__(self, dimensionality, eta=None, lam=None):
        self.dim = dimensionality

        self.mu = manifpy.SE2Tangent(np.array([0, 0, 0]))

        if eta is not None and len(eta) == self.dim:
            self.eta = eta
        else:
            self.eta = np.zeros(self.dim)

        if lam is not None and lam.shape == (self.dim, self.dim):
            self.lam = lam
        else:
            self.lam = np.zeros([self.dim, self.dim])


