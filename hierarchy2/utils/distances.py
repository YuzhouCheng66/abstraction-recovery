import numpy as np
from scipy import linalg

np.seterr(divide='ignore')

def bhattacharyya(Gauss1, Gauss2):
    if Gauss1.dim > 1:
        cov1 = linalg.inv(Gauss1.lam)
        cov2 = linalg.inv(Gauss2.lam)
        cov = (cov1 + cov2)/2
        dist = 0.125 * (Gauss1.eta - Gauss2.eta).T @ cov @ (Gauss1.eta - Gauss2.eta) \
                + 0.5 * np.log(linalg.det(cov)/np.sqrt(linalg.det(cov1) * linalg.det(cov2)))
    else: #  inverse and determinant functions don't like 1D inputs so use the simplified 1D distance calculation
        cov1 = 1/Gauss1.lam
        cov2 = 1/Gauss2.lam
        cov = (cov1 + cov2)/2
        dist = 0.25 * ((Gauss1.eta - Gauss2.eta) ** 2) / (Gauss1.lam + Gauss2.lam) \
                + 0.5 * np.log((Gauss1.lam + Gauss2.lam) / (2 * np.sqrt(Gauss1.lam) * np.sqrt(Gauss2.lam)))  
    
    return dist

def mahalanobis(Gauss1, Gauss2):
    dist = np.linalg.norm(Gauss1.eta - Gauss2.eta) / np.sqrt(np.diag(Gauss1.lam))

    return dist