#include "NdimGaussian.h"
#include <iostream>
#include <iomanip>

using namespace utils;

int main() {
    const int dim = 3;
    NdimGaussian g(dim);

    // lam = I, eta = [1,2,3]^T
    NdimGaussian::Matrix lam = NdimGaussian::Matrix::Identity(dim, dim);
    NdimGaussian::Vector eta(dim);
    eta << 1.0, 2.0, 3.0;

    g.setLam(lam);
    g.setEta(eta);

    // mu should equal eta when lam = I
    auto mu = g.mu();
    auto Sigma = g.Sigma();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "dim = " << g.dim() << "\n";
    std::cout << "mu    = " << mu.transpose() << "\n";
    std::cout << "Sigma =\n" << Sigma << "\n";

    // Additional sanity check: lam * mu should equal eta
    auto check = lam * mu;
    std::cout << "lam*mu = " << check.transpose() << "\n";
    std::cout << "eta    = " << eta.transpose() << "\n";

    return 0;
}
