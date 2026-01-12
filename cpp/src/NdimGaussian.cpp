#include "NdimGaussian.h"
#include <stdexcept>

namespace utils {
NdimGaussian::NdimGaussian(int dimensionality)
    : dim_(dimensionality),
      eta_(Vector::Zero(dimensionality)),
      lam_(Matrix::Identity(dimensionality, dimensionality) * 1e-12),
      factored_(false) {}

// 维度 + eta + lam
NdimGaussian::NdimGaussian(int dimensionality,
                           const Vector& eta,
                           const Matrix& lam)
    : dim_(dimensionality),
      eta_(eta),
      lam_(lam),
      factored_(false) {
    if (eta_.size() != dim_) {
        throw std::runtime_error("NdimGaussian: eta dimension mismatch");
    }
    if (lam_.rows() != dim_ || lam_.cols() != dim_) {
        throw std::runtime_error("NdimGaussian: lam shape mismatch");
    }
}

void NdimGaussian::setEta(const Vector& eta) {
    if (eta.size() != dim_) {
        throw std::runtime_error("NdimGaussian::setEta: dimension mismatch");
    }
    eta_ = eta;
    // 这里只改右边向量，不改 lam_，Cholesky 分解仍然有效，所以不重置 factored_
}

void NdimGaussian::setLam(const Matrix& lam) {
    if (lam.rows() != dim_ || lam.cols() != dim_) {
        throw std::runtime_error("NdimGaussian::setLam: shape mismatch");
    }
    lam_ = lam;
    factored_ = false;  // 信息矩阵变了，需要重新做 Cholesky
}

void NdimGaussian::ensureFactorized() {
    if (!factored_) {
        // 这里就是 Cholesky 分解，对应 Python 的 cho_factor(self.lam, ...)
        llt_.compute(lam_);
        if (llt_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: Cholesky failed");
        }
        factored_ = true;
    }
}

NdimGaussian::Vector NdimGaussian::mu() {
    // Python:
    // if self.lower:
    //     return cho_solve((self.c, self.lower), self.eta)
    // else:
    //     self.c, self.lower = cho_factor(self.lam, ...)
    //     return cho_solve((self.c, self.lower), self.eta)

    ensureFactorized();
    return llt_.solve(eta_);
}

NdimGaussian::Matrix NdimGaussian::Sigma() {
    // Python:
    // if self.lower:
    //     return cho_solve((self.c, self.lower), I)
    // else:
    //     self.c, self.lower = cho_factor(...)
    //     return cho_solve((self.c, self.lower), I)

    ensureFactorized();
    Matrix I = Matrix::Identity(dim_, dim_);
    return llt_.solve(I);
}

} // namespace utils
