#include "NdimGaussian.h"

#include <cassert>

namespace utils {

NdimGaussian::NdimGaussian()
    : dim_(0), eta_(), lam_(), factored_(false), llt_() {}

NdimGaussian::NdimGaussian(int dimensionality)
    : dim_(dimensionality),
      eta_(Vector::Zero(dimensionality)),
      lam_(Matrix::Zero(dimensionality, dimensionality)),
      factored_(false),
      llt_() {}

NdimGaussian::NdimGaussian(int dimensionality, const Vector& eta, const Matrix& lam)
    : dim_(dimensionality),
      eta_(eta),
      lam_(lam),
      factored_(false),
      llt_() {
    assert(eta_.size() == dim_);
    assert(lam_.rows() == dim_ && lam_.cols() == dim_);
}

const NdimGaussian::Vector& NdimGaussian::eta() const { return eta_; }
const NdimGaussian::Matrix& NdimGaussian::lam() const { return lam_; }

NdimGaussian::Vector& NdimGaussian::etaRef() {
    factored_ = false;
    return eta_;
}

NdimGaussian::Matrix& NdimGaussian::lamRef() {
    factored_ = false;
    return lam_;
}

void NdimGaussian::setEta(const Vector& eta) {
    if (dim_ == 0) {
        dim_ = static_cast<int>(eta.size());
        eta_.resize(dim_);
        lam_.resize(dim_, dim_);
    }
    assert(eta.size() == dim_);
    eta_ = eta;
    factored_ = false;
}

void NdimGaussian::setLam(const Matrix& lam) {
    if (dim_ == 0) {
        dim_ = static_cast<int>(lam.rows());
        eta_.resize(dim_);
        lam_.resize(dim_, dim_);
    }
    assert(lam.rows() == dim_ && lam.cols() == dim_);
    lam_ = lam;
    factored_ = false;
}

void NdimGaussian::resizeLikeDim(int dimensionality) {
    if (dim_ == dimensionality && eta_.size() == dimensionality && lam_.rows() == dimensionality && lam_.cols() == dimensionality) {
        return;
    }
    dim_ = dimensionality;
    eta_.setZero(dim_);
    lam_.setZero(dim_, dim_);
    factored_ = false;
}

NdimGaussian::Vector NdimGaussian::mu() {
    ensureFactorized();
    return llt_.solve(eta_);
}

NdimGaussian::Matrix NdimGaussian::Sigma() {
    ensureFactorized();
    Matrix I = Matrix::Identity(dim_, dim_);
    return llt_.solve(I);
}

void NdimGaussian::ensureFactorized() {
    if (dim_ <= 0) {
        throw std::runtime_error("NdimGaussian::ensureFactorized: dim_ <= 0");
    }
    if (!factored_) {
        llt_.compute(lam_);
        if (llt_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: LLT failed (matrix not SPD?)");
        }
        factored_ = true;
    }
}

} // namespace utils
