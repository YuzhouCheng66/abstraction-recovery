// NdimGaussian.cpp
#include "NdimGaussian.h"

#include <cassert>

namespace utils {

NdimGaussian::NdimGaussian() = default;

NdimGaussian::NdimGaussian(int dimensionality) {
    resizeLikeDim(dimensionality);
}

NdimGaussian::NdimGaussian(int dimensionality, const Vector& eta, const Matrix& lam) {
    resizeLikeDim(dimensionality);
    setEta(eta);
    setLam(lam);
}

NdimGaussian::VectorMapConst NdimGaussian::eta() const {
    if (dim_ == 2) return VectorMapConst(eta2_.data(), 2);
    return VectorMapConst(eta_.data(), dim_);
}

NdimGaussian::MatrixMapConst NdimGaussian::lam() const {
    if (dim_ == 2) return MatrixMapConst(lam2_.data(), 2, 2);
    return MatrixMapConst(lam_.data(), dim_, dim_);
}

NdimGaussian::VectorMap NdimGaussian::etaRef() {
    // LLT cache depends only on lam, so DO NOT invalidate here.
    if (dim_ == 2) return VectorMap(eta2_.data(), 2);
    return VectorMap(eta_.data(), dim_);
}

NdimGaussian::MatrixMap NdimGaussian::lamRef() {
    // Any change to lam invalidates LLT cache.
    cache_valid_ = false;
    cached_dim_  = -1;

    if (dim_ == 2) return MatrixMap(lam2_.data(), 2, 2);
    return MatrixMap(lam_.data(), dim_, dim_);
}

void NdimGaussian::setEta(const Vector& eta) {
    if (dim_ == 0) resizeLikeDim((int)eta.size());
    assert((int)eta.size() == dim_);

    if (dim_ == 2) {
        eta2_ = eta.head<2>();
    } else {
        eta_ = eta;
    }
    // DO NOT invalidate LLT cache (depends only on lam).
}

void NdimGaussian::setLam(const Matrix& lam) {
    if (dim_ == 0) resizeLikeDim((int)lam.rows());
    assert((int)lam.rows() == dim_ && (int)lam.cols() == dim_);

    if (dim_ == 2) {
        lam2_ = lam.topLeftCorner<2, 2>();
    } else {
        lam_ = lam;
    }

    // Changing lam invalidates LLT cache.
    cache_valid_ = false;
    cached_dim_  = -1;
}

void NdimGaussian::resizeLikeDim(int dimensionality) {
    if (dimensionality < 0) {
        throw std::runtime_error("NdimGaussian::resizeLikeDim: negative dim");
    }
    if (dim_ == dimensionality) return;

    dim_ = dimensionality;

    if (dim_ == 2) {
        // Fixed-size storage only; keep dynamic buffers empty to avoid heap.
        eta2_.setZero();
        lam2_.setZero();
        eta_.resize(0);
        lam_.resize(0, 0);
    } else {
        eta_.setZero(dim_);
        lam_.setZero(dim_, dim_);
    }

    // Dim change invalidates LLT cache.
    cache_valid_ = false;
    cached_dim_  = -1;
}

NdimGaussian::Vector NdimGaussian::mu() const {
    ensureFactorized();

    if (dim_ == 2) {
        Eigen::Vector2d mu2 = llt2_.solve(eta2_);
        Vector out(2);
        out = mu2;
        return out;
    }
    return lltx_.solve(eta_);
}

NdimGaussian::Matrix NdimGaussian::Sigma() const {
    ensureFactorized();

    if (dim_ == 2) {
        const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        const Eigen::Matrix2d S = llt2_.solve(I);
        Matrix out(2, 2);
        out = S;
        return out;
    }

    const Matrix I = Matrix::Identity(dim_, dim_);
    return lltx_.solve(I);
}

void NdimGaussian::ensureFactorized() const {
    if (dim_ == 0) {
        throw std::runtime_error("NdimGaussian::ensureFactorized: dim==0");
    }

    // Cache hit
    if (cache_valid_ && cached_dim_ == dim_) return;

    if (dim_ == 2) {
        llt2_.compute(lam2_);
        if (llt2_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: LLT failed (dim==2, not SPD?)");
        }
    } else {
        lltx_.compute(lam_);
        if (lltx_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: LLT failed (dim!=2, not SPD?)");
        }
    }

    cache_valid_ = true;
    cached_dim_  = dim_;
}

} // namespace utils
