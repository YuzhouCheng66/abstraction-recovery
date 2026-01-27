// NdimGaussian.h
#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <stdexcept>

namespace utils {

// Gaussian in information form (eta, lam).
// SOO fast-path:
//   - dim==2: stores eta/lam in fixed-size Vector2d/Matrix2d (no heap alloc)
//   - dim!=2: stores eta/lam in VectorXd/MatrixXd
//
// Exposes a uniform API via Eigen::Map views (zero-copy).
class NdimGaussian {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = double;
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    using VectorMapConst = Eigen::Map<const Eigen::VectorXd>;
    using MatrixMapConst = Eigen::Map<const Eigen::MatrixXd>;
    using VectorMap      = Eigen::Map<Eigen::VectorXd>;
    using MatrixMap      = Eigen::Map<Eigen::MatrixXd>;

    NdimGaussian();
    explicit NdimGaussian(int dimensionality);
    NdimGaussian(int dimensionality, const Vector& eta, const Matrix& lam);

    int dim() const { return dim_; }

    // Zero-copy views
    VectorMapConst eta() const;
    MatrixMapConst lam() const;

    // Writable views
    // NOTE: eta changes do NOT invalidate LLT cache (LLT depends only on lam)
    VectorMap etaRef();
    // lam changes DO invalidate LLT cache
    MatrixMap lamRef();

    void setEta(const Vector& eta); // does NOT invalidate LLT cache
    void setLam(const Matrix& lam); // invalidates LLT cache

    void resizeLikeDim(int dimensionality); // invalidates LLT cache

    // Returns dense mu and Sigma (VectorXd/MatrixXd) for simplicity.
    // (Hot paths should avoid calling these if possible.)
    Vector mu() const;
    Matrix Sigma() const;

private:
    int dim_ = 0;

    // dim==2 storage
    Eigen::Vector2d eta2_ = Eigen::Vector2d::Zero();
    Eigen::Matrix2d lam2_ = Eigen::Matrix2d::Zero();

    // dim!=2 storage
    Vector eta_;
    Matrix lam_;

    // Factorization cache for solving (depends ONLY on lam)
    mutable bool cache_valid_ = false;
    mutable int  cached_dim_  = -1;

    mutable Eigen::LLT<Eigen::Matrix2d> llt2_;
    mutable Eigen::LLT<Matrix>          lltx_;

    void ensureFactorized() const;
};

} // namespace utils
