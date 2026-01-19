#pragma once

#include <Eigen/Dense>
#include <stdexcept>

namespace utils {

class NdimGaussian {
public:
    using Scalar = double;
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    // Constructors
    NdimGaussian();
    explicit NdimGaussian(int dimensionality);
    NdimGaussian(int dimensionality, const Vector& eta, const Matrix& lam);

    // Dimension
    int dim() const { return dim_; }

    // Accessors (read-only)
    const Vector& eta() const;
    const Matrix& lam() const;

    // C-optimization: writable references (NO COPY)
    // Any write invalidates cached factorization.
    Vector& etaRef();
    Matrix& lamRef();

    // Setters (copy)
    void setEta(const Vector& eta);
    void setLam(const Matrix& lam);

    // Optional: resize internal buffers to dimensionality and zero-initialize.
    void resizeLikeDim(int dimensionality);

    // Information form -> mean mu (solve lam * mu = eta)
    Vector mu();

    // Information form -> covariance Sigma (solve lam * Sigma = I)
    Matrix Sigma();

private:
    int dim_;
    Vector eta_;
    Matrix lam_;

    bool factored_;
    Eigen::LLT<Matrix> llt_;

    void ensureFactorized();
};

} // namespace utils
