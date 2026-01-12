#pragma once

#include <Eigen/Dense>

namespace utils {

class NdimGaussian {
public:
    using Scalar = double;
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    // 默认构造函数
    NdimGaussian() : dim_(0), eta_(), lam_(), factored_(false) {}

    // 构造函数：只给维度
    explicit NdimGaussian(int dimensionality);

    // 构造函数：维度 + eta + lam
    NdimGaussian(int dimensionality,
                 const Vector& eta,
                 const Matrix& lam);

    int dim() const { return dim_; }

    // 访问器
    const Vector& eta() const { return eta_; }
    const Matrix& lam() const { return lam_; }

    // 修改 eta / lam
    void setEta(const Vector& eta);
    void setLam(const Matrix& lam);

    // 信息形式 -> 均值 mu （解 lam * mu = eta）
    Vector mu();

    // 信息形式 -> 协方差 Sigma （解 lam * Sigma = I）
    Matrix Sigma();

private:
    int dim_;
    Vector eta_;
    Matrix lam_;

    // 是否已经做过 Cholesky 分解（对应 Python 的 self.c/self.lower）
    bool factored_;
    Eigen::LLT<Matrix> llt_;   // Cholesky 分解缓存

    // 内部辅助：确保已经 factorize 过
    void ensureFactorized();
};

} // namespace utils
