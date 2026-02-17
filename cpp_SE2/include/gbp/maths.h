#pragma once
#include <Eigen/Dense>
#include <vector>

inline double condition(const Eigen::MatrixXd &X) {
  if (X.size() == 0) {
    return 0.f;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(X);
  return svd.singularValues()(0) /
         svd.singularValues()(svd.singularValues().size() - 1);
}

inline bool is_symmetric(const Eigen::MatrixXd &X) {
  if (X.size() == 0) {
    return true;
  }
  return X.isApprox(X.transpose());
}

inline std::vector<size_t> seq(size_t i, size_t j) {
  std::vector<size_t> v;
  for (size_t k = i; k <= j; ++k) {
    v.push_back(k);
  }
  return v;
}

inline Eigen::MatrixXd slice(const Eigen::MatrixXd &X,
                             const std::vector<size_t> &idxs) {
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(idxs.size(), X.cols());
  size_t i = 0;
  for (size_t idx : idxs) {
    Y.row(i++) = X.row(idx);
  }
  return Y;
}

inline Eigen::MatrixXd slice(const Eigen::MatrixXd &X,
                             const std::vector<size_t> &row_idxs,
                             const std::vector<size_t> &col_idxs) {
  Eigen::MatrixXd Y = slice(X, row_idxs);
  Eigen::MatrixXd Z = slice(Y.transpose(), col_idxs);
  return Z.transpose();
}

inline std::pair<Eigen::VectorXd, Eigen::MatrixXd>
marginalize(const Eigen::VectorXd &eta, const Eigen::MatrixXd &Lam, size_t i,
            size_t j) {
  size_t k = eta.size();
  std::vector<size_t> a = seq(i, j);
  std::vector<size_t> b;
  for (size_t n = 0; n < k; ++n) {
    if (n < i || n > j) {
      b.push_back(n);
    }
  }

  Eigen::VectorXd eta_a = slice(eta, a);
  Eigen::MatrixXd lam_aa = slice(Lam, a, a);
  if (b.size() == 0) {
    return std::make_pair(eta_a, lam_aa);
  }

  Eigen::VectorXd eta_b = slice(eta, b);
  Eigen::MatrixXd lam_ab = slice(Lam, a, b);
  Eigen::MatrixXd lam_ba = lam_ab.transpose();
  Eigen::MatrixXd lam_bb = slice(Lam, b, b);
  Eigen::MatrixXd lam_bb_inv = lam_bb.inverse();
  Eigen::VectorXd m_eta = eta_a - lam_ab * lam_bb_inv * eta_b;
  Eigen::MatrixXd m_Lam = lam_aa - lam_ab * lam_bb_inv * lam_ba;

  return std::make_pair(m_eta, m_Lam);
}
