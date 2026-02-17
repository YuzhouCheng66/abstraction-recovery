#pragma once

#include <Eigen/Dense>
#include <manif/SE2.h>
#include <manif/SO2.h>

inline std::pair<double, Eigen::Matrix<double, 1, 2>> hypot(const Eigen::Vector2d &p) {
  double eps = 1e-8;
  double r = p.norm();
  Eigen::Matrix<double, 1, 2> J = Eigen::Matrix<double, 1, 2>::Zero();
  if (r >= eps) {
    J = p.transpose() / r;
  }
  return std::make_pair(r, J);
}

inline std::pair<manif::SO2d, Eigen::Matrix<double, 1, 2>> atan2(const Eigen::Vector2d &p) {
  double eps = 1e-8;
  double r = p.norm();
  double x = p(0);
  double y = p(1);
  Eigen::Matrix<double, 1, 2> J = Eigen::Matrix<double, 1, 2>::Zero();
  if (r >= eps) {
    J << -y / (r * r), x / (r * r);
  }
  return std::make_pair(manif::SO2d(std::atan2(y, x)), J);
};

inline std::pair<Eigen::Vector2d, Eigen::Matrix<double, 2, 6>>
range_bearing(const manif::SE2d &X1, const manif::SE2d &X2) {

  Eigen::Matrix<double, 3, 3> J_X1inv_X1;
  Eigen::Matrix<double, 2, 3> J_p_X1inv;
  Eigen::Matrix<double, 2, 2> J_p_X2t;
  Eigen::Matrix<double, 2, 2> J_X2t_X2 = X2.rotation();

  manif::SE2d X1_inv = X1.inverse(J_X1inv_X1);
  Eigen::Vector2d p = X1_inv.act(X2.translation(), J_p_X1inv, J_p_X2t);
  Eigen::Matrix<double, 2, 3> J_p_X1 = J_p_X1inv * J_X1inv_X1;
  Eigen::Matrix<double, 2, 2> J_p_X2 = J_p_X2t * J_X2t_X2;

  auto [r, J_r_p] = hypot(p);
  auto [b, J_b_p] = atan2(p);

  Eigen::Matrix<double, 1, 1> J_b;
  Eigen::Vector2d ret;
  ret << r, b.angle();
  Eigen::Matrix<double, 2, 6> J;
  // clang-format off
  J << J_r_p * J_p_X1, J_r_p * J_p_X2, 0.f,
       J_b_p * J_p_X1, J_b_p * J_p_X2, 0.f;
  // clang-format on
  return std::make_pair(ret, J);
}

inline std::pair<Eigen::Vector2d, Eigen::Matrix<double, 2, 6>>
residual_range_bearing(const manif::SE2d &X1, const manif::SE2d &X2,
                       const std::pair<double, manif::SO2d> &m) {
  auto [rb, J] = range_bearing(X1, X2);
  Eigen::Vector2d ret;
  Eigen::Matrix<double, 1, 1> J_r;
  ret(0) = m.first - rb(0);
  ret(1) = m.second.rminus(manif::SO2d(rb(1)), {}, J_r).angle();
  J.block<1, 6>(0, 0) = -J.block<1, 6>(0, 0);
  J.block<1, 6>(1, 0) = J_r * J.block<1, 6>(1, 0);
  return std::make_pair(ret, J);
}
