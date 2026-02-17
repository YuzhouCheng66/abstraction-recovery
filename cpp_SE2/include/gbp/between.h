#pragma once

#include <Eigen/Dense>
#include <manif/SE2.h>
#include <manif/SO2.h>

inline std::pair<manif::SE2d, Eigen::Matrix<double, 3, 6>>
between(const manif::SE2d &X1, const manif::SE2d &X2) {
  Eigen::Matrix3d J1_X2_X1, J2_X2_X1;
  manif::SE2d ret = X1.between(X2, J1_X2_X1, J2_X2_X1);
  Eigen::Matrix<double, 3, 6> J;
  J.block<3, 3>(0, 0) = J1_X2_X1;
  J.block<3, 3>(0, 3) = J2_X2_X1;
  return std::make_pair(ret, J);
}

inline std::pair<Eigen::Vector3d, Eigen::Matrix<double, 3, 6>>
residual_between(const manif::SE2d &X1, const manif::SE2d &X2,
                 const manif::SE2d &m) {
  auto [bt, J] = between(X1, X2);
  Eigen::Matrix<double, 3, 3> J_r;
  Eigen::Vector3d ret = m.rminus(bt, {}, J_r).coeffs();
  J = J_r * J;
  return std::make_pair(ret, J);
}
