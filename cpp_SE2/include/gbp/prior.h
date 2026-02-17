#pragma once

#include <Eigen/Dense>
#include <manif/SE2.h>
#include <manif/SO2.h>

inline std::pair<Eigen::Vector3d, Eigen::Matrix<double, 3, 3>>
residual_prior(const manif::SE2d &X, const manif::SE2d &p) {
  Eigen::Matrix3d J;
  Eigen::Vector3d ret = p.minus(X, {}, J).coeffs();
  return std::make_pair(ret, J);
}
