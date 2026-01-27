#pragma once

#include "slam/SlamGraph.h"
#include "gbp/FactorGraph.h"
#include <Eigen/Dense>

#include <vector>
#include <map>
#include <cmath>

namespace slam {

// ============================================================
// Existing linear (x,y) builder
// ============================================================

struct NoiseConfig {
    double prior_sigma = 10.0;
    double odom_sigma  = 10.0;
    double tiny_prior  = 1e-12;
    unsigned int seed  = 0;
    bool use_seed      = false;
};

struct NoiseCache {
    std::map<std::pair<int, int>, Eigen::Vector2d> odom_noises;
    std::map<int, Eigen::Vector2d> prior_noises;
};

// measurement / jac functions (same as你原本写的 inline)
inline std::vector<Eigen::VectorXd> measFnUnary(const Eigen::VectorXd& x) {
    return {x};
}
inline std::vector<Eigen::MatrixXd> jacFnUnary(const Eigen::VectorXd& x) {
    return {Eigen::MatrixXd::Identity((int)x.size(), (int)x.size())};
}
inline std::vector<Eigen::VectorXd> measFnOdom(const Eigen::VectorXd& x) {
    Eigen::Vector2d x0 = x.head(2);
    Eigen::Vector2d x1 = x.tail(2);
    return {x1 - x0};
}
inline std::vector<Eigen::MatrixXd> jacFnOdom(const Eigen::VectorXd&) {
    Eigen::MatrixXd J(2, 4);
    J << -1, 0, 1, 0,
          0,-1, 0, 1;
    return {J};
}

NoiseCache generateNoiseCache(
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config
);

gbp::FactorGraph buildNoisyPoseGraph(
    const std::vector<SlamNode>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config = NoiseConfig()
);

// ============================================================
// SE(2) nonlinear pose-graph builder (x, y, theta)
// ============================================================

struct NoiseConfigSE2 {
    double prior_sigma = 1.0;   // xy
    double odom_sigma  = 1.0;   // xy
    double loop_sigma  = 1.0;   // xy
    double theta_ratio = 0.01;  // sigma_theta = sigma_xy * theta_ratio
    double tiny_prior  = 1e-10;

    unsigned int seed  = 0;
    bool use_seed      = false;
};

struct NoiseCacheSE2 {
    std::map<std::pair<int, int>, Eigen::Vector3d> between_noises; // (i,j)->noise
    std::map<int, Eigen::Vector3d> prior_noises;                   // i->noise
};

inline double wrapAngleSE2(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

// Unary: z = x
inline std::vector<Eigen::VectorXd> measFnUnarySE2(const Eigen::VectorXd& x) {
    return {x};
}
inline std::vector<Eigen::MatrixXd> jacFnUnarySE2(const Eigen::VectorXd& x) {
    return {Eigen::MatrixXd::Identity((int)x.size(), (int)x.size())};
}

// Between SE2:
// xij = [xi, yi, thi, xj, yj, thj]
inline std::vector<Eigen::VectorXd> measFnBetweenSE2(const Eigen::VectorXd& xij) {
    const double thi = xij(2);
    const double xj  = xij(3), yj = xij(4), thj = xij(5);
    const double xi  = xij(0), yi = xij(1);

    const double c = std::cos(thi), s = std::sin(thi);
    Eigen::Matrix2d RT;
    RT <<  c, s,
          -s, c; // R(thi)^T

    Eigen::Vector2d dp(xj - xi, yj - yi);
    Eigen::Vector2d r = RT * dp;
    const double dth = wrapAngleSE2(thj - thi);

    Eigen::Vector3d out(r(0), r(1), dth);
    return {out};
}

inline std::vector<Eigen::MatrixXd> jacFnBetweenSE2(const Eigen::VectorXd& xij) {
    const double thi = xij(2);
    const double xi  = xij(0), yi = xij(1);
    const double xj  = xij(3), yj = xij(4);

    const double c = std::cos(thi), s = std::sin(thi);
    Eigen::Matrix2d RT;
    RT <<  c, s,
          -s, c;

    Eigen::Vector2d dp(xj - xi, yj - yi);
    Eigen::Vector2d r = RT * dp; // [rx, ry]

    Eigen::Vector2d dr_dthi(r(1), -r(0)); // [ry, -rx]

    Eigen::Matrix<double, 3, 6> J;
    J.setZero();
    // wrt i
    J.block<2,2>(0,0) = -RT;
    J.block<2,1>(0,2) = dr_dthi;
    // wrt j
    J.block<2,2>(0,3) = RT;
    // theta row
    J(2,2) = -1.0;
    J(2,5) =  1.0;

    return {J};
}

NoiseCacheSE2 generateNoiseCacheSE2(
    const std::vector<SlamEdge>& edges,
    const NoiseConfigSE2& config
);

gbp::FactorGraph buildNoisyPoseGraphSE2(
    const std::vector<SlamNodeSE2>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfigSE2& config = NoiseConfigSE2()
);

} // namespace slam
