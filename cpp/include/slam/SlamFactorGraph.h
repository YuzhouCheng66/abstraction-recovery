#pragma once

#include "slam/SlamGraph.h"
#include "gbp/FactorGraph.h"   // 关键：用 gbp::FactorGraph 取代 SimpleFactorGraph
#include <Eigen/Dense>
#include <vector>
#include <map>

namespace slam {

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

// 关键：返回 gbp::FactorGraph
gbp::FactorGraph buildNoisyPoseGraph(
    const std::vector<SlamNode>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config = NoiseConfig()
);

} // namespace slam
