#pragma once

#include "slam/SlamGraph.h"
#include "NdimGaussian.h"
#include "gbp/VariableNode.h"
#include "gbp/Factor.h"
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include <vector>
#include <map>

namespace slam {

/**
 * Noise configuration for pose graph optimization
 */
struct NoiseConfig {
    double prior_sigma = 10.0;      // Std dev of strong prior
    double odom_sigma = 10.0;        // Std dev of odometry measurement noise
    double tiny_prior = 1e-12;       // Tiny prior on all nodes to prevent singularity
    unsigned int seed = 0;
    bool use_seed = false;
};

/**
 * Noisy measurement for a factor
 */
struct MeasurementData {
    Eigen::VectorXd value;           // Measurement value
    Eigen::MatrixXd precision;       // Precision matrix (inverse of covariance)
};

/**
 * Stores pre-computed noise for all edges
 */
struct NoiseCache {
    std::map<std::pair<int, int>, Eigen::Vector2d> odom_noises;   // Binary edge noises
    std::map<int, Eigen::Vector2d> prior_noises;                  // Unary edge noises
};

/**
 * Simple wrapper for factor graph (to avoid circular dependencies)
 */
struct SimpleFactorGraph {
    std::vector<std::unique_ptr<gbp::VariableNode>> var_nodes;
    std::vector<std::unique_ptr<gbp::Factor>> factors;
    
    // Parameters
    double eta_damping = 0.0;
    
    int getNumNodes() const { return var_nodes.size(); }
    int getNumFactors() const { return factors.size(); }
    
    void synchronousIteration() {
        // Update all factors
        for (auto& f : factors) {
            f->syncAdjBeliefsFromVariables();
            f->computeMessages(eta_damping);
        }
        // Update all variables
        for (auto& v : var_nodes) {
            v->updateBelief();
        }
        // *** CRITICAL: Send updated beliefs back to factors ***
        for (auto& v : var_nodes) {
            v->sendBeliefToFactors();
        }
    }
};

/**
 * Build a noisy 2D pose-only factor graph
 * 
 * Supports both linear (Gaussian) and nonlinear optimization.
 * Creates odometry factors for sequential poses and optional loop closures,
 * plus strong priors on selected nodes.
 * 
 * @param nodes Vector of pose nodes with ground truth positions
 * @param edges Vector of edges (sequential, loop closures, priors, anchor)
 * @param config Noise and prior configuration
 * @return Factor graph ready for optimization
 */
SimpleFactorGraph buildNoisyPoseGraph(
    const std::vector<SlamNode>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config = NoiseConfig()
);

/**
 * Helper functions for factor graph construction
 */

/**
 * Measurement function for unary (prior) factors: f(x) = [x]
 */
inline std::vector<Eigen::VectorXd> measFnUnary(const Eigen::VectorXd& x) {
    return {x};
}

/**
 * Jacobian function for unary factors: J(x) = [I]
 */
inline std::vector<Eigen::MatrixXd> jacFnUnary(const Eigen::VectorXd& x) {
    return {Eigen::MatrixXd::Identity(x.size(), x.size())};
}

/**
 * Measurement function for binary (odometry) factors: f(x0, x1) = [x1 - x0]
 */
inline std::vector<Eigen::VectorXd> measFnOdom(const Eigen::VectorXd& x) {
    // x is concatenated [x0; x1], each 2D
    Eigen::Vector2d x0 = x.head(2);
    Eigen::Vector2d x1 = x.tail(2);
    return {x1 - x0};
}

/**
 * Jacobian function for binary factors: J = [-I, I]
 */
inline std::vector<Eigen::MatrixXd> jacFnOdom(const Eigen::VectorXd& x) {
    Eigen::MatrixXd J(2, 4);
    J << -1,  0,  1,  0,
          0, -1,  0,  1;
    return {J};
}

/**
 * Generate random noise according to configuration
 */
NoiseCache generateNoiseCache(
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config
);

} // namespace slam
