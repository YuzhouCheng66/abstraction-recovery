#include "slam/SlamFactorGraph.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace slam {

NoiseCache generateNoiseCache(
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config)
{
    NoiseCache cache;
    
    // Initialize RNG
    std::mt19937 rng(config.use_seed ? config.seed : std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // Pre-generate noise for all edges
    for (const auto& e : edges) {
        // Binary edge (odometry)
        if (!e.is_prior && !e.is_anchor) {
            int src = e.source;
            int dst = e.target;
            if (src >= 0 && dst >= 0) {
                Eigen::Vector2d noise;
                noise(0) = normal_dist(rng) * config.odom_sigma;
                noise(1) = normal_dist(rng) * config.odom_sigma;
                cache.odom_noises[{src, dst}] = noise;
            }
        }
        // Unary edge (strong prior)
        else if (e.is_prior && !e.is_anchor) {
            int src = e.source;
            Eigen::Vector2d noise;
            noise(0) = normal_dist(rng) * config.prior_sigma;
            noise(1) = normal_dist(rng) * config.prior_sigma;
            cache.prior_noises[src] = noise;
        }
    }
    
    return cache;
}

SimpleFactorGraph buildNoisyPoseGraph(
    const std::vector<SlamNode>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config)
{
    SimpleFactorGraph fg;
    
    int N = nodes.size();
    Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
    
    // ============ Pre-generate noise ============
    NoiseCache noise_cache = generateNoiseCache(edges, config);
    
    // ============ Create variable nodes ============
    std::vector<gbp::VariableNode*> var_nodes;
    for (int i = 0; i < N; ++i) {
        auto v = std::make_unique<gbp::VariableNode>(i, 2);  // id=i, dofs=2
        
        // Store ground truth
        v->GT = nodes[i].GT;
        
        // Add tiny prior to all nodes to prevent singularity
        v->prior.setLam(config.tiny_prior * I2);
        v->prior.setEta(Eigen::Vector2d::Zero());
        
        var_nodes.push_back(v.get());
        fg.var_nodes.push_back(std::move(v));
    }
    
    // ============ Create factors ============
    int fid = 0;
    
    for (const auto& e : edges) {
        // Binary factor (odometry between two poses)
        if (!e.is_prior && !e.is_anchor && e.source >= 0 && e.target >= 0) {
            int i = e.source;
            int j = e.target;
            
            gbp::VariableNode* vi = var_nodes[i];
            gbp::VariableNode* vj = var_nodes[j];
            
            // Noisy measurement: (vj.GT - vi.GT) + noise
            Eigen::Vector2d meas = (vj->GT - vi->GT);
            if (noise_cache.odom_noises.count({i, j})) {
                meas += noise_cache.odom_noises.at({i, j});
            }
            
            // Precision matrix (inverse of covariance)
            Eigen::Matrix2d meas_precision = Eigen::Matrix2d::Identity() / (config.odom_sigma * config.odom_sigma);
            
            auto factor = std::make_unique<gbp::Factor>(fid++, std::vector<gbp::VariableNode*>{vi, vj});
            factor->dim = 2;
            factor->measurements = {meas};
            factor->precisions = {meas_precision};
            
            // Compute factor distribution from measurement
            factor->computeFactor();
            
            // Store for adjacency bookkeeping and message passing
            vi->adj_factors_raw.push_back(factor.get());
            vj->adj_factors_raw.push_back(factor.get());
            
            // CRITICAL: Establish bidirectional relationship for message passing
            vi->adj_factors.push_back(gbp::AdjFactorRef{factor.get(), 0});  // vi is the 0-th variable
            vj->adj_factors.push_back(gbp::AdjFactorRef{factor.get(), 1});  // vj is the 1-st variable
            
            fg.factors.push_back(std::move(factor));
        }
        // Unary factor (strong prior)
        else if (e.is_prior && !e.is_anchor && e.source >= 0) {
            int i = e.source;
            gbp::VariableNode* vi = var_nodes[i];
            
            // Prior measurement: GT + noise
            Eigen::Vector2d z = vi->GT;
            if (noise_cache.prior_noises.count(i)) {
                z += noise_cache.prior_noises.at(i);
            }
            
            // Precision matrix
            Eigen::Matrix2d z_precision = Eigen::Matrix2d::Identity() / (config.prior_sigma * config.prior_sigma);
            
            auto factor = std::make_unique<gbp::Factor>(fid++, std::vector<gbp::VariableNode*>{vi});
            factor->dim = 2;
            factor->measurements = {z};
            factor->precisions = {z_precision};
            
            // Compute unary factor distribution
            factor->computeFactor();
            
            // CRITICAL: Establish bidirectional relationship for message passing
            vi->adj_factors.push_back(gbp::AdjFactorRef{factor.get(), 0});  // vi is the 0-th variable
            vi->adj_factors_raw.push_back(factor.get());
            
            fg.factors.push_back(std::move(factor));
        }
        // Anchor factor (fixed node 0)
        else if (e.is_anchor && e.source == 0) {
            int i = e.source;
            gbp::VariableNode* vi = var_nodes[i];
            
            // Very strong prior on node 0
            Eigen::Vector2d z = vi->GT;
            Eigen::Matrix2d z_precision = Eigen::Matrix2d::Identity() / (1e-4 * 1e-4);
            
            auto factor = std::make_unique<gbp::Factor>(fid++, std::vector<gbp::VariableNode*>{vi});
            factor->dim = 2;
            factor->measurements = {z};
            factor->precisions = {z_precision};
            
            // Compute anchor factor distribution (very strong)
            factor->computeFactor();
            
            // CRITICAL: Establish bidirectional relationship for message passing
            vi->adj_factors.push_back(gbp::AdjFactorRef{factor.get(), 0});  // vi is the 0-th variable
            vi->adj_factors_raw.push_back(factor.get());
            
            fg.factors.push_back(std::move(factor));
        }
    }
    
    return fg;
}

} // namespace slam
