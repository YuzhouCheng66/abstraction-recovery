#include "slam/SlamGraph.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace slam {

SlamGraph makeSlamLikeGraph(
    int N,
    double step_size,
    double loop_prob,
    double loop_radius,
    double prior_prop,
    unsigned int seed,
    bool use_seed)
{
    SlamGraph graph;
    
    // Initialize random number generator
    std::mt19937 rng(use_seed ? seed : std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    // ============ Generate trajectory ============
    double x = 0.0, y = 0.0;
    std::vector<std::pair<double, double>> positions;
    positions.push_back({x, y});
    
    // Random walk: each step is a normalized Gaussian sample
    for (int i = 1; i < N; ++i) {
        double dx = normal_dist(rng);
        double dy = normal_dist(rng);
        double norm = std::sqrt(dx * dx + dy * dy) + 1e-6;
        dx = (dx / norm) * step_size;
        dy = (dy / norm) * step_size;
        x += dx;
        y += dy;
        positions.push_back({x, y});
    }
    
    // ============ Create nodes ============
    for (int i = 0; i < N; ++i) {
        SlamNode node;
        node.id = i;
        node.layer = 0;
        node.dim = 2;
        node.position = Eigen::Vector2d(positions[i].first, positions[i].second);
        node.GT = node.position;
        graph.nodes.push_back(node);
    }
    
    // ============ Sequential edges along the path ============
    for (int i = 0; i < N - 1; ++i) {
        SlamEdge edge(i, i + 1);
        graph.edges.push_back(edge);
    }
    
    // ============ Loop closure edges ============
    for (int i = 0; i < N; ++i) {
        for (int j = i + 5; j < N; ++j) {
            if (uniform_dist(rng) < loop_prob) {
                double xi = positions[i].first;
                double yi = positions[i].second;
                double xj = positions[j].first;
                double yj = positions[j].second;
                double dist = std::hypot(xi - xj, yi - yj);
                
                if (dist < loop_radius) {
                    SlamEdge edge(i, j);
                    graph.edges.push_back(edge);
                }
            }
        }
    }
    
    // ============ Determine nodes with strong priors ============
    std::vector<int> strong_ids;
    
    if (prior_prop <= 0.0) {
        // Only anchor node 0
        strong_ids.push_back(0);
    } else if (prior_prop >= 1.0) {
        // All nodes have strong priors
        for (int i = 0; i < N; ++i) {
            strong_ids.push_back(i);
        }
    } else {
        // Randomly select a proportion of nodes
        int k = std::max(1, static_cast<int>(std::floor(prior_prop * N)));
        std::vector<int> all_ids(N);
        std::iota(all_ids.begin(), all_ids.end(), 0);
        
        // Fisher-Yates shuffle to select k random nodes
        for (int i = 0; i < k; ++i) {
            int j = i + static_cast<int>(uniform_dist(rng) * (N - i));
            std::swap(all_ids[i], all_ids[j]);
        }
        strong_ids.insert(strong_ids.end(), all_ids.begin(), all_ids.begin() + k);
    }
    
    // ============ Add prior edges ============
    for (int id : strong_ids) {
        SlamEdge prior_edge;
        prior_edge.source = id;
        prior_edge.target = -1;  // Special marker for prior
        prior_edge.is_prior = true;
        prior_edge.is_anchor = false;
        graph.edges.push_back(prior_edge);
    }
    
    // ============ Add anchor edge for node 0 ============
    SlamEdge anchor_edge;
    anchor_edge.source = 0;
    anchor_edge.target = -2;  // Special marker for anchor
    anchor_edge.is_prior = true;
    anchor_edge.is_anchor = true;
    graph.edges.push_back(anchor_edge);
    
    return graph;
}

std::vector<Layer> initLayers(
    int N,
    double step_size,
    double loop_prob,
    double loop_radius,
    double prior_prop,
    unsigned int seed,
    bool use_seed)
{
    std::vector<Layer> layers;
    Layer base_layer;
    base_layer.name = "base";
    base_layer.graph = makeSlamLikeGraph(N, step_size, loop_prob, loop_radius, prior_prop, seed, use_seed);
    layers.push_back(base_layer);
    return layers;
}

} // namespace slam
