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

    if (N <= 0) return graph;

    // Single RNG drives EVERYTHING, matching Python default_rng(seed)
    std::mt19937 rng(use_seed ? seed : std::random_device{}());
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    // ============ 1) Generate trajectory (Python: standard_normal + normalize) ============
    double x = 0.0, y = 0.0;
    std::vector<std::pair<double, double>> positions;
    positions.reserve(N);
    positions.push_back({x, y});

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

    // ============ 2) Create nodes ============
    graph.nodes.reserve(N);
    for (int i = 0; i < N; ++i) {
        SlamNode node;
        node.id = i;
        node.layer = 0;
        node.dim = 2;
        node.position = Eigen::Vector2d(positions[i].first, positions[i].second);
        node.GT = node.position;
        graph.nodes.push_back(node);
    }

    // ============ 3) Sequential edges (i -> i+1) ============
    // Python: for i in range(N-1): edges.append({source:i, target:i+1})
    for (int i = 0; i < N - 1; ++i) {
        SlamEdge e(i, i + 1);
        // is_prior/is_anchor remain false
        graph.edges.push_back(e);
    }

    // ============ 4) Loop closures ============
    // Python: for i in range(N): for j in range(i+5,N):
    //         if rng.random()<loop_prob and dist<loop_radius: add edge(i,j)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 5; j < N; ++j) {
            if (uniform01(rng) < loop_prob) {
                double xi = positions[i].first;
                double yi = positions[i].second;
                double xj = positions[j].first;
                double yj = positions[j].second;
                double dist = std::hypot(xi - xj, yi - yj);
                if (dist < loop_radius) {
                    SlamEdge e(i, j);
                    graph.edges.push_back(e);
                }
            }
        }
    }

    // ============ 5) Strong prior node selection (match numpy choice replace=False) ============
    std::vector<int> strong_ids;
    if (prior_prop <= 0.0) {
        strong_ids = {0};
    } else if (prior_prop >= 1.0) {
        strong_ids.resize(N);
        std::iota(strong_ids.begin(), strong_ids.end(), 0);
    } else {
        int k = std::max(1, static_cast<int>(std::floor(prior_prop * N)));

        std::vector<int> all_ids(N);
        std::iota(all_ids.begin(), all_ids.end(), 0);

        // closest to numpy rng.choice(N, k, replace=False)
        std::shuffle(all_ids.begin(), all_ids.end(), rng);

        strong_ids.insert(strong_ids.end(), all_ids.begin(), all_ids.begin() + k);
    }

    // ============ 6) Add prior edges ============
    // Python: edges.append({source:i, target:"prior"})
    for (int id : strong_ids) {
        SlamEdge prior_e;
        prior_e.source = id;
        prior_e.target = -1;     // marker for prior
        prior_e.is_prior = true;
        prior_e.is_anchor = false;
        graph.edges.push_back(prior_e);
    }

    // ============ 7) Add anchor edge (always) ============
    // Python: edges.append({source:0, target:"anchor"})
    {
        SlamEdge anchor_e;
        anchor_e.source = 0;
        anchor_e.target = -2;    // marker for anchor
        anchor_e.is_prior = true;
        anchor_e.is_anchor = true;
        graph.edges.push_back(anchor_e);
    }

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
