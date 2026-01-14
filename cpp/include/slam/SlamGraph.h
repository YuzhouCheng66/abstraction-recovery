#pragma once

#include <vector>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <Eigen/Dense>

namespace slam {

/**
 * Represents a single node in the SLAM graph
 */
struct SlamNode {
    int id;
    int layer;
    int dim;
    Eigen::Vector2d position;
    Eigen::Vector2d GT;  // Ground truth position
    
    SlamNode() : id(0), layer(0), dim(2), position(0, 0), GT(0, 0) {}
    SlamNode(int id_, const Eigen::Vector2d& pos) 
        : id(id_), layer(0), dim(2), position(pos), GT(pos) {}
};

/**
 * Represents a single edge in the SLAM graph
 */
struct SlamEdge {
    int source;
    int target;
    bool is_prior;
    bool is_anchor;
    
    SlamEdge() : source(-1), target(-1), is_prior(false), is_anchor(false) {}
    SlamEdge(int src, int tgt) 
        : source(src), target(tgt), is_prior(false), is_anchor(false) {}
};

/**
 * Complete SLAM-like factor graph structure
 */
class SlamGraph {
public:
    std::vector<SlamNode> nodes;
    std::vector<SlamEdge> edges;
    
    SlamGraph() = default;
    
    int numNodes() const { return nodes.size(); }
    int numEdges() const { return edges.size(); }
};

/**
 * Generate a SLAM-like graph with trajectory and loop closures
 * 
 * @param N Number of nodes along the trajectory
 * @param step_size Distance between consecutive poses
 * @param loop_prob Probability of creating a loop closure
 * @param loop_radius Maximum distance threshold for loop closures
 * @param prior_prop Proportion of nodes with strong priors (0.0=anchor only, 1.0=all nodes)
 * @param seed Random seed for reproducibility (optional)
 * @return SlamGraph containing nodes and edges
 */
SlamGraph makeSlamLikeGraph(
    int N = 100,
    double step_size = 25.0,
    double loop_prob = 0.05,
    double loop_radius = 50.0,
    double prior_prop = 0.0,
    unsigned int seed = 0,
    bool use_seed = false
);

/**
 * Initialize layers from SLAM graph (future use for hierarchical structures)
 */
struct Layer {
    std::string name;
    SlamGraph graph;
};

std::vector<Layer> initLayers(
    int N = 100,
    double step_size = 25.0,
    double loop_prob = 0.05,
    double loop_radius = 50.0,
    double prior_prop = 0.0,
    unsigned int seed = 0,
    bool use_seed = false
);



} // namespace slam
