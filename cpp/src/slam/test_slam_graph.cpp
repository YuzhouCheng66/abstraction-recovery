#include <iostream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "slam/SlamGraph.h"
#include "slam/SlamFactorGraph.h"

void printVec(const Eigen::VectorXd& v, const std::string& name = "") {
    if (!name.empty()) std::cout << name << ": ";
    std::cout << "[";
    for (int i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(5) << v(i);
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== SLAM Graph Generation Test ===\n\n";
    
    // ========== Test 1: Generate SLAM-like graph ==========
    std::cout << "Test 1: Generating SLAM-like graph...\n";
    slam::SlamGraph graph = slam::makeSlamLikeGraph(
        20,      // N nodes
        25.0,    // step_size
        0.1,     // loop_prob
        60.0,    // loop_radius
        0.3,     // prior_prop (30% of nodes get strong priors)
        12345,   // seed
        true     // use_seed
    );
    
    std::cout << "  Nodes: " << graph.numNodes() << "\n";
    std::cout << "  Edges: " << graph.numEdges() << "\n";
    
    // Count edge types
    int sequential_edges = 0, loop_edges = 0, prior_edges = 0, anchor_edges = 0;
    for (const auto& e : graph.edges) {
        if (e.is_anchor) anchor_edges++;
        else if (e.is_prior) prior_edges++;
        else if (e.target < e.source) loop_edges++;
        else sequential_edges++;
    }
    
    std::cout << "  Sequential edges: " << sequential_edges << "\n";
    std::cout << "  Loop closures: " << loop_edges << "\n";
    std::cout << "  Prior edges: " << prior_edges << "\n";
    std::cout << "  Anchor edges: " << anchor_edges << "\n\n";
    
    // Show first few nodes
    std::cout << "First 5 nodes:\n";
    for (int i = 0; i < std::min(5, graph.numNodes()); ++i) {
        const auto& n = graph.nodes[i];
        std::cout << "  Node " << n.id << ": pos=(" << std::fixed << std::setprecision(2) 
                  << n.position.x() << ", " << n.position.y() << ")\n";
    }
    
    // ========== Test 2: Initialize layers ==========
    std::cout << "\nTest 2: Initializing layers...\n";
    auto layers = slam::initLayers(
        10,      // smaller graph for clarity
        20.0,
        0.15,
        50.0,
        0.5,
        54321,
        true
    );
    
    std::cout << "  Layers: " << layers.size() << "\n";
    std::cout << "  Layer name: \"" << layers[0].name << "\"\n";
    std::cout << "  Graph nodes: " << layers[0].graph.numNodes() << "\n";
    std::cout << "  Graph edges: " << layers[0].graph.numEdges() << "\n";
    
    // ========== Test 3: Build noisy pose graph ==========
    std::cout << "\nTest 3: Building noisy pose graph...\n";
    
    slam::NoiseConfig config;
    config.prior_sigma = 5.0;
    config.odom_sigma = 2.0;
    config.tiny_prior = 1e-10;
    config.seed = 99999;
    config.use_seed = true;
    
    slam::SimpleFactorGraph fg = slam::buildNoisyPoseGraph(graph.nodes, graph.edges, config);
    
    std::cout << "  Variable nodes: " << fg.getNumNodes() << "\n";
    std::cout << "  Factors: " << fg.getNumFactors() << "\n\n";
    
    // Show variable node information
    std::cout << "Variable node details (first 3):\n";
    for (int i = 0; i < std::min(3, (int)fg.var_nodes.size()); ++i) {
        const auto& v = fg.var_nodes[i];
        std::cout << "  Var " << v->id << ": GT=";
        std::cout << "(" << std::fixed << std::setprecision(3) 
                  << v->GT.x() << ", " << v->GT.y() << ")";
        std::cout << ", prior.lam(0,0)=" << v->prior.lam()(0, 0);
        std::cout << ", adj_factors=" << v->adj_factors_raw.size() << "\n";
    }
    
    // ========== Test 4: Simple factor graph optimization ==========
    std::cout << "\nTest 4: Running simple GBP iterations...\n";
    
    // Initialize beliefs
    for (auto& v : fg.var_nodes) {
        v->updateBelief();
    }
    
    // Run a few iterations
    std::cout << "Initial state:\n";
    std::cout << "  x0.mu = "; printVec(fg.var_nodes[0]->mu);
    std::cout << "  x1.mu = "; printVec(fg.var_nodes[1]->mu);
    
    for (int iter = 1; iter <= 5; ++iter) {
        fg.synchronousIteration();
        
        std::cout << "\nAfter iteration " << iter << ":\n";
        std::cout << "  x0.mu = "; printVec(fg.var_nodes[0]->mu);
        std::cout << "  x1.mu = "; printVec(fg.var_nodes[1]->mu);
    }
    
    std::cout << "\n=== All tests completed successfully ===\n";
    
    return 0;
}
