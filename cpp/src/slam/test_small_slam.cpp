#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "slam/SlamGraph.h"
#include "slam/SlamFactorGraph.h"

int main() {
    std::cout << "=== Small SLAM Test (3 nodes) ===\n\n";
    
    // Create a tiny graph manually
    std::vector<slam::SlamNode> nodes;
    nodes.push_back(slam::SlamNode(0, Eigen::Vector2d(0.0, 0.0)));
    nodes.push_back(slam::SlamNode(1, Eigen::Vector2d(-25.0, -0.978)));
    nodes.push_back(slam::SlamNode(2, Eigen::Vector2d(-48.5, 7.53)));
    
    // Set ground truth
    nodes[0].GT = nodes[0].position;
    nodes[1].GT = nodes[1].position;
    nodes[2].GT = nodes[2].position;
    
    std::vector<slam::SlamEdge> edges;
    
    // Anchor on node 0
    slam::SlamEdge anchor;
    anchor.source = 0;
    anchor.is_anchor = true;
    edges.push_back(anchor);
    
    // Odometry: 0 -> 1
    slam::SlamEdge e01;
    e01.source = 0;
    e01.target = 1;
    e01.is_prior = false;
    e01.is_anchor = false;
    edges.push_back(e01);
    
    // Odometry: 1 -> 2
    slam::SlamEdge e12;
    e12.source = 1;
    e12.target = 2;
    e12.is_prior = false;
    e12.is_anchor = false;
    edges.push_back(e12);
    
    // Build factor graph
    slam::NoiseConfig config;
    config.prior_sigma = 1.0;
    config.odom_sigma = 1.0;
    config.tiny_prior = 1e-12;
    config.seed = 2001;
    config.use_seed = true;
    
    auto fbp = slam::buildNoisyPoseGraph(nodes, edges, config);
    
    std::cout << "Graph: " << fbp.getNumNodes() << " variables, " << fbp.getNumFactors() << " factors\n\n";
    
    // Initialize with iterative message passing
    std::cout << "=== Initialization (iterative bootstrap) ===\n";
    
    // Iteration 0: Just sync initial beliefs (from priors)
    for (auto& f : fbp.factors) {
        f->syncAdjBeliefsFromVariables();
    }
    
    for (int iter = 0; iter < 3; ++iter) {
        std::cout << "  Bootstrap iter " << iter << "\n";
        
        // Compute messages based on current beliefs
        for (auto& f : fbp.factors) {
            f->computeMessages(0.0);
        }
        
        // Update beliefs from messages
        for (auto& v : fbp.var_nodes) {
            v->updateBelief();
        }
        
        // Sync updated beliefs back to factors
        for (auto& v : fbp.var_nodes) {
            v->sendBeliefToFactors();
        }
    }
    
    std::cout << "Done with bootstrap initialization\n";
    
    std::cout << "\nAfter first message computation:\n";
    for (int i = 0; i < fbp.getNumFactors(); ++i) {
        const auto& f = fbp.factors[i];
        std::cout << "Factor " << f->id << ":\n";
        for (int j = 0; j < (int)f->messages.size(); ++j) {
            std::cout << "  msg[" << j << "]: eta norm=" << f->messages[j].eta().norm()
                      << ", lam norm=" << f->messages[j].lam().norm();
            if (f->messages[j].lam().rows() <= 2) {
                std::cout << "\n    lam:\n" << f->messages[j].lam();
                std::cout << "\n    eta: " << f->messages[j].eta().transpose() << "\n";
            } else {
                std::cout << "\n";
            }
        }
    }
    
    for (auto& v : fbp.var_nodes) {
        v->updateBelief();
    }
    
    std::cout << "\nAfter updateBelief:\n";
    for (int i = 0; i < fbp.getNumNodes(); ++i) {
        const auto& v = fbp.var_nodes[i];
        std::cout << "  Node " << v->id << ": prior lam norm=" << v->prior.lam().norm()
                  << ", belief eta norm=" << v->belief.eta().norm()
                  << ", mu=(" << v->mu(0) << ", " << v->mu(1) << ")\n";
        if (i == 0) {
            std::cout << "    prior lam:\n" << v->prior.lam() << "\n";
            std::cout << "    belief lam:\n" << v->belief.lam() << "\n";
        }
    }
    
    return 0;
}
