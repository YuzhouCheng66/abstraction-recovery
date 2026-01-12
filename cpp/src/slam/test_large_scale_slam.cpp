#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>
#include "slam/SlamGraph.h"
#include "slam/SlamFactorGraph.h"

/**
 * Compute energy map (sum of squared distances from ground truth)
 * 
 * Energy = 0.5 * sum_i ||mu_i - GT_i||^2
 */
double energyMap(const slam::SimpleFactorGraph& graph, bool include_priors = true, bool include_factors = true) {
    double total = 0.0;
    
    for (const auto& v : graph.var_nodes) {
        if (v->dim < 2) continue;
        
        // Extract 2D position
        Eigen::Vector2d gt = v->GT.head(2);
        Eigen::Vector2d mu = v->mu.head(2);
        Eigen::Vector2d residual = mu - gt;
        
        total += 0.5 * residual.dot(residual);
    }
    
    return total;
}

int main() {
    std::cout << "=== Large-Scale SLAM Graph Convergence Test ===\n\n";
    
    // Test parameters
    const int N = 5000;
    const double step = 25.0;
    const double prob = 0.05;
    const double radius = 50.0;
    const double prior_prop = 0.02;
    const double prior_sigma = 1.0;
    const double odom_sigma = 1.0;
    const unsigned int seed = 2001;
    
    std::cout << "Parameters:\n";
    std::cout << "  N (nodes): " << N << "\n";
    std::cout << "  step_size: " << step << "\n";
    std::cout << "  loop_prob: " << prob << "\n";
    std::cout << "  loop_radius: " << radius << "\n";
    std::cout << "  prior_prop: " << prior_prop << "\n";
    std::cout << "  prior_sigma: " << prior_sigma << "\n";
    std::cout << "  odom_sigma: " << odom_sigma << "\n";
    std::cout << "  seed: " << seed << "\n\n";
    
    // ============ Step 1: Generate SLAM graph ============
    std::cout << "Step 1: Generating SLAM-like graph with " << N << " nodes...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    slam::SlamGraph graph = slam::makeSlamLikeGraph(
        N,           // N nodes
        step,        // step_size
        prob,        // loop_prob
        radius,      // loop_radius
        prior_prop,  // prior_prop
        seed,        // seed
        true         // use_seed
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  ✓ Generated " << graph.numNodes() << " nodes and " << graph.numEdges() << " edges\n";
    std::cout << "  Time: " << duration.count() << " ms\n\n";
    
    // ============ Step 2: Build noisy pose graph ============
    std::cout << "Step 2: Building noisy pose graph...\n";
    std::cout << "  Generating noise cache...\n";
    start = std::chrono::high_resolution_clock::now();
    
    slam::NoiseConfig config;
    config.prior_sigma = prior_sigma;
    config.odom_sigma = odom_sigma;
    config.tiny_prior = 1e-12;
    config.seed = seed;
    config.use_seed = true;
    
    std::cout << "  Creating " << graph.numEdges() << " factors...\n";
    std::cout.flush();
    
    slam::SimpleFactorGraph gbp_graph = slam::buildNoisyPoseGraph(
        graph.nodes,
        graph.edges,
        config
    );
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "  ✓ Built factor graph with " << gbp_graph.getNumNodes() << " variables\n";
    std::cout << "    and " << gbp_graph.getNumFactors() << " factors\n";
    std::cout << "  Time: " << duration.count() << " ms\n\n";
    
    // ============ Step 3: Initialize beliefs ============
    std::cout << "Step 3: Initializing beliefs...\n";
    
    // Debug: print first factor details
    if (!gbp_graph.factors.empty()) {
        const auto& f = gbp_graph.factors[0];
        std::cout << "  Factor 0: lam shape " << f->factor.lam().rows() << "x" << f->factor.lam().cols()
                  << ", eta norm = " << f->factor.eta().norm() << "\n";
    }
    
    // Do one iteration of message passing to bootstrap beliefs from factors
    std::cout << "  Computing initial messages...\n";
    
    // CRITICAL: Initialize variable beliefs from their priors
    for (auto& v : gbp_graph.var_nodes) {
        v->belief = v->prior;
        // mu will be set by Schur solve, but for initialization with singular lam, use prior eta direction
        if (v->prior.lam().norm() > 0) {
            // Try to compute mu from prior (will be zero if prior is tiny)
            Eigen::MatrixXd lam = v->prior.lam();
            lam.diagonal().array() += 1e-12;
            Eigen::LLT<Eigen::MatrixXd> llt(lam);
            v->mu = llt.solve(v->prior.eta());
        } else {
            v->mu.setZero();
        }
    }
    
    // Synchronize adj_beliefs from variables (priors ONLY, not cumulative beliefs)
    for (auto& f : gbp_graph.factors) {
        f->syncAdjBeliefsFromVariables();
    }
    
    // Now compute messages with adj_beliefs initialized from priors only
    for (auto& f : gbp_graph.factors) {
        f->computeMessages(0.0);  // Compute messages without damping
    }
    for (auto& v : gbp_graph.var_nodes) {
        v->updateBelief();
    }
    for (auto& v : gbp_graph.var_nodes) {
        v->sendBeliefToFactors();  // Send beliefs to factors for next iteration
    }
    
    std::cout << "  ✓ Initialized\n\n";
    
    // ============ Step 4: Run optimization iterations ============
    std::cout << "Step 4: Running GBP iterations with eta_damping = 0.4...\n";
    std::cout << "  Target: Convergence (energy change < 1e-2 for 2 consecutive iterations)\n\n";
    
    // Debug: check first variable after initialization
    if (!gbp_graph.var_nodes.empty()) {
        const auto& v1 = gbp_graph.var_nodes[1];
        std::cout << "  Before Iter 1: Node 1\n";
        std::cout << "    belief eta norm = " << v1->belief.eta().norm()
                  << ", belief lam norm = " << v1->belief.lam().norm() << "\n";
        std::cout << "    mu = (" << v1->mu(0) << ", " << v1->mu(1) << ")\n";
        std::cout << "    prior eta norm = " << v1->prior.eta().norm()
                  << ", prior lam norm = " << v1->prior.lam().norm() << "\n";
        
        // Check message from first factor to node 1
        if (v1->adj_factors.size() > 0) {
            const auto& msg = v1->adj_factors[0].factor->messages[v1->adj_factors[0].local_idx];
            std::cout << "    First message eta norm = " << msg.eta().norm()
                      << ", msg lam norm = " << msg.lam().norm() << "\n";
        }
    }
    
    double energy_prev = energyMap(gbp_graph);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial energy: " << energy_prev << "\n\n";
    
    int counter = 0;
    const int max_iters = 2000;
    const double energy_threshold = 1e-2;
    const int convergence_patience = 2;
    
    start = std::chrono::high_resolution_clock::now();
    
    for (int it = 0; it < max_iters; ++it) {
        gbp_graph.synchronousIteration();
        
        double energy = energyMap(gbp_graph);
        double delta_energy = std::abs(energy_prev - energy);
        
        // Debug output after iter 1 and 2
        if (it < 2) {
            const auto& v1 = gbp_graph.var_nodes[1];
            std::cout << "\nAfter Iter " << (it + 1) << ":\n";
            std::cout << "  Node 1 GT = (" << v1->GT(0) << ", " << v1->GT(1) << ")\n";
            std::cout << "  Node 1 mu = (" << v1->mu(0) << ", " << v1->mu(1) << ")\n";
            std::cout << "  Node 1 belief.eta = (" << v1->belief.eta()(0) << ", " 
                      << v1->belief.eta()(1) << ")\n";
            std::cout << "  Node 1 belief.lam.norm = " << v1->belief.lam().norm() << "\n";
            std::cout << "  Node 1 belief.lam diagonal = (" << v1->belief.lam()(0,0) 
                      << ", " << v1->belief.lam()(1,1) << ")\n";
            
            // Check factors connected to v1
            if (v1->adj_factors.size() > 0) {
                const auto& aref = v1->adj_factors[0];
                const auto& factor = aref.factor;
                std::cout << "  First factor to Node 1:\n";
                std::cout << "    factor.lam.norm = " << factor->factor.lam().norm() << "\n";
                std::cout << "    msg.eta = (" << factor->messages[aref.local_idx].eta()(0) 
                          << ", " << factor->messages[aref.local_idx].eta()(1) << ")\n";
                std::cout << "    msg.lam.norm = " << factor->messages[aref.local_idx].lam().norm() << "\n";
            }
            std::cout.flush();
        }
        
        // Print every iteration (or less frequently for large graphs)
        if ((it + 1) % 10 == 0 || it < 50) {
            std::cout << "Iter " << std::setw(4) << (it + 1) 
                      << " | Energy = " << std::setw(12) << energy 
                      << " | ΔE = " << std::scientific << std::setprecision(3) << delta_energy 
                      << std::fixed << std::setprecision(6) << "\n";
        }
        
        // Check for convergence
        if (delta_energy < energy_threshold) {
            counter++;
            if (counter >= convergence_patience) {
                std::cout << "\n✓ CONVERGED at iteration " << (it + 1) 
                          << " with energy = " << energy << "\n";
                break;
            }
        } else {
            counter = 0;  // Reset counter if energy change is significant
        }
        
        energy_prev = energy;
    }
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\nOptimization time: " << duration.count() << " ms\n";
    
    // ============ Step 5: Final statistics ============
    std::cout << "\n=== Final Statistics ===\n";
    double final_energy = energyMap(gbp_graph);
    std::cout << "Final energy: " << final_energy << "\n";
    
    // Sample some final estimates
    std::cout << "\nSample final estimates (first 5 nodes):\n";
    for (int i = 0; i < std::min(5, gbp_graph.getNumNodes()); ++i) {
        const auto& v = gbp_graph.var_nodes[i];
        Eigen::Vector2d gt = v->GT.head(2);
        Eigen::Vector2d mu = v->mu.head(2);
        Eigen::Vector2d err = mu - gt;
        
        std::cout << "  Node " << std::setw(4) << i 
                  << " | GT = (" << std::setw(8) << gt.x() << ", " << std::setw(8) << gt.y() << ")"
                  << " | Est = (" << std::setw(8) << mu.x() << ", " << std::setw(8) << mu.y() << ")"
                  << " | Err = " << std::scientific << std::setprecision(2) << err.norm() << "\n";
    }
    
    std::cout << "\n=== Test Complete ===\n";
    
    return 0;
}
