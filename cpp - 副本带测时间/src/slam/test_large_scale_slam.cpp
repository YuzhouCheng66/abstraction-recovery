// test_large_scale_slam.cpp

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <stdexcept>

#include <Eigen/Dense>

#include "slam/SlamGraph.h"
#include "slam/SlamFactorGraph.h"   // buildNoisyPoseGraph returns gbp::FactorGraph now
#include "gbp/FactorGraph.h"

// ---- proxy energy: 0.5 * sum ||mu_i - GT_i||^2 ----
static double energyMapFromGraphMu(const gbp::FactorGraph& graph) {
    double total = 0.0;

    for (const auto& vptr : graph.var_nodes) {
        if (!vptr) continue;
        const auto& v = *vptr;
        if (v.dofs < 2) continue;

        Eigen::Vector2d gt = v.GT.head(2);
        Eigen::Vector2d mu = v.mu.head(2);
        Eigen::Vector2d r  = mu - gt;
        total += 0.5 * r.dot(r);
    }

    return total;
}

// ---- same proxy energy, but using externally provided stacked mu (MAP solution) ----
// IMPORTANT: we do NOT assume off = 2*id. We use var_ix from jointDistributionInfSparse.
static double energyMapFromStackedMu(
    const gbp::FactorGraph& graph,
    const Eigen::VectorXd& mu_stacked,
    const std::vector<int>& var_ix)
{
    double total = 0.0;

    for (const auto& vptr : graph.var_nodes) {
        if (!vptr) continue;
        const auto& v = *vptr;
        if (v.dofs < 2) continue;

        const int id = v.variableID;
        if (id < 0 || id >= (int)var_ix.size() || var_ix[id] < 0) {
            throw std::runtime_error("energyMapFromStackedMu: missing var_ix for variableID");
        }

        const int off = var_ix[id];
        if (off + v.dofs > mu_stacked.size()) {
            throw std::runtime_error("energyMapFromStackedMu: mu_stacked too small for var offset");
        }

        Eigen::Vector2d gt = v.GT.head(2);
        Eigen::Vector2d mu = mu_stacked.segment(off, 2);
        Eigen::Vector2d r  = mu - gt;
        total += 0.5 * r.dot(r);
    }

    return total;
}

int main() {
    std::cout << "=== Large-Scale SLAM Graph Convergence Test ===\n\n";

    // ---------------- parameters ----------------
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

    // ---------------- Step 1: build SlamGraph ----------------
    std::cout << "Step 1: Generating SLAM-like graph with " << N << " nodes...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    slam::SlamGraph graph = slam::makeSlamLikeGraph(
        N, step, prob, radius, prior_prop, seed, true
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "  ✓ Generated " << graph.numNodes() << " nodes and " << graph.numEdges() << " edges\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    // ---------------- Step 2: build gbp::FactorGraph ----------------
    std::cout << "Step 2: Building noisy pose graph...\n";
    t0 = std::chrono::high_resolution_clock::now();

    slam::NoiseConfig config;
    config.prior_sigma = prior_sigma;
    config.odom_sigma  = odom_sigma;
    config.tiny_prior  = 1e-12;
    config.seed        = seed;
    config.use_seed    = true;

    std::cout << "  Creating factors from " << graph.numEdges() << " edges...\n";
    std::cout.flush();

    // IMPORTANT: buildNoisyPoseGraph now returns gbp::FactorGraph
    gbp::FactorGraph gbp_graph = slam::buildNoisyPoseGraph(graph.nodes, graph.edges, config);

    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "  ✓ Built factor graph with " << gbp_graph.var_nodes.size() << " variables\n";
    std::cout << "    and " << gbp_graph.factors.size() << " factors\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    // ---------------- Step 3: init beliefs (Python style) ----------------
    std::cout << "Step 3: Initializing beliefs...\n";

    for (auto& vptr : gbp_graph.var_nodes) {
        if (!vptr) continue;
        auto& v = *vptr;

        v.belief = v.prior;

        // Optional: initialize mu from prior (debug only)
        Eigen::MatrixXd lam = v.prior.lam();
        if (lam.size() > 0) {
            lam.diagonal().array() += 1e-12;
            Eigen::LLT<Eigen::MatrixXd> llt(lam);
            if (llt.info() == Eigen::Success) v.mu = llt.solve(v.prior.eta());
            else v.mu.setZero();
        } else {
            v.mu.setZero();
        }
    }


    std::cout << "  ✓ Initialized\n\n";

    // ---------------- Step 3.5: Batch MAP baseline (sparse) ----------------
    std::cout << "Step 3.5: Computing batch MAP (sparse) baseline...\n";
    t0 = std::chrono::high_resolution_clock::now();

    // (A) get var_ix mapping consistent with how jointMAPSparse stacks variables
    auto J = gbp_graph.jointDistributionInfSparse();

    // (B) solve for global MAP
    Eigen::VectorXd mu_opt = gbp_graph.jointMAPSparse(1e-12);

    // (C) compute GT proxy energy for the batch MAP
    double e_opt = energyMapFromStackedMu(gbp_graph, mu_opt, J.var_ix);

    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "  ✓ Batch MAP computed\n";
    std::cout << "  Batch-MAP proxy energy (vs GT): " << std::fixed << std::setprecision(6) << e_opt << "\n";
    std::cout << "  Time: " << ms << " ms\n\n";

    // ---------------- Step 4: run GBP iterations ----------------
    gbp_graph.eta_damping = 0.0;  // match你的当前实验设置
    std::cout << "Step 4: Running GBP iterations with eta_damping = " << gbp_graph.eta_damping << "...\n";
    std::cout << "  Target: Convergence (energy change < 1e-2 for 2 consecutive iterations)\n\n";

    double e_prev = energyMapFromGraphMu(gbp_graph);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial GBP proxy energy: " << e_prev
              << " | gap to batch-MAP: " << (e_prev - e_opt) << "\n\n";

    int counter = 0;
    const int max_iters = 2000;
    const double thr = 1e-2;
    const int patience = 2;


    t0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sync_total(0);
    int sync_iters = 0;

    for (int it = 0; it < max_iters; ++it) {
        auto sync_start = std::chrono::high_resolution_clock::now();
        gbp_graph.synchronousIteration();
        auto sync_end = std::chrono::high_resolution_clock::now();
        sync_total += (sync_end - sync_start);
        ++sync_iters;

        double e = energyMapFromGraphMu(gbp_graph);
        double de = std::abs(e_prev - e);

        if ((it + 1) % 10 == 0 || it < 20) {
            std::cout << "Iter " << std::setw(4) << (it + 1)
                      << " | Energy = " << std::setw(12) << e
                      << " | ΔE = " << std::scientific << std::setprecision(3) << de
                      << std::fixed << std::setprecision(6)
                      << " | gap_to_MAP = " << (e - e_opt) << "\n";
        }

        if (de < thr) {
            counter++;
            if (counter >= patience) {
                std::cout << "\n✓ CONVERGED at iteration " << (it + 1)
                          << " with GBP proxy energy = " << e
                          << " | gap_to_MAP = " << (e - e_opt) << "\n";
                break;
            }
        } else {
            counter = 0;
        }

        e_prev = e;
    }

    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\nOptimization time: " << ms << " ms\n";
    if (sync_iters > 0) {
        double avg_sync_ms = 1000.0 * sync_total.count() / sync_iters;
        std::cout << "Average synchronousIteration() time per iter: " << avg_sync_ms << " ms\n";
    }

    // ---------------- Step 5: final report ----------------
    std::cout << "\n=== Final Statistics ===\n";
    double e_final = energyMapFromGraphMu(gbp_graph);
    std::cout << "Final GBP proxy energy: " << e_final << "\n";
    std::cout << "Batch-MAP proxy energy: " << e_opt << "\n";
    std::cout << "Final gap (GBP - MAP): " << (e_final - e_opt) << "\n";

    std::cout << "\nSample final estimates (first 5 nodes):\n";
    for (int i = 0; i < std::min(5, (int)gbp_graph.var_nodes.size()); ++i) {
        if (!gbp_graph.var_nodes[i]) continue;
        const auto& v = *gbp_graph.var_nodes[i];

        Eigen::Vector2d gt = v.GT.head(2);
        Eigen::Vector2d mu = v.mu.head(2);
        Eigen::Vector2d err = mu - gt;

        std::cout << "  Node " << std::setw(4) << i
                  << " | GT = (" << std::setw(8) << gt.x() << ", " << std::setw(8) << gt.y() << ")"
                  << " | Est = (" << std::setw(8) << mu.x() << ", " << std::setw(8) << mu.y() << ")"
                  << " | Err = " << std::scientific << std::setprecision(2) << err.norm() << "\n";
    }

    std::cout << "\n=== Test Complete ===\n";
    return 0;
}
