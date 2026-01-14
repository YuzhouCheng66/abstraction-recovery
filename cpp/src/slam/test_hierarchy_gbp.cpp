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
#include "gbp/HierarchyGBP.h"       // NEW: for SuperLayer + HierarchyGBP
#include "gbp/Factor.h"             // for printComputeFactorProfile

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

// ---- scatter super.mu back to base 2D mu (order grouping + local_idx) ----
// Returns a vector of size 2*N where entry (2*i:2*i+2) is the recovered 2D estimate for base node i.
static Eigen::VectorXd scatterSuperMuToBase2D(
    const gbp::FactorGraph& base_graph,
    const gbp::SuperLayer& super)
{
    const int n_base = (int)base_graph.var_nodes.size();
    Eigen::VectorXd mu2 = Eigen::VectorXd::Zero(2 * n_base);

    for (int bid = 0; bid < n_base; ++bid) {
        if (!base_graph.var_nodes[bid]) continue;
        const auto& bv = *base_graph.var_nodes[bid];
        if (bv.dofs < 2) continue;

        if (bid < 0 || bid >= (int)super.node_map.size()) {
            throw std::runtime_error("scatterSuperMuToBase2D: bid out of node_map range");
        }
        const int sid = super.node_map[bid];

        if (!super.graph) {
            throw std::runtime_error("scatterSuperMuToBase2D: super.graph is null");
        }
        if (sid < 0 || sid >= (int)super.graph->var_nodes.size() || !super.graph->var_nodes[sid]) {
            throw std::runtime_error("scatterSuperMuToBase2D: invalid super sid");
        }

        const auto& sv = *super.graph->var_nodes[sid];

        if (sid < 0 || sid >= (int)super.local_idx.size()) {
            throw std::runtime_error("scatterSuperMuToBase2D: sid out of local_idx range");
        }
        const auto it = super.local_idx[sid].find(bid);
        if (it == super.local_idx[sid].end()) {
            throw std::runtime_error("scatterSuperMuToBase2D: missing local_idx for (sid,bid)");
        }

        const int off = it->second.first;
        if (off + 2 > sv.mu.size()) {
            throw std::runtime_error("scatterSuperMuToBase2D: super.mu too small for slice");
        }

        mu2.segment(2 * bid, 2) = sv.mu.segment(off, 2);
    }

    return mu2;
}

// ---- proxy energy on base GT using externally provided base 2D estimates ----
static double energyMapFromBaseGT2D(
    const gbp::FactorGraph& base_graph,
    const Eigen::VectorXd& base_mu2) // size = 2*N
{
    const int n_base = (int)base_graph.var_nodes.size();
    if (base_mu2.size() != 2 * n_base) {
        throw std::runtime_error("energyMapFromBaseGT2D: base_mu2 has wrong size");
    }

    double total = 0.0;
    for (int bid = 0; bid < n_base; ++bid) {
        if (!base_graph.var_nodes[bid]) continue;
        const auto& bv = *base_graph.var_nodes[bid];
        if (bv.dofs < 2) continue;

        Eigen::Vector2d gt = bv.GT.head(2);
        Eigen::Vector2d mu = base_mu2.segment(2 * bid, 2);
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

    // ---------------- hierarchy test params (only for testing) ----------------
    const int k_group = 10;          // order grouping: group size
    const int super_iters = 500;     // run a few iters on super graph
    const double super_eta_damping = 0.0;

    std::cout << "Parameters:\n";
    std::cout << "  N (nodes): " << N << "\n";
    std::cout << "  step_size: " << step << "\n";
    std::cout << "  loop_prob: " << prob << "\n";
    std::cout << "  loop_radius: " << radius << "\n";
    std::cout << "  prior_prop: " << prior_prop << "\n";
    std::cout << "  prior_sigma: " << prior_sigma << "\n";
    std::cout << "  odom_sigma: " << odom_sigma << "\n";
    std::cout << "  seed: " << seed << "\n";
    std::cout << "  [Hierarchy test] k_group=" << k_group
              << ", super_iters=" << super_iters
              << ", super_eta_damping=" << super_eta_damping << "\n\n";

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

    // make factor-side adj_beliefs consistent once
    gbp_graph.syncAllFactorAdjBeliefs();

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
            std::cout << "BGraph Iter " << std::setw(4) << (it + 1)
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

    // ---------------- Step 4.5: HierarchyGBP (super) quick test ----------------
    std::cout << "\nStep 4.5: HierarchyGBP super-graph quick test...\n";
    std::cout << "  Build super graph (order grouping), bottom-up inject base beliefs, run a few super iterations.\n";

    // Move gbp_graph into shared_ptr for HierarchyGBP
    auto base_ptr = std::make_shared<gbp::FactorGraph>(std::move(gbp_graph));

    gbp::HierarchyGBP H(k_group);
    auto super = H.buildSuperFromBase(base_ptr, super_eta_damping);

    // Inject from base -> super
    H.bottomUpUpdateSuper(base_ptr, super);

    // Run some super iterations
    super->graph->eta_damping = super_eta_damping;
    auto super_t0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> super_sync_total(0);
    for (int it = 0; it < super_iters; ++it) {
        auto super_sync_start = std::chrono::high_resolution_clock::now();
        super->graph->synchronousIteration();
        auto super_sync_end = std::chrono::high_resolution_clock::now();
        super_sync_total += (super_sync_end - super_sync_start);
    }
    auto super_t1 = std::chrono::high_resolution_clock::now();
    auto super_ms = std::chrono::duration_cast<std::chrono::milliseconds>(super_t1 - super_t0).count();

    // Recover super solution back to base-2D and compute proxy energy vs base GT
    Eigen::VectorXd mu2_from_super = scatterSuperMuToBase2D(*base_ptr, *super);
    double e_super_to_base = energyMapFromBaseGT2D(*base_ptr, mu2_from_super);

    std::cout << "  ✓ Super test done\n";
    std::cout << "  Super optimization time: " << super_ms << " ms\n";
    if (super_iters > 0) {
        double avg_super_sync_ms = 1000.0 * super_sync_total.count() / super_iters;
        std::cout << "  Average super synchronousIteration() time per iter: " << avg_super_sync_ms << " ms\n";
    }
    std::cout << "  Final super->base proxy energy (vs base GT): " << std::fixed << std::setprecision(6)
              << e_super_to_base << "\n";


    // ---------------- Step 4.6: Abs build + bottom-up + top-down smoke test ----------------
    // Goal: just ensure the Abs-layer build/modify + top-down projections run without errors.
    const int r_reduced = 2;
    const double abs_eta_damping = 0.4;

    std::cout << "\n[Abs] Building abs graph from super...\n";
    auto abs = H.buildAbsFromSuper(super->graph, r_reduced, abs_eta_damping);

    std::cout << "[Abs] Bottom-up update abs from super...\n";
    H.bottomUpUpdateAbs(super->graph, abs, r_reduced, abs_eta_damping);

    // Run a couple of abs iterations (smoke test)
    abs->graph->eta_damping = abs_eta_damping;
    for (int it = 0; it < 2; ++it) {
        abs->graph->synchronousIteration();
    }

    // Top-down passes (Python parity):
    //  - abs -> super (lift mu back, refresh beliefs, and adjust factor messages)
    //  - super -> base (split mu down, refresh beliefs, and adjust factor messages)
    std::cout << "[TopDown] abs -> super...\n";
    H.topDownModifySuperFromAbs(super->graph, abs);

    std::cout << "[TopDown] super -> base...\n";
    H.topDownModifyBaseFromSuper(base_ptr, super);

    // Optional: one base iteration to ensure the graph remains consistent
    base_ptr->synchronousIteration();

    std::cout << "  ✓ Abs/TopDown smoke test done\n";

    // ---------------- Step 5: final report ----------------
    std::cout << "\n=== Final Statistics ===\n";
    double e_final = energyMapFromGraphMu(*base_ptr);
    std::cout << "Final GBP proxy energy: " << e_final << "\n";
    std::cout << "Batch-MAP proxy energy: " << e_opt << "\n";
    std::cout << "Final gap (GBP - MAP): " << (e_final - e_opt) << "\n";
    std::cout << "Final super->base proxy energy: " << e_super_to_base << "\n";
    std::cout << "Gap (super->base - MAP): " << (e_super_to_base - e_opt) << "\n";

    std::cout << "\nSample final estimates (first 5 nodes):\n";
    for (int i = 0; i < std::min(5, (int)base_ptr->var_nodes.size()); ++i) {
        if (!base_ptr->var_nodes[i]) continue;
        const auto& v = *base_ptr->var_nodes[i];

        Eigen::Vector2d gt = v.GT.head(2);
        Eigen::Vector2d mu = v.mu.head(2);
        Eigen::Vector2d err = mu - gt;

        std::cout << "  Node " << std::setw(4) << i
                  << " | GT = (" << std::setw(8) << gt.x() << ", " << std::setw(8) << gt.y() << ")"
                  << " | Est = (" << std::setw(8) << mu.x() << ", " << std::setw(8) << mu.y() << ")"
                  << " | Err = " << std::scientific << std::setprecision(2) << err.norm() << "\n";
    }

    // ========================
    //  Hierarchy bottom-up test
    // ========================
    std::cout << "\n[HierarchyGBP test] Build super & run bottomUpUpdateSuper\n";

    // Build hierarchy (reusing base_ptr)
    gbp::HierarchyGBP H2(8);  // e.g. group size k=8
    auto super_test = H2.buildSuperFromBase(base_ptr, 0.0);

    // Inject base beliefs -> super
    H2.bottomUpUpdateSuper(base_ptr, super_test);

    // Now check super.mu vs base.mu concatenation
    bool ok = true;

    for (int sid = 0; sid < (int)super_test->groups.size(); ++sid) {
        const auto& sv = *super_test->graph->var_nodes[sid];
        int D = super_test->total_dofs[sid];

        Eigen::VectorXd ref = Eigen::VectorXd::Zero(D);
        int off = 0;

        for (int bid : super_test->groups[sid]) {
            const auto& bv = *base_ptr->var_nodes[bid];
            ref.segment(off, bv.dofs) = bv.mu;
            off += bv.dofs;
        }

        double err2 = (sv.mu - ref).norm();

        if (sid < 5) {
            std::cout << "Super " << sid
                      << " | mu error vs base stack = " << err2 << "\n";
        }

        if (err2 > 1e-8) {
            ok = false;
        }
    }

    if (ok)
        std::cout << "✓ bottomUpUpdateSuper PASSED: super.mu matches base.mu stacking\n";
    else
        std::cout << "✗ bottomUpUpdateSuper FAILED: mismatch detected\n";

    // =======================
    // Step 5: V-loop demo (HierarchyGBP::vLoop)  [Python VGraph.vloop parity]
    // 5 layers init order: base -> super1 -> abs1 -> super2 -> abs2
    // Then vLoop() only modifies/iterates already-built layers.
    // =======================
    std::cout << "\nStep 5: Running 5-layer V-loop (base -> super1 -> abs1 -> super2 -> abs2) ...\n";

    // ---- (A) base: ensure it has run at least once ----
    std::cout << "  [Init] base: one synchronousIteration() before building super1...\n";
    base_ptr->eta_damping = 0.0;
    base_ptr->synchronousIteration(false);

    // ---- (B) build super1 from base (outside vLoop) ----
    std::cout << "  [Init] build super1 from base...\n";
    auto super1 = H.buildSuperFromBase(base_ptr, /*eta_damping=*/0.0);
    H.bottomUpUpdateSuper(base_ptr, super1);

    // ---- (C) super1: ensure it has run at least once ----
    std::cout << "  [Init] super1: one synchronousIteration() before building abs1...\n";
    super1->graph->eta_damping = 0.0;
    super1->graph->synchronousIteration(false);

    // ---- (D) build abs1 from super1 (outside vLoop) ----
    std::cout << "  [Init] build abs1 from super1...\n";
    auto abs1 = H.buildAbsFromSuper(super1->graph, /*r_reduced=*/2, /*eta_damping=*/0.0);
    H.bottomUpUpdateAbs(super1->graph, abs1, /*r_reduced=*/2, /*eta_damping=*/0.0);

    // ---- (E) abs1: ensure it has run at least once ----
    std::cout << "  [Init] abs1: one synchronousIteration() before building super2...\n";
    abs1->graph->eta_damping = 0.0;
    abs1->graph->synchronousIteration(false);

    // ---- (F) build super2 from abs1 (treating abs1 as base for next level) ----
    std::cout << "  [Init] build super2 from abs1...\n";
    auto super2 = H.buildSuperFromBase(abs1->graph, /*eta_damping=*/0.0);
    H.bottomUpUpdateSuper(abs1->graph, super2);

    // ---- (G) super2: ensure it has run at least once ----
    std::cout << "  [Init] super2: one synchronousIteration() before building abs2...\n";
    super2->graph->eta_damping = 0.0;
    super2->graph->synchronousIteration(false);

    // ---- (H) build abs2 from super2 (outside vLoop) ----
    std::cout << "  [Init] build abs2 from super2...\n";
    auto abs2 = H.buildAbsFromSuper(super2->graph, /*r_reduced=*/2, /*eta_damping=*/0.0);
    H.bottomUpUpdateAbs(super2->graph, abs2, /*r_reduced=*/2, /*eta_damping=*/0.0);

    // ---- (I) pack 5 layers for vLoop ----
    std::vector<gbp::HierarchyGBP::VLayerEntry> vLayers;
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"base",   base_ptr,       nullptr, nullptr});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"super1", super1->graph,  super1,  nullptr});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"abs1",   abs1->graph,    nullptr, abs1});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"super2", super2->graph,  super2,  nullptr});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"abs2",   abs2->graph,    nullptr, abs2});

    // ---- (F) iterate V-loop with detailed profiling ----
    double energy_prev = 0.0;
    int stable = 0;
    std::chrono::duration<double> vloop_total(0);
    int iters_done = 0;

    // Profiling accumulators
    double t_super1_bottomUp = 0, t_super1_iter = 0;
    double t_abs1_bottomUp = 0, t_abs1_iter = 0;
    double t_super2_bottomUp = 0, t_super2_iter = 0;
    double t_abs2_bottomUp = 0, t_abs2_iter = 0;
    double t_abs2_topDown_iter = 0, t_abs2_topDown = 0;
    double t_super2_topDown_iter = 0, t_super2_topDown = 0;
    double t_abs1_topDown_iter = 0, t_abs1_topDown = 0;
    double t_super1_topDown_iter = 0, t_super1_topDown = 0;

    for (int it = 0; it < 500; ++it) {
        auto vloop_start = std::chrono::high_resolution_clock::now();
        
        // ========== BOTTOM-UP ==========
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // super1: no bottom-up modify, just iteration
        auto t1 = std::chrono::high_resolution_clock::now();
        vLayers[1].graph->synchronousIteration(false);
        auto t2 = std::chrono::high_resolution_clock::now();
        t_super1_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // abs1: bottom-up update + iteration
        t1 = std::chrono::high_resolution_clock::now();
        H.bottomUpUpdateAbs(vLayers[1].graph, abs1, /*r_reduced=*/2, /*eta_damping=*/0.0);
        vLayers[2].graph = abs1->graph;
        t2 = std::chrono::high_resolution_clock::now();
        t_abs1_bottomUp += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[2].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_abs1_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // super2: bottom-up update + iteration
        t1 = std::chrono::high_resolution_clock::now();
        H.bottomUpUpdateSuper(vLayers[2].graph, super2);
        vLayers[3].graph = super2->graph;
        t2 = std::chrono::high_resolution_clock::now();
        t_super2_bottomUp += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[3].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_super2_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // abs2: bottom-up update + iteration
        t1 = std::chrono::high_resolution_clock::now();
        H.bottomUpUpdateAbs(vLayers[3].graph, abs2, /*r_reduced=*/2, /*eta_damping=*/0.0);
        vLayers[4].graph = abs2->graph;
        t2 = std::chrono::high_resolution_clock::now();
        t_abs2_bottomUp += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[4].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_abs2_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // ========== TOP-DOWN ==========
        // abs2: iteration + top-down to super2
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[4].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_abs2_topDown_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        H.topDownModifySuperFromAbs(vLayers[3].graph, abs2);
        t2 = std::chrono::high_resolution_clock::now();
        t_abs2_topDown += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // super2: iteration + top-down to abs1
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[3].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_super2_topDown_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        H.topDownModifyBaseFromSuper(vLayers[2].graph, super2);
        t2 = std::chrono::high_resolution_clock::now();
        t_super2_topDown += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // abs1: iteration + top-down to super1
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[2].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_abs1_topDown_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        H.topDownModifySuperFromAbs(vLayers[1].graph, abs1);
        t2 = std::chrono::high_resolution_clock::now();
        t_abs1_topDown += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // super1: iteration + top-down to base
        t1 = std::chrono::high_resolution_clock::now();
        vLayers[1].graph->synchronousIteration(false);
        t2 = std::chrono::high_resolution_clock::now();
        t_super1_topDown_iter += std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        t1 = std::chrono::high_resolution_clock::now();
        H.topDownModifyBaseFromSuper(vLayers[0].graph, super1);
        t2 = std::chrono::high_resolution_clock::now();
        t_super1_topDown += std::chrono::duration<double, std::milli>(t2 - t1).count();

        auto vloop_end = std::chrono::high_resolution_clock::now();

        vloop_total += (vloop_end - vloop_start);
        ++iters_done;

        double vloop_ms = std::chrono::duration<double, std::milli>(vloop_end - vloop_start).count();

        const double energy = energyMapFromGraphMu(*vLayers[0].graph);
        const double gap_to_map = energy - e_opt;
        if ((it + 1) % 10 == 0 || it < 20) {
            std::cout << "  VLoop Iter " << std::setw(3) << (it + 1)
                    << " | Energy = " << std::fixed << std::setprecision(6) << energy
                    << " | gap_to_MAP = " << std::setprecision(6) << gap_to_map
                    << " | Time = " << std::setprecision(2) << vloop_ms << " ms\n";
        }

        if (std::abs(energy_prev - energy) < 1e-2) {
            stable++;
            if (stable >= 2) break;
        } else {
            stable = 0;
        }
        energy_prev = energy;
    }

    double avg_vloop_ms = (iters_done > 0) ? (1000.0 * vloop_total.count() / iters_done) : 0.0;
    std::cout << "  ✓ V-loop demo finished.\n";
    std::cout << "  Total V-loop time: " << std::fixed << std::setprecision(0)
            << (1000.0 * vloop_total.count()) << " ms\n";
    std::cout << "  Average vLoop() time per iter: " << std::setprecision(3) << avg_vloop_ms << " ms\n";

    // Print detailed profiling
    std::cout << "\n=== V-Loop Detailed Profiling (avg per iteration) ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  [BOTTOM-UP]\n";
    std::cout << "    super1: iter=" << (t_super1_iter / iters_done) << " ms\n";
    std::cout << "    abs1:   bottomUp=" << (t_abs1_bottomUp / iters_done) << " ms, iter=" << (t_abs1_iter / iters_done) << " ms\n";
    std::cout << "    super2: bottomUp=" << (t_super2_bottomUp / iters_done) << " ms, iter=" << (t_super2_iter / iters_done) << " ms\n";
    std::cout << "    abs2:   bottomUp=" << (t_abs2_bottomUp / iters_done) << " ms, iter=" << (t_abs2_iter / iters_done) << " ms\n";
    std::cout << "  [TOP-DOWN]\n";
    std::cout << "    abs2:   iter=" << (t_abs2_topDown_iter / iters_done) << " ms, topDown=" << (t_abs2_topDown / iters_done) << " ms\n";
    std::cout << "    super2: iter=" << (t_super2_topDown_iter / iters_done) << " ms, topDown=" << (t_super2_topDown / iters_done) << " ms\n";
    std::cout << "    abs1:   iter=" << (t_abs1_topDown_iter / iters_done) << " ms, topDown=" << (t_abs1_topDown / iters_done) << " ms\n";
    std::cout << "    super1: iter=" << (t_super1_topDown_iter / iters_done) << " ms, topDown=" << (t_super1_topDown / iters_done) << " ms\n";
    
    double total_bottomUp = t_super1_iter + t_abs1_bottomUp + t_abs1_iter + t_super2_bottomUp + t_super2_iter + t_abs2_bottomUp + t_abs2_iter;
    double total_topDown = t_abs2_topDown_iter + t_abs2_topDown + t_super2_topDown_iter + t_super2_topDown + t_abs1_topDown_iter + t_abs1_topDown + t_super1_topDown_iter + t_super1_topDown;
    std::cout << "  [SUMMARY]\n";
    std::cout << "    Total bottom-up avg: " << (total_bottomUp / iters_done) << " ms\n";
    std::cout << "    Total top-down avg:  " << (total_topDown / iters_done) << " ms\n";

    // Print detailed bottomUp profile
    printBottomUpProfile();

    // Print detailed computeFactor profile
    printComputeFactorProfile();

    std::cout << "\n=== Test Complete ===\n";
    return 0;
}

