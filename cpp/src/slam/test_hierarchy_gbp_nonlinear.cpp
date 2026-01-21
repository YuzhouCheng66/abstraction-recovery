// test_hierarchy_gbp_nonlinear.cpp
//
// SE(2) nonlinear tests with identical seed + identical graph:
//   Step 1: Batch Gauss-Newton (relinearize -> sparse solve -> update mu)
//   Step 2: Synchronous GBP with outer relinearization schedule (m,k)
//   Step 3: HierarchyGBP V-loop (one V-loop per relinearization) [strict GN]
//
// Usage:
//   ./test_hierarchy_gbp_nonlinear.exe --step 1
//   ./test_hierarchy_gbp_nonlinear.exe --step 2
//   ./test_hierarchy_gbp_nonlinear.exe --step 3
// Optional:
//   --m <int>   outer GN steps (default 20)
//   --k <int>   inner iterations per GN step (Step2 only; default 50)
//   --group <int> HierarchyGBP group size (Step3 only; default 4)
//   --r <int>     r_reduced (Step3; default 2)
//   --damp <double> eta_damping (Step3; default 0.4)
//   --jitter <double> diag jitter for sparse solve (Step1; default 1e-9)

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

#include "slam/SlamGraph.h"
#include "slam/SlamFactorGraph.h"
#include "gbp/FactorGraph.h"
#include "gbp/HierarchyGBP.h"

static inline double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

// Objective value: sum_f 0.5 * (h(mu)-z)^T * Lambda * (h(mu)-z)
// For SE2 convention: component 2 is angle, wrap it before quadratic.
static double objectiveSE2(const gbp::FactorGraph& fg) {
    double total = 0.0;
    for (const auto& fup : fg.factors) {
        const gbp::Factor* f = fup.get();
        if (!f || !f->active) continue;

        // build concatenated mu for this factor
        int D = 0;
        for (auto* vn : f->adj_var_nodes) D += vn->dofs;
        Eigen::VectorXd x(D);
        int off = 0;
        for (auto* vn : f->adj_var_nodes) {
            x.segment(off, vn->dofs) = vn->mu;
            off += vn->dofs;
        }

        const std::vector<Eigen::VectorXd> h = f->meas_fn(x);
        const int K = (int)h.size();
        for (int k = 0; k < K; ++k) {
            Eigen::VectorXd e = h[k] - f->measurement[k];
            if (e.size() >= 3) e(2) = wrapAngle(e(2));
            total += 0.5 * (e.transpose() * f->measurement_lambda[k] * e)(0, 0);
        }
    }
    return total;
}

static void setMuFromStackedVector(gbp::FactorGraph& fg, const Eigen::VectorXd& mu_stack) {
    // The stacked vector uses the same variable ordering as jointDistributionInfSparse():
    // increasing variableID and concatenating their dofs.
    int offset = 0;
    for (auto& vptr : fg.var_nodes) {
        if (!vptr) continue;
        auto& v = *vptr;
        const int d = v.dofs;
        if (offset + d > mu_stack.size()) {
            throw std::runtime_error("setMuFromStackedVector: mu_stack too small");
        }
        v.mu = mu_stack.segment(offset, d);
        offset += d;
    }
}

static gbp::FactorGraph buildBaseSE2(unsigned int seed, bool use_seed) {
    // 1) Build an SE(2) SLAM-like graph
    const int N = 5000;
    const double step_size = 25.0;
    const double loop_prob = 0.05;
    const double loop_radius = 50.0;
    const double prior_prop = 0.02; // anchor only

    slam::SlamGraphSE2 g = slam::makeSlamLikeGraphSE2(
        N, step_size, loop_prob, loop_radius, prior_prop,
        seed, use_seed
    );

    // 2) Build nonlinear factor graph (SE2 between)
    slam::NoiseConfigSE2 cfg;
    cfg.prior_sigma = 1.0;
    cfg.odom_sigma  = 1.0;
    cfg.loop_sigma  = 10.0;
    cfg.theta_ratio = 0.01;
    cfg.tiny_prior  = 1e-10;
    cfg.seed = seed;
    cfg.use_seed = use_seed;

    return slam::buildNoisyPoseGraphSE2(g.nodes, g.edges, cfg);
}

// ---------------- Step 1: Batch GN ----------------
static int runStep1BatchGN(unsigned int seed, bool use_seed, int m, double jitter) {
    gbp::FactorGraph fg = buildBaseSE2(seed, use_seed);

    std::cout << "[Step1] SE2 base graph built: vars=" << fg.var_nodes.size()
              << ", factors=" << fg.factors.size()
              << ", nonlinear_factors=" << (fg.nonlinear_factors ? "true" : "false")
              << "\n";

    // Initial linearization
    if (fg.nonlinear_factors) fg.relinearizeAllFactors();

    for (int outer = 0; outer < m; ++outer) {
        // (1) relinearize at current mu
        if (fg.nonlinear_factors) fg.relinearizeAllFactors();

        // (2) solve the quadratic approximation in one sparse solve
        Eigen::VectorXd mu_stack = fg.jointMAPSparse(jitter);
        setMuFromStackedVector(fg, mu_stack);

        // (3) report objective at updated mu (true nonlinear objective)
        const double E = objectiveSE2(fg);
        std::cout << "outer " << std::setw(3) << outer
                  << "   obj=" << std::setprecision(12) << E
                  << "\n";
    }

    return 0;
}

// ---------------- Step 2: Synchronous GBP + outer relinearization ----------------
static int runStep2SyncGBP(unsigned int seed, bool use_seed, int m, int k) {
    gbp::FactorGraph fg = buildBaseSE2(seed, use_seed);

    std::cout << "[Step2] SE2 base graph built: vars=" << fg.var_nodes.size()
              << ", factors=" << fg.factors.size()
              << ", nonlinear_factors=" << (fg.nonlinear_factors ? "true" : "false")
              << "\n";

    if (fg.nonlinear_factors) fg.relinearizeAllFactors();

    for (int outer = 0; outer < m; ++outer) {
        for (int inner = 0; inner < k; ++inner) {
            fg.synchronousIteration(false);
        }
        if (fg.nonlinear_factors) fg.relinearizeAllFactors();

        const int effective_iter = (outer + 1) * k;
        const double E = objectiveSE2(fg);
        std::cout << "outer " << std::setw(3) << outer
                  << " (iters=" << std::setw(4) << effective_iter << ")"
                  << "   obj=" << std::setprecision(12) << E
                  << "\n";
    }

    return 0;
}

// ---------------- Step 3: HierarchyGBP vLoop, strict GN (one vLoop per relinearization) ----------------
static int runStep3VLoopGN(unsigned int seed, bool use_seed, int m, int group_size, int r_reduced, double eta_damping) {
    // Use shared_ptr graphs so HierarchyGBP can mutate layers.
    auto base = std::make_shared<gbp::FactorGraph>(buildBaseSE2(seed, use_seed));

    std::cout << "[Step3] SE2 base graph built: vars=" << base->var_nodes.size()
              << ", factors=" << base->factors.size()
              << ", nonlinear_factors=" << (base->nonlinear_factors ? "true" : "false")
              << "\n";

    // ---- (A) Initial linearization on base
    if (base->nonlinear_factors) base->relinearizeAllFactors();
    base->eta_damping = 0.0;
    base->synchronousIteration(false);
    gbp::HierarchyGBP H(group_size);

    // Build 5-layer V-cycle scaffold: base -> super1 -> abs1 -> super2 -> abs2
    auto super1 = H.buildSuperFromBase(base, /*eta_damping=*/0.0);

    // ---- (B) build super1 from base (outside vLoop) ----
    std::cout << "  [Init] build super1 from base...\n";
    H.bottomUpUpdateSuper(base, super1);

    // ---- (C) super1: ensure it has run at least once ----
    std::cout << "  [Init] super1: one synchronousIteration() before building abs1...\n";
    super1->graph->eta_damping = 0.0;
    super1->graph->synchronousIteration(false);

    // ---- (D) build abs1 from super1 (outside vLoop) ----
    std::cout << "  [Init] build abs1 from super1...\n";
    auto abs1 = H.buildAbsFromSuper(super1->graph, /*r_reduced=*/r_reduced, /*eta_damping=*/0.0);
    H.bottomUpUpdateAbs(super1->graph, abs1, /*r_reduced=*/r_reduced, /*eta_damping=*/0.0);

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
    auto abs2 = H.buildAbsFromSuper(super2->graph, /*r_reduced=*/r_reduced, /*eta_damping=*/0.0);
    H.bottomUpUpdateAbs(super2->graph, abs2, /*r_reduced=*/r_reduced, /*eta_damping=*/0.0);

    // ---- pack layers for vLoop ----
    std::vector<gbp::HierarchyGBP::VLayerEntry> vLayers;
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"base",   base,           nullptr, nullptr});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"super1", super1->graph,  super1,  nullptr});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"abs1",   abs1->graph,    nullptr, abs1});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"super2", super2->graph,  super2,  nullptr});
    vLayers.push_back(gbp::HierarchyGBP::VLayerEntry{"abs2",   abs2->graph,    nullptr, abs2});

    const double E = objectiveSE2(*base);
    std::cout << "outer " << std::setw(3) 
                << "   obj=" << std::setprecision(12) << E
                << "\n";
    for (int outer = 0; outer < m; ++outer) {
        // Strict GN policy: relinearize ONCE per outer step, then run exactly one V-loop.
        //if (base->nonlinear_factors) base->relinearizeAllFactors();

        //H.vLoop(vLayers, r_reduced, eta_damping);

        base->synchronousIteration(false);
        const double E = objectiveSE2(*base);
        std::cout << "outer " << std::setw(3) << outer
                  << "   obj=" << std::setprecision(12) << E
                  << "\n";
    }

    return 0;
}

static bool parseInt(const char* s, int& out) {
    try {
        out = std::stoi(std::string(s));
        return true;
    } catch (...) {
        return false;
    }
}

static bool parseDouble(const char* s, double& out) {
    try {
        out = std::stod(std::string(s));
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char** argv) {
    int step = 2;
    int m = 100;
    int k = 200;
    int group = 5;
    int r_reduced = 3;
    double damp = 0.4;
    double jitter = 1e-12;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--step" && i + 1 < argc) { parseInt(argv[++i], step); }
        else if (a == "--m" && i + 1 < argc) { parseInt(argv[++i], m); }
        else if (a == "--k" && i + 1 < argc) { parseInt(argv[++i], k); }
        else if (a == "--group" && i + 1 < argc) { parseInt(argv[++i], group); }
        else if (a == "--r" && i + 1 < argc) { parseInt(argv[++i], r_reduced); }
        else if (a == "--damp" && i + 1 < argc) { parseDouble(argv[++i], damp); }
        else if (a == "--jitter" && i + 1 < argc) { parseDouble(argv[++i], jitter); }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: --step {1|2|3} [--m N] [--k K] [--group G] [--r R] [--damp D] [--jitter J]\n";
            return 0;
        }
    }

    const unsigned int seed = 42;
    const bool use_seed = true;

    if (step == 1) return runStep1BatchGN(seed, use_seed, m, jitter);
    if (step == 2) return runStep2SyncGBP(seed, use_seed, m, k);
    if (step == 3) return runStep3VLoopGN(seed, use_seed, m, group, r_reduced, damp);

    std::cerr << "Unknown --step=" << step << " (expected 1/2/3)\n";
    return 1;
}
