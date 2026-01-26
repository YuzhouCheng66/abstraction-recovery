#include "gbp/VariableNode.h"
#include "gbp/Factor.h"
#include <cassert>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <iostream>


// ============================================================
// Sampling profiler helpers (updateBelief)
// ============================================================
#ifndef GBP_PROFILE_SAMPLE_LOG2
#define GBP_PROFILE_SAMPLE_LOG2 10   // 2^10 = 1024 samples
#endif

namespace {
static constexpr uint32_t kUBMask  = (1u << GBP_PROFILE_SAMPLE_LOG2) - 1u;
static constexpr uint32_t kUBScale = (1u << GBP_PROFILE_SAMPLE_LOG2);

inline bool ubSampled_() {
    static thread_local uint32_t c = 0;
    return ((++c) & kUBMask) == 0;
}
} // anonymous namespace


// ===== Profiling counters for VariableNode::updateBelief =====
static std::atomic<long long> g_ub_total_ns{0};
static std::atomic<long long> g_ub_acc_ns{0};
static std::atomic<long long> g_ub_write_ns{0};
static std::atomic<long long> g_ub_llt_ns{0};
static std::atomic<long long> g_ub_solve_mu_ns{0};
static std::atomic<long long> g_ub_solve_sigma_ns{0};
static std::atomic<long long> g_ub_sigma_skip_ns{0};
static std::atomic<long long> g_ub_total2_ns{0};
static std::atomic<long long> g_ub_acc2_ns{0};
static std::atomic<long long> g_ub_llt2_ns{0};
static std::atomic<long long> g_ub_solve2_mu_ns{0};
static std::atomic<long long> g_ub_solve2_sigma_ns{0};
static std::atomic<int> g_ub_calls{0};
static std::atomic<int> g_ub_calls2{0};

namespace gbp {

void printUpdateBeliefProfile() {
    const int calls = g_ub_calls.load();
    const int calls2 = g_ub_calls2.load();
    if (calls == 0 && calls2 == 0) {
        std::cout << "[updateBelief Profile] No calls recorded.\n";
        return;
    }
    auto toMs = [](long long ns) { return ns / 1e6; };

    if (calls2 > 0) {
        const double total_ms = toMs(g_ub_total2_ns.load());
        std::cout << "[updateBelief Profile][dofs==2][sampling 2^" << GBP_PROFILE_SAMPLE_LOG2 << "] calls=" << calls2
                  << " total=" << total_ms << " ms (avg " << (total_ms / calls2) << " ms/call)\n";
        const double acc = toMs(g_ub_acc2_ns.load());
        const double llt = toMs(g_ub_llt2_ns.load());
        const double smu = toMs(g_ub_solve2_mu_ns.load());
        const double ssig = toMs(g_ub_solve2_sigma_ns.load());
        std::cout << "  - accumulate (prior+msgs): " << acc  << " ms\n";
        std::cout << "  - LLT (2x2):               " << llt  << " ms\n";
        std::cout << "  - solve mu:                " << smu  << " ms\n";
        std::cout << "  - solve Sigma:             " << ssig << " ms\n";
    }

    if (calls > 0) {
        const double total_ms = toMs(g_ub_total_ns.load());
        std::cout << "[updateBelief Profile][dofs!=2] calls=" << calls
                  << " total=" << total_ms << " ms (avg " << (total_ms / calls) << " ms/call)\n";
        const double acc = toMs(g_ub_acc_ns.load());
        const double wr  = toMs(g_ub_write_ns.load());
        const double llt = toMs(g_ub_llt_ns.load());
        const double smu = toMs(g_ub_solve_mu_ns.load());
        const double ssig = toMs(g_ub_solve_sigma_ns.load());
        const double skip = toMs(g_ub_sigma_skip_ns.load());
        std::cout << "  - accumulate (prior+msgs): " << acc  << " ms\n";
        std::cout << "  - write belief (eta/lam):  " << wr   << " ms\n";
        std::cout << "  - LLT:                     " << llt  << " ms\n";
        std::cout << "  - solve mu:                " << smu  << " ms\n";
        std::cout << "  - solve Sigma:             " << ssig << " ms\n";
        if (skip > 0.0) {
            std::cout << "  - Sigma skipped (flag off): " << skip << " ms\n";
        }
    }
}

void resetUpdateBeliefProfile() {
    g_ub_total_ns = 0;
    g_ub_acc_ns = 0;
    g_ub_write_ns = 0;
    g_ub_llt_ns = 0;
    g_ub_solve_mu_ns = 0;
    g_ub_solve_sigma_ns = 0;
    g_ub_sigma_skip_ns = 0;
    g_ub_total2_ns = 0;
    g_ub_acc2_ns = 0;
    g_ub_llt2_ns = 0;
    g_ub_solve2_mu_ns = 0;
    g_ub_solve2_sigma_ns = 0;
    g_ub_calls = 0;
    g_ub_calls2 = 0;
}

VariableNode::VariableNode(int id_, int dofs_)
    : id(id_),
      variableID(id_),
      dofs(dofs_),
      dim(dofs_),
      active(true),
      prior(dofs_),
      belief(dofs_),
      mu(Eigen::VectorXd::Zero(dofs_)),
      Sigma(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      GT(Eigen::VectorXd::Zero(dofs_)),
      eta_acc_(Eigen::VectorXd::Zero(dofs_)),
      lam_acc_(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      lam_work_(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      I_(Eigen::MatrixXd::Identity(dofs_, dofs_)),
      llt_(),
      compute_sigma_(true)
{
    // Nothing else
}

VariableNode::VariableNode()
    : id(-1),
      variableID(-1),
      dofs(0),
      dim(0),
      active(true),
      prior(0),
      belief(0),
      mu(),
      Sigma(),
      GT(),
      eta_acc_(),
      lam_acc_(),
      lam_work_(),
      I_(),
      llt_(),
      compute_sigma_(true)
{
    // Nothing else
}

void VariableNode::updateBelief() {
    const bool sampled = ubSampled_();
    using Clock = std::chrono::high_resolution_clock;
    auto add_ns = [](std::atomic<long long>& acc, const Clock::time_point& a, const Clock::time_point& b) {
        acc.fetch_add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(),
                      std::memory_order_relaxed);
    };

    const auto t_all0 = Clock::now();

    if (!active) return;
    if (dofs <= 0) return;

    // Ensure caches are allocated once and match dofs
    ensureCache_();

    // ------------------------------------------------------------
    // Fast path: 2D variables (fixed-size Eigen kernels)
    //   - Avoid dynamic-size Eigen overhead for the common dofs==2 case
    //   - Still uses LLT.solve (no explicit inverse)
    // ------------------------------------------------------------
    if (dofs == 2) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;

        g_ub_calls2.fetch_add(1, std::memory_order_relaxed);

        const auto t_acc0 = Clock::now();
        Vec2 eta2 = prior.eta().head<2>();
        Mat2 lam2 = prior.lam().topLeftCorner<2, 2>();

        for (const auto& aref : adj_factors) {
            const Factor* f = aref.factor;
            const int k = aref.local_idx;

            assert(f != nullptr);
            assert(k >= 0 && k < (int)f->messages.size());

            eta2.noalias() += f->messages[k].eta().head<2>();
            lam2.noalias() += f->messages[k].lam().topLeftCorner<2, 2>();
        }

        const auto t_acc1 = Clock::now();
        add_ns(g_ub_acc2_ns, t_acc0, t_acc1);

        // Write belief (information form) without extra copies
        // (counted inside accumulate for dofs==2, since it's typically tiny vs LLT)
        belief.etaRef().head<2>() = eta2;
        belief.lamRef().topLeftCorner<2, 2>() = lam2;

        // Solve for mu (and optionally Sigma) using fixed-size LLT
        const auto t_llt0 = Clock::now();
        Eigen::LLT<Mat2> llt2;
        llt2.compute(lam2);
        if (llt2.info() != Eigen::Success) {
            lam2(0, 0) += kJitter;
            lam2(1, 1) += kJitter;
            llt2.compute(lam2);
            if (llt2.info() != Eigen::Success) {
                throw std::runtime_error("VariableNode::updateBelief (2D): LLT failed (matrix not SPD?)");
            }
        }
        const auto t_llt1 = Clock::now();
        add_ns(g_ub_llt2_ns, t_llt0, t_llt1);

        const auto t_mu0 = Clock::now();
        mu.head<2>().noalias() = llt2.solve(eta2);
        const auto t_mu1 = Clock::now();
        add_ns(g_ub_solve2_mu_ns, t_mu0, t_mu1);

        if (compute_sigma_) {
            const Mat2 I2 = Mat2::Identity();
            const auto t_s0 = Clock::now();
            Sigma.topLeftCorner<2, 2>().noalias() = llt2.solve(I2);
            const auto t_s1 = Clock::now();
            add_ns(g_ub_solve2_sigma_ns, t_s0, t_s1);
        }
        const auto t_all1 = Clock::now();
        add_ns(g_ub_total2_ns, t_all0, t_all1);
        return;
    }

    // -------------------------
    // Accumulate belief in scratch (information form):
    //   eta_acc = prior.eta + sum_k msg_k.eta
    //   lam_acc = prior.lam + sum_k msg_k.lam
    // -------------------------
    g_ub_calls.fetch_add(1, std::memory_order_relaxed);

    const auto t_acc0 = Clock::now();
    eta_acc_.noalias() = prior.eta();
    lam_acc_.noalias() = prior.lam();

    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;

        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());

        // Accumulate incoming messages
        eta_acc_.noalias() += f->messages[k].eta();
        lam_acc_.noalias() += f->messages[k].lam();
    }

    const auto t_acc1 = Clock::now();
    add_ns(g_ub_acc_ns, t_acc0, t_acc1);

    // ---- VA: write belief without setEta/setLam overhead (no extra copies) ----
    const auto t_wr0 = Clock::now();
    belief.etaRef().noalias() = eta_acc_;
    belief.lamRef().noalias() = lam_acc_;
    const auto t_wr1 = Clock::now();
    add_ns(g_ub_write_ns, t_wr0, t_wr1);

    // -------------------------
    // Compute mu, Sigma from belief (SPD expected):
    //   (lam + jitter*I) * mu    = eta
    //   (lam + jitter*I) * Sigma = I
    // -------------------------
    // ---- VB: avoid lam_work_ copy unless LLT fails ----
    // Try factorization without jitter first (common case: SPD already)
    const auto t_llt0 = Clock::now();
    llt_.compute(lam_acc_);
    if (llt_.info() != Eigen::Success) {
        lam_work_.noalias() = lam_acc_;
        lam_work_.diagonal().array() += kJitter;
        llt_.compute(lam_work_);
        if (llt_.info() != Eigen::Success) {
            throw std::runtime_error("VariableNode::updateBelief: LLT failed (matrix not SPD?)");
        }
    }
    const auto t_llt1 = Clock::now();
    add_ns(g_ub_llt_ns, t_llt0, t_llt1);


    // mu = inv(lam)*eta
    const auto t_mu0 = Clock::now();
    mu.noalias() = llt_.solve(eta_acc_);
    const auto t_mu1 = Clock::now();
    add_ns(g_ub_solve_mu_ns, t_mu0, t_mu1);

    // Sigma = inv(lam)*I (optional)
    if (compute_sigma_) {
        const auto t_s0 = Clock::now();
        Sigma.noalias() = llt_.solve(I_);
        const auto t_s1 = Clock::now();
        add_ns(g_ub_solve_sigma_ns, t_s0, t_s1);
    } else {
        // In case you toggle compute_sigma_ off for performance experiments.
        const auto t_s0 = Clock::now();
        const auto t_s1 = Clock::now();
        add_ns(g_ub_sigma_skip_ns, t_s0, t_s1);
    }

    const auto t_all1 = Clock::now();
    add_ns(g_ub_total_ns, t_all0, t_all1);

}

} // namespace gbp