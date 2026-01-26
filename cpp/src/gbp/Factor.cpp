#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

#include <cassert>
#include <stdexcept>
#include <utility>   // std::move
#include <chrono>
#include <atomic>
#include <iostream>

// ===== Profiling counters for computeFactor =====
static std::atomic<long long> g_cf_jac_ns{0};
static std::atomic<long long> g_cf_meas_ns{0};
static std::atomic<long long> g_cf_loop_lam_ns{0};
static std::atomic<long long> g_cf_loop_eta_ns{0};
static std::atomic<int> g_cf_call_count{0};

// ===== Profiling counters for computeMessagesFixedLam =====
static std::atomic<long long> g_cmf_total_ns{0};
static std::atomic<long long> g_cmf_init_fallback_ns{0};
static std::atomic<long long> g_cmf_generic_fallback_ns{0};
static std::atomic<long long> g_cmf_unary_ns{0};
static std::atomic<long long> g_cmf_solve0_ns{0};
static std::atomic<long long> g_cmf_solve1_ns{0};
static std::atomic<long long> g_cmf_pack0_ns{0};
static std::atomic<long long> g_cmf_pack1_ns{0};
static std::atomic<long long> g_cmf_swap_ns{0};
static std::atomic<long long> g_cmf_misc_ns{0};
static std::atomic<int> g_cmf_calls{0};
static std::atomic<int> g_cmf_init_fallback_calls{0};
static std::atomic<int> g_cmf_generic_fallback_calls{0};
static std::atomic<int> g_cmf_unary_calls{0};

void printComputeFactorProfile() {
    // int calls = g_cf_call_count.load();
    // if (calls == 0) {
    //     std::cout << "[computeFactor Profile] No calls recorded.\n";
    //     return;
    // }
    // auto toMs = [](long long ns) { return ns / 1e6; };
    // std::cout << "\n=== computeFactor Detailed Profile (" << calls << " calls) ===\n";
    // std::cout << "  jac_fn:        " << toMs(g_cf_jac_ns.load()) << " ms\n";
    // std::cout << "  meas_fn:       " << toMs(g_cf_meas_ns.load()) << " ms\n";
    // std::cout << "  loop (lambda): " << toMs(g_cf_loop_lam_ns.load()) << " ms\n";
    // std::cout << "  loop (eta):    " << toMs(g_cf_loop_eta_ns.load()) << " ms\n";
    // std::cout << "==============================================\n";
}

void resetComputeFactorProfile() {
    g_cf_jac_ns = 0;
    g_cf_meas_ns = 0;
    g_cf_loop_lam_ns = 0;
    g_cf_loop_eta_ns = 0;
    g_cf_call_count = 0;
}

void printComputeMessagesFixedLamProfile() {
    const int calls = g_cmf_calls.load();
    if (calls == 0) {
        std::cout << "[computeMessagesFixedLam Profile] No calls recorded.\n";
        return;
    }
    auto toMs = [](long long ns) { return ns / 1e6; };

    const double total_ms = toMs(g_cmf_total_ns.load());
    std::cout << "[computeMessagesFixedLam Profile] calls=" << calls
              << " total=" << total_ms << " ms (avg " << (total_ms / calls) << " ms/call)\n";

    const int init_calls = g_cmf_init_fallback_calls.load();
    const int gen_calls  = g_cmf_generic_fallback_calls.load();
    const int unary_calls = g_cmf_unary_calls.load();

    if (init_calls > 0) {
        const double ms = toMs(g_cmf_init_fallback_ns.load());
        std::cout << "  - init fallback (computeMessages once): " << ms
                  << " ms (avg " << (ms / init_calls) << " ms/call)\n";
    }
    if (gen_calls > 0) {
        const double ms = toMs(g_cmf_generic_fallback_ns.load());
        std::cout << "  - generic fallback (non-2D / other):    " << ms
                  << " ms (avg " << (ms / gen_calls) << " ms/call)\n";
    }
    if (unary_calls > 0) {
        const double ms = toMs(g_cmf_unary_ns.load());
        std::cout << "  - unary:                                " << ms
                  << " ms (avg " << (ms / unary_calls) << " ms/call)\n";
    }

    const double s0 = toMs(g_cmf_solve0_ns.load());
    const double s1 = toMs(g_cmf_solve1_ns.load());
    const double p0 = toMs(g_cmf_pack0_ns.load());
    const double p1 = toMs(g_cmf_pack1_ns.load());
    const double sw = toMs(g_cmf_swap_ns.load());
    const double mi = toMs(g_cmf_misc_ns.load());

    std::cout << "  - solve0:                               " << s0 << " ms\n";
    std::cout << "  - pack0 (eta+damp+write):               " << p0 << " ms\n";
    std::cout << "  - solve1:                               " << s1 << " ms\n";
    std::cout << "  - pack1 (eta+damp+write):               " << p1 << " ms\n";
    std::cout << "  - swap:                                 " << sw << " ms\n";
    std::cout << "  - misc (branching/guards/maps):          " << mi << " ms\n";
}

void resetComputeMessagesFixedLamProfile() {
    g_cmf_total_ns = 0;
    g_cmf_init_fallback_ns = 0;
    g_cmf_generic_fallback_ns = 0;
    g_cmf_unary_ns = 0;
    g_cmf_solve0_ns = 0;
    g_cmf_solve1_ns = 0;
    g_cmf_pack0_ns = 0;
    g_cmf_pack1_ns = 0;
    g_cmf_swap_ns = 0;
    g_cmf_misc_ns = 0;
    g_cmf_calls = 0;
    g_cmf_init_fallback_calls = 0;
    g_cmf_generic_fallback_calls = 0;
    g_cmf_unary_calls = 0;
}

namespace gbp {

// ==============================
// ctor + workspace init
// ==============================

Factor::Factor(
    int id_,
    const std::vector<VariableNode*>& vars,
    const std::vector<Eigen::VectorXd>& z,
    const std::vector<Eigen::MatrixXd>& lambda,
    std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas,
    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)>  jac
)
    : factorID(id_),
      active(true),
      adj_var_nodes(vars),
      measurement(z),
      measurement_lambda(lambda),
      meas_fn(std::move(meas)),
      jac_fn(std::move(jac)),
      factor(0) // resized below
{
    assert(adj_var_nodes.size() == 1 || adj_var_nodes.size() == 2);

    // Cache IDs and compute dofs
    adj_vIDs.reserve(adj_var_nodes.size());

    int total_dofs = 0;
    if (adj_var_nodes.size() == 1) {
        is_unary_  = true;
        is_binary_ = false;

        auto* v0 = adj_var_nodes[0];
        assert(v0 != nullptr);

        adj_vIDs.push_back(v0->variableID);

        d0_ = v0->dofs;
        d1_ = 0;
        D_  = d0_;
        total_dofs = D_;
    } else {
        is_unary_  = false;
        is_binary_ = true;

        auto* v0 = adj_var_nodes[0];
        auto* v1 = adj_var_nodes[1];
        assert(v0 != nullptr && v1 != nullptr);

        adj_vIDs.push_back(v0->variableID);
        adj_vIDs.push_back(v1->variableID);

        d0_ = v0->dofs;
        d1_ = v1->dofs;
        D_  = d0_ + d1_;
        total_dofs = D_;
    }

    // Allocate factor gaussian + linpoint
    factor   = utils::NdimGaussian(total_dofs);
    linpoint = Eigen::VectorXd::Zero(total_dofs);

    // Allocate messages (fixed per-variable dofs) [C: ping-pong buffers]
    messages.reserve(adj_var_nodes.size());
    messages_next.reserve(adj_var_nodes.size());
    for (auto* v : adj_var_nodes) {
        assert(v != nullptr);
        messages.emplace_back(v->dofs);
        messages_next.emplace_back(v->dofs);
    }

    // Sanity: Python assumes same length
    if (measurement.size() != measurement_lambda.size()) {
        throw std::runtime_error("Factor ctor: measurement and measurement_lambda size mismatch");
    }

    // Pre-allocate workspace (only meaningful for binary; unary is trivial)
    initWorkspace_();
}

void Factor::initWorkspace_() {
    // Unary factor: computeMessages is just copy factor into messages[0]
    if (is_unary_) return;

    // Binary: allocate all buffers at maximum needed sizes
    // max dofs among the two vars
    const int max_d = (d0_ > d1_) ? d0_ : d1_;

    // factor scratch
    eta_f_.resize(D_);
    lam_f_.resize(D_, D_);

    // Schur scratch
    lnono_.resize(max_d, max_d);
    Y_.resize(max_d, max_d);     // use topLeftCorner(d_no, d_o)
    y_.resize(max_d);            // use head(d_no)

    // tmp for msg computations
    tmpLam_.resize(max_d, max_d); // use topLeftCorner(d_o, d_o)
    tmpEta_.resize(max_d);        // use head(d_o)

    // NOTE (C optimization): message outputs write directly into messages_next,
    // so no separate new_eta_/new_lam_ staging buffers are needed.
}

// ==============================
// factor computation
// ==============================
void Factor::invalidateJacobianCache() {
    jcache_valid_ = false;
    lamcache_set_ = false;
    J_cache_.clear();
    JO_cache_.clear();
    lambda_cache_.resize(0, 0);
}

void Factor::computeFactor(const Eigen::VectorXd& linpoint_in, bool update_self) {
    // ++g_cf_call_count;  // disabled for performance

    // Avoid redundant self-copy when caller reuses Factor::linpoint as the input buffer.
    if (&linpoint_in != &linpoint) {
        linpoint = linpoint_in;
    }

    // meas_fn may depend on current linpoint (e.g. via k), so we evaluate it every call.
    // auto t0 = std::chrono::high_resolution_clock::now();
    auto pred = meas_fn(linpoint);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // g_cf_meas_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    const int D = (int)linpoint.size();

    // Build Jacobian-derived caches lazily (assumes J and measurement_lambda do not change
    // across iterations for the current graph / abstraction).
    if (!jcache_valid_) {
        // auto tJ0 = std::chrono::high_resolution_clock::now();
        auto J = jac_fn(linpoint);
        // auto tJ1 = std::chrono::high_resolution_clock::now();
        // g_cf_jac_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tJ1 - tJ0).count();

        if (J.size() != measurement.size() ||
            J.size() != measurement_lambda.size() ||
            J.size() != pred.size()) {
            throw std::runtime_error("computeFactor: block list size mismatch among J, measurement, measurement_lambda, pred");
        }

        // Cache J blocks (move) and precompute JO = J^T * O, and Lambda = sum JO * J.
        J_cache_ = std::move(J);
        JO_cache_.resize(J_cache_.size());

        lambda_cache_.resize(D, D);
        lambda_cache_.setZero();

        for (size_t i = 0; i < J_cache_.size(); ++i) {
            const Eigen::MatrixXd& Ji = J_cache_[i];
            const Eigen::MatrixXd& Oi = measurement_lambda[i];

            // JO_i = Ji^T * Oi   (D x m)
            JO_cache_[i].resize(Ji.cols(), Oi.cols());
            JO_cache_[i].noalias() = Ji.transpose() * Oi;

            // Lambda += JO_i * Ji   (D x m) * (m x D) -> (D x D)
            lambda_cache_.noalias() += JO_cache_[i] * Ji;
        }

        jcache_valid_ = true;
        lamcache_set_ = false; // ensure we push cached lambda to factor once
    } else {
        // Validate sizes in cached mode (cheap safety).
        if (J_cache_.size() != measurement.size() ||
            J_cache_.size() != measurement_lambda.size() ||
            J_cache_.size() != pred.size()) {
            throw std::runtime_error("computeFactor(cached): block list size mismatch among cached J, measurement, measurement_lambda, pred");
        }
        // Dimension may change only if graph structure changed; force rebuild in that case.
        if (lambda_cache_.rows() != D || lambda_cache_.cols() != D) {
            // Conservative: rebuild cache.
            jcache_valid_ = false;
            computeFactor(linpoint_in, update_self);
            return;
        }
    }

    // Compute eta only (Lambda is cached and constant).
    eta_f_.resize(D);
    eta_f_.setZero();

    // auto t2 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < J_cache_.size(); ++i) {
        const Eigen::MatrixXd& Ji = J_cache_[i];
        const Eigen::VectorXd& zi = measurement[i];
        const Eigen::VectorXd& hi = pred[i];

        const int m = (int)zi.size();

        // ri = Ji * linpoint + zi - hi
        // Use head() view if ri_cf_ is large enough, else resize once
        if (ri_cf_.size() < m) ri_cf_.resize(m);
        auto ri = ri_cf_.head(m);
        ri.noalias() = Ji * linpoint;
        ri += zi;
        ri -= hi;

        // eta += (Ji^T * Oi) * ri  == JO_i * ri
        eta_f_.noalias() += JO_cache_[i] * ri;
    }

    if (update_self) {
        if (!lamcache_set_) {
            factor.setLam(lambda_cache_);
            lamcache_set_ = true;
        }
        factor.setEta(eta_f_);
    }
}

// ==============================
// computeMessages (no resize path)
// ==============================



void Factor::computeMessages(double eta_damping) {

    if (!active) return;
    

    // Unary: trivial
    if (is_unary_) {
        auto& outMsg = messages_next[0];
        outMsg.etaRef().noalias() = factor.eta();
        outMsg.lamRef().noalias() = factor.lam();
        messages.swap(messages_next);
        return;
    }

    // Binary: dimensions are fixed (d0_, d1_, D_)
    // old messages (copies) - unavoidable if messages[k] returns by value internally
    const Eigen::VectorXd& old_eta0 = messages[0].eta();
    const Eigen::MatrixXd& old_lam0 = messages[0].lam();
    const Eigen::VectorXd& old_eta1 = messages[1].eta();
    const Eigen::MatrixXd& old_lam1 = messages[1].lam();

    const double a = eta_damping;

    // ------------------------------------------------------------
    // Fast path: 2D-2D binary factor (fixed-size Eigen kernels)
    // ------------------------------------------------------------
    // This keeps the same math as the dynamic path but uses fixed-size
    // matrices/vectors to reduce Eigen's dynamic-size overhead.
    // No explicit inverses are used; we still rely on LLT solves.
    if (d0_ == 2 && d1_ == 2 && D_ == 4) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;
        using Mat4 = Eigen::Matrix<double, 4, 4>;

        // Map factor blocks (assumes column-major Eigen default, contiguous storage).
        const auto& eta_dyn = factor.eta();
        const auto& lam_dyn = factor.lam();
        assert(eta_dyn.size() == 4);
        assert(lam_dyn.rows() == 4 && lam_dyn.cols() == 4);

        const Eigen::Map<const Vec4> eta_f0(eta_dyn.data());
        const Eigen::Map<const Mat4> lam_f0(lam_dyn.data());

        // Old messages (maps)
        const Eigen::Map<const Vec2> old0_eta(old_eta0.data());
        const Eigen::Map<const Vec2> old1_eta(old_eta1.data());
        const Eigen::Map<const Mat2> old0_lam(old_lam0.data());
        const Eigen::Map<const Mat2> old1_lam(old_lam1.data());

        // Beliefs (maps)
        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec2> b0_eta(b0.eta().data());
        const Eigen::Map<const Vec2> b1_eta(b1.eta().data());
        const Eigen::Map<const Mat2> b0_lam(b0.lam().data());
        const Eigen::Map<const Mat2> b1_lam(b1.lam().data());

        // Note: we intentionally fully unroll target=0/1 here to avoid
        // any target-dependent branches in the hot path (including damping).

        const double s = 1.0 - a;  // damping scale

        // =====================
        // target = 0 (to v0, eliminate v1)
        // =====================
        {
            const Vec2 eo   = eta_f0.template segment<2>(0);
            const Vec2 eno  = eta_f0.template segment<2>(2) + (b1_eta - old1_eta);

            const Mat2 loo   = lam_f0.template block<2, 2>(0, 0);
            const Mat2 lono  = lam_f0.template block<2, 2>(0, 2);
            const Mat2 lnoo  = lam_f0.template block<2, 2>(2, 0);
            Mat2 lnono       = lam_f0.template block<2, 2>(2, 2);
            lnono.noalias() += (b1_lam - old1_lam);
            lnono.diagonal().array() += kJitter;

            llt0_.compute(lnono);
            if (llt0_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (2D fast path, target=0)");
            }
            llt_valid0_ = (llt0_.info() == Eigen::Success);
            const Mat2 Y = llt0_.solve(lnoo);
            const Vec2 y = llt0_.solve(eno);

            Mat2 outLam2 = loo - lono * Y;
            Vec2 outEta2 = eo  - lono * y;

            // Write message[0]
            utils::NdimGaussian& outMsg = messages_next[0];
            Eigen::MatrixXd& outLam = outMsg.lamRef();
            Eigen::VectorXd& outEta = outMsg.etaRef();
            assert(outLam.rows() == 2 && outLam.cols() == 2);
            assert(outEta.size() == 2);

            if (a != 0.0) {
                outLam2 *= s;
                outEta2 *= s;
                outLam2.noalias() += a * old0_lam;
                outEta2.noalias() += a * old0_eta;
            }

            outLam = outLam2;
            outEta = outEta2;
        }

        // =====================
        // target = 1 (to v1, eliminate v0)
        // =====================
        {
            const Vec2 eo   = eta_f0.template segment<2>(2);
            const Vec2 eno  = eta_f0.template segment<2>(0) + (b0_eta - old0_eta);

            const Mat2 loo   = lam_f0.template block<2, 2>(2, 2);
            const Mat2 lono  = lam_f0.template block<2, 2>(2, 0);
            const Mat2 lnoo  = lam_f0.template block<2, 2>(0, 2);
            Mat2 lnono       = lam_f0.template block<2, 2>(0, 0);
            lnono.noalias() += (b0_lam - old0_lam);
            lnono.diagonal().array() += kJitter;

            llt1_.compute(lnono);
            llt_valid1_ = (llt1_.info() == Eigen::Success);
            if (llt1_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (2D fast path, target=1)");
            }
            const Mat2 Y = llt1_.solve(lnoo);
            const Vec2 y = llt1_.solve(eno);
            Mat2 outLam2 = loo - lono * Y;
            Vec2 outEta2 = eo  - lono * y;

            // Write message[1]
            utils::NdimGaussian& outMsg = messages_next[1];
            Eigen::MatrixXd& outLam = outMsg.lamRef();
            Eigen::VectorXd& outEta = outMsg.etaRef();
            assert(outLam.rows() == 2 && outLam.cols() == 2);
            assert(outEta.size() == 2);

            if (a != 0.0) {
                outLam2 *= s;
                outEta2 *= s;
                outLam2.noalias() += a * old1_lam;
                outEta2.noalias() += a * old1_eta;
            }

            outLam = outLam2;
            outEta = outEta2;
        }

        // Commit outputs with O(1) swap (NO deep copy)
        messages.swap(messages_next);
        return;
    }

    // We compute two directed messages: target=0 => to v0 (eliminate v1), target=1 => to v1 (eliminate v0)
    for (int target = 0; target < 2; ++target) {
        Eigen::LLT<Eigen::Matrix2d>& llt_ =
            (target == 0) ? llt0_ : llt1_;
        bool& llt_valid_ =
            (target == 0) ? llt_valid0_ : llt_valid1_;
            
        // ------------------------------------------------------------
        // 1) eta_f_, lam_f_ = factor + belief_correction (no resize)
        // ------------------------------------------------------------
        eta_f_.noalias() = factor.eta();
        lam_f_.noalias() = factor.lam();

        if (target == 0) {
            const auto& b1 = adj_var_nodes[1]->belief;
            eta_f_.segment(d0_, d1_).noalias() += (b1.eta() - old_eta1);
            lam_f_.block(d0_, d0_, d1_, d1_).noalias() += (b1.lam() - old_lam1);
        } else {
            const auto& b0 = adj_var_nodes[0]->belief;
            eta_f_.segment(0, d0_).noalias() += (b0.eta() - old_eta0);
            lam_f_.block(0, 0, d0_, d0_).noalias() += (b0.lam() - old_lam0);
        }


        // ------------------------------------------------------------
        // 2) Views for this target
        // ------------------------------------------------------------
        const int d_o  = (target == 0) ? d0_ : d1_;
        const int d_no = (target == 0) ? d1_ : d0_;

        // eta views
        auto eo  = (target == 0) ? eta_f_.segment(0,   d0_) : eta_f_.segment(d0_, d1_);
        auto eno = (target == 0) ? eta_f_.segment(d0_, d1_) : eta_f_.segment(0,   d0_);

        // lam views
        auto loo_view   = (target == 0) ? lam_f_.block(0,   0,   d0_, d0_) : lam_f_.block(d0_, d0_, d1_, d1_);
        auto lono_view  = (target == 0) ? lam_f_.block(0,   d0_, d0_, d1_) : lam_f_.block(d0_, 0,   d1_, d0_);
        auto lnoo_view  = (target == 0) ? lam_f_.block(d0_, 0,   d1_, d0_) : lam_f_.block(0,   d0_, d0_, d1_);
        auto lnono_view = (target == 0) ? lam_f_.block(d0_, d0_, d1_, d1_) : lam_f_.block(0,   0,   d0_, d0_);

        // ------------------------------------------------------------
        // 3) lnono copy + jitter (write into preallocated top-left)
        // ------------------------------------------------------------
        auto lnono = lnono_.topLeftCorner(d_no, d_no);
        lnono.noalias() = lnono_view;
        lnono.diagonal().array() += kJitter;

        // ------------------------------------------------------------
        // 4) LLT + solves into preallocated blocks
        // ------------------------------------------------------------
        llt_.compute(lnono);
        if (llt_.info() != Eigen::Success) {
            throw std::runtime_error("LLT failed in Factor::computeMessages");
        }
        llt_valid_ = (llt_.info() == Eigen::Success);
        auto Y = Y_.topLeftCorner(d_no, d_o);
        Y.noalias() = llt_.solve(lnoo_view);
        auto y = y_.head(d_no);
        y.noalias() = llt_.solve(eno);
        // ------------------------------------------------------------
        // 5) outLam/outEta: write DIRECTLY into messages_next[target]  [C]
        // ------------------------------------------------------------
        utils::NdimGaussian& outMsg = messages_next[target];
        Eigen::MatrixXd& outLam = outMsg.lamRef();
        Eigen::VectorXd& outEta = outMsg.etaRef();

        // outLam = loo - lono*Y   (NO intermediate tmp)
        outLam.noalias() = loo_view;
        outLam.noalias() -= (lono_view * Y);

        // outEta = eo - lono*y    (NO intermediate tmp)
        outEta.noalias() = eo;
        outEta.noalias() -= (lono_view * y);

        // ------------------------------------------------------------
        // 6) damping (in place)
        // ------------------------------------------------------------
        if (a != 0.0) {
            const double s = 1.0 - a;
            outLam *= s;
            outEta *= s;

            if (target == 0) {
                outLam.noalias() += a * old_lam0;
                outEta.noalias() += a * old_eta0;
            } else {
                outLam.noalias() += a * old_lam1;
                outEta.noalias() += a * old_eta1;
            }
        }
    }

    // Commit outputs with O(1) swap (NO deep copy)  [C]
    messages.swap(messages_next);
}



void Factor::computeMessagesFixedLam(double eta_damping) {

    using Clock = std::chrono::high_resolution_clock;
    const auto t_all0 = Clock::now();
    g_cmf_calls.fetch_add(1, std::memory_order_relaxed);

    // We treat guards / branching / mapping cost as "misc".
    auto add_ns = [](std::atomic<long long>& acc, const Clock::time_point& a, const Clock::time_point& b) {
        acc.fetch_add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(),
                      std::memory_order_relaxed);
    };

    // 1) If fixed-lam cache is not ready, do a ONE-TIME full lam update
        //    (same math as computeMessages 2D fast path), and mark cache valid.
    if (!fixed_lam_valid_) {
        const auto t0 = Clock::now();
        computeMessages(eta_damping); // one-time full update
        const auto t1 = Clock::now();
        add_ns(g_cmf_init_fallback_ns, t0, t1);
        g_cmf_init_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        fixed_lam_valid_ = true;
        const auto t_all1 = Clock::now();
        add_ns(g_cmf_total_ns, t_all0, t_all1);
        return;
    }

    if (!active) {
        const auto t_all1 = Clock::now();
        add_ns(g_cmf_total_ns, t_all0, t_all1);
        return;
    }

    // Unary: trivial
    if (is_unary_) {
        const auto t0 = Clock::now();
        auto& outMsg = messages_next[0];
        outMsg.etaRef().noalias() = factor.eta();
        outMsg.lamRef().noalias() = factor.lam();
        const auto t1 = Clock::now();
        add_ns(g_cmf_unary_ns, t0, t1);
        g_cmf_unary_calls.fetch_add(1, std::memory_order_relaxed);

        const auto ts0 = Clock::now();
        messages.swap(messages_next);
        const auto ts1 = Clock::now();
        add_ns(g_cmf_swap_ns, ts0, ts1);

        const auto t_all1 = Clock::now();
        add_ns(g_cmf_total_ns, t_all0, t_all1);
        return;
    }

    // Binary: dimensions are fixed (d0_, d1_, D_)
    // NOTE: "fixed-lam" is implemented only for the hot 2D-2D between-factor.
    //       For other sizes we fall back to the normal computeMessages path.
    const Eigen::VectorXd& old_eta0 = messages[0].eta();
    const Eigen::MatrixXd& old_lam0 = messages[0].lam();
    const Eigen::VectorXd& old_eta1 = messages[1].eta();
    const Eigen::MatrixXd& old_lam1 = messages[1].lam();

    const double a = eta_damping;

    if (!(d0_ == 2 && d1_ == 2 && D_ == 4)) {
        const auto t0 = Clock::now();
        computeMessages(eta_damping);
        const auto t1 = Clock::now();
        add_ns(g_cmf_generic_fallback_ns, t0, t1);
        g_cmf_generic_fallback_calls.fetch_add(1, std::memory_order_relaxed);
        const auto t_all1 = Clock::now();
        add_ns(g_cmf_total_ns, t_all0, t_all1);
        return;
    }
    // ------------------------------------------------------------
    // Fixed-lam fast path: 2D-2D binary factor
    //   - Only recompute the LLT (and message.lam) ONCE when cache is invalid.
    //   - In subsequent calls: keep message.lam unchanged, update only message.eta.
    // ------------------------------------------------------------
    if (d0_ == 2 && d1_ == 2 && D_ == 4) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;
        using Mat4 = Eigen::Matrix<double, 4, 4>;

        const auto t_misc0 = Clock::now();
        const auto& eta_dyn = factor.eta();
        const auto& lam_dyn = factor.lam();
        assert(eta_dyn.size() == 4);
        assert(lam_dyn.rows() == 4 && lam_dyn.cols() == 4);

        const Eigen::Map<const Vec4> eta_f0(eta_dyn.data());
        const Eigen::Map<const Mat4> lam_f0(lam_dyn.data());

        // Old messages (maps)
        const Eigen::Map<const Vec2> old0_eta(old_eta0.data());
        const Eigen::Map<const Vec2> old1_eta(old_eta1.data());
        const Eigen::Map<const Mat2> old0_lam(old_lam0.data());
        const Eigen::Map<const Mat2> old1_lam(old_lam1.data());

        // Beliefs (maps)
        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec2> b0_eta(b0.eta().data());
        const Eigen::Map<const Vec2> b1_eta(b1.eta().data());
        const Eigen::Map<const Mat2> b0_lam(b0.lam().data());
        const Eigen::Map<const Mat2> b1_lam(b1.lam().data());

    
        const auto t_misc1 = Clock::now();
        add_ns(g_cmf_misc_ns, t_misc0, t_misc1);

        // 2) Cache is valid: keep lam fixed (copy old lam), update eta only.
        // ---------------------
        // target = 0 (to v0)
        // ---------------------
        {
            const Vec2 eo  = eta_f0.template segment<2>(0);
            const Vec2 eno = eta_f0.template segment<2>(2) + (b1_eta - old1_eta);
            const Mat2 lono = lam_f0.template block<2, 2>(0, 2);

            const auto ts0 = Clock::now();
            const Vec2 y = llt0_.solve(eno);
            const auto ts1 = Clock::now();
            add_ns(g_cmf_solve0_ns, ts0, ts1);

            const auto tp0 = Clock::now();
            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old0_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[0];
            outMsg.lamRef().noalias() = old_lam0;  // fixed
            outMsg.etaRef() = outEta2;
            const auto tp1 = Clock::now();
            add_ns(g_cmf_pack0_ns, tp0, tp1);
        }

        // ---------------------
        // target = 1 (to v1)
        // ---------------------
        {
            const Vec2 eo  = eta_f0.template segment<2>(2);
            const Vec2 eno = eta_f0.template segment<2>(0) + (b0_eta - old0_eta);
            const Mat2 lono = lam_f0.template block<2, 2>(2, 0);

            const auto ts0 = Clock::now();
            const Vec2 y = llt1_.solve(eno);
            const auto ts1 = Clock::now();
            add_ns(g_cmf_solve1_ns, ts0, ts1);

            const auto tp0 = Clock::now();
            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old1_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[1];
            outMsg.lamRef().noalias() = old_lam1;  // fixed
            outMsg.etaRef() = outEta2;
            const auto tp1 = Clock::now();
            add_ns(g_cmf_pack1_ns, tp0, tp1);
        }

        const auto ts0 = Clock::now();
        messages.swap(messages_next);
        const auto ts1 = Clock::now();
        add_ns(g_cmf_swap_ns, ts0, ts1);

        const auto t_all1 = Clock::now();
        add_ns(g_cmf_total_ns, t_all0, t_all1);
        return;
    }
}

} // namespace gbp
