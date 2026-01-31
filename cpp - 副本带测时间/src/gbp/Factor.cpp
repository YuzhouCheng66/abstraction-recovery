#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

#include <cassert>
#include <stdexcept>
#include <utility>   // std::move
#include <cstring>   // std::memcpy
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
    const int max_d = (d0_ > d1_) ? d0_ : d1_;

    // factor scratch
    eta_f_.resize(D_);
    lam_f_.resize(D_, D_);

    // Schur scratch
    lnono_.resize(max_d, max_d);
    Y_.resize(max_d, max_d);
    y_.resize(max_d);

    // tmp for msg computations
    tmpLam_.resize(max_d, max_d);
    tmpEta_.resize(max_d);

    // message outputs write directly into messages_next
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
    if (&linpoint_in != &linpoint) {
        linpoint = linpoint_in;
    }

    auto pred = meas_fn(linpoint);
    const int D = (int)linpoint.size();

    if (!jcache_valid_) {
        auto J = jac_fn(linpoint);

        if (J.size() != measurement.size() ||
            J.size() != measurement_lambda.size() ||
            J.size() != pred.size()) {
            throw std::runtime_error("computeFactor: block list size mismatch among J, measurement, measurement_lambda, pred");
        }

        J_cache_ = std::move(J);
        JO_cache_.resize(J_cache_.size());

        lambda_cache_.resize(D, D);
        lambda_cache_.setZero();

        for (size_t i = 0; i < J_cache_.size(); ++i) {
            const Eigen::MatrixXd& Ji = J_cache_[i];
            const Eigen::MatrixXd& Oi = measurement_lambda[i];

            JO_cache_[i].resize(Ji.cols(), Oi.cols());
            JO_cache_[i].noalias() = Ji.transpose() * Oi;

            lambda_cache_.noalias() += JO_cache_[i] * Ji;
        }

        jcache_valid_ = true;
        lamcache_set_ = false;
    } else {
        if (J_cache_.size() != measurement.size() ||
            J_cache_.size() != measurement_lambda.size() ||
            J_cache_.size() != pred.size()) {
            throw std::runtime_error("computeFactor(cached): block list size mismatch among cached J, measurement, measurement_lambda, pred");
        }
        if (lambda_cache_.rows() != D || lambda_cache_.cols() != D) {
            jcache_valid_ = false;
            computeFactor(linpoint_in, update_self);
            return;
        }
    }

    eta_f_.resize(D);
    eta_f_.setZero();

    for (size_t i = 0; i < J_cache_.size(); ++i) {
        const Eigen::MatrixXd& Ji = J_cache_[i];
        const Eigen::VectorXd& zi = measurement[i];
        const Eigen::VectorXd& hi = pred[i];

        const int m = (int)zi.size();

        if (ri_cf_.size() < m) ri_cf_.resize(m);
        auto ri = ri_cf_.head(m);
        ri.noalias() = Ji * linpoint;
        ri += zi;
        ri -= hi;

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

    // Old message views (Map)
    const auto old_eta0 = messages[0].eta();
    const auto old_lam0 = messages[0].lam();
    const auto old_eta1 = messages[1].eta();
    const auto old_lam1 = messages[1].lam();

    const double a = eta_damping;

    // ------------------------------------------------------------
    // Fast path: 2D-2D binary factor (fixed-size Eigen kernels)
    // ------------------------------------------------------------
    if (d0_ == 2 && d1_ == 2 && D_ == 4) {
        using Vec2 = Eigen::Vector2d;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix2d;
        using Mat4 = Eigen::Matrix<double, 4, 4>;

        // Use raw pointers (avoid Map-return overhead in hot path).
        const double* eta_ptr_f = factor.etaData();
        const double* lam_ptr_f = factor.lamData();
        const Eigen::Map<const Vec4> eta_f0(eta_ptr_f);
        const Eigen::Map<const Mat4> lam_f0(lam_ptr_f);

        const Eigen::Map<const Vec2> old0_eta(messages[0].etaData());
        const Eigen::Map<const Vec2> old1_eta(messages[1].etaData());
        const Eigen::Map<const Mat2> old0_lam(messages[0].lamData());
        const Eigen::Map<const Mat2> old1_lam(messages[1].lamData());

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec2> b0_eta(b0.etaData());
        const Eigen::Map<const Vec2> b1_eta(b1.etaData());
        const Eigen::Map<const Mat2> b0_lam(b0.lamData());
        const Eigen::Map<const Mat2> b1_lam(b1.lamData());

        const double s = 1.0 - a;

        // ---------------------
        // target = 0 (to v0, eliminate v1)
        // ---------------------
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
            llt_valid0_ = true;

            const Mat2 Y = llt0_.solve(lnoo);
            const Vec2 y = llt0_.solve(eno);

            Mat2 outLam2 = loo - lono * Y;
            Vec2 outEta2 = eo  - lono * y;

            if (a != 0.0) {
                outLam2 *= s;
                outEta2 *= s;
                outLam2.noalias() += a * old0_lam;
                outEta2.noalias() += a * old0_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[0];
            // Aggressive hot path: raw pointer write-back (NO Map construction, NO lamRef invalidation)
            // NOTE: messages' lam is never factorized (no mu()/Sigma() calls), so we do not maintain LLT cache here.
            std::memcpy(outMsg.etaData(), outEta2.data(), 2 * sizeof(double));
            std::memcpy(outMsg.lamData(), outLam2.data(), 4 * sizeof(double));
        }

        // ---------------------
        // target = 1 (to v1, eliminate v0)
        // ---------------------
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
            if (llt1_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (2D fast path, target=1)");
            }
            llt_valid1_ = true;

            const Mat2 Y = llt1_.solve(lnoo);
            const Vec2 y = llt1_.solve(eno);

            Mat2 outLam2 = loo - lono * Y;
            Vec2 outEta2 = eo  - lono * y;

            if (a != 0.0) {
                outLam2 *= s;
                outEta2 *= s;
                outLam2.noalias() += a * old1_lam;
                outEta2.noalias() += a * old1_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[1];
            std::memcpy(outMsg.etaData(), outEta2.data(), 2 * sizeof(double));
            std::memcpy(outMsg.lamData(), outLam2.data(), 4 * sizeof(double));
        }

        messages.swap(messages_next);
        return;
    }

    // Generic path
    for (int target = 0; target < 2; ++target) {
        // 1) eta_f_, lam_f_ = factor + belief_correction (no resize)
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

        const int d_o  = (target == 0) ? d0_ : d1_;
        const int d_no = (target == 0) ? d1_ : d0_;

        auto eo  = (target == 0) ? eta_f_.segment(0,   d0_) : eta_f_.segment(d0_, d1_);
        auto eno = (target == 0) ? eta_f_.segment(d0_, d1_) : eta_f_.segment(0,   d0_);

        auto loo_view   = (target == 0) ? lam_f_.block(0,   0,   d0_, d0_) : lam_f_.block(d0_, d0_, d1_, d1_);
        auto lono_view  = (target == 0) ? lam_f_.block(0,   d0_, d0_, d1_) : lam_f_.block(d0_, 0,   d1_, d0_);
        auto lnoo_view  = (target == 0) ? lam_f_.block(d0_, 0,   d1_, d0_) : lam_f_.block(0,   d0_, d0_, d1_);
        auto lnono_view = (target == 0) ? lam_f_.block(d0_, d0_, d1_, d1_) : lam_f_.block(0,   0,   d0_, d0_);

        auto lnono = lnono_.topLeftCorner(d_no, d_no);
        lnono.noalias() = lnono_view;
        lnono.diagonal().array() += kJitter;

        llt_.compute(lnono);
        if (llt_.info() != Eigen::Success) {
            throw std::runtime_error("LLT failed in Factor::computeMessages");
        }

        auto Y = Y_.topLeftCorner(d_no, d_o);
        Y.noalias() = llt_.solve(lnoo_view);

        auto y = y_.head(d_no);
        y.noalias() = llt_.solve(eno);

        utils::NdimGaussian& outMsg = messages_next[target];
        auto outLam = outMsg.lamRef();
        auto outEta = outMsg.etaRef();

        outLam.noalias() = loo_view;
        outLam.noalias() -= (lono_view * Y);

        outEta.noalias() = eo;
        outEta.noalias() -= (lono_view * y);

        if (a != 0.0) {
            const double ss = 1.0 - a;
            outLam *= ss;
            outEta *= ss;

            if (target == 0) {
                outLam.noalias() += a * old_lam0;
                outEta.noalias() += a * old_eta0;
            } else {
                outLam.noalias() += a * old_lam1;
                outEta.noalias() += a * old_eta1;
            }
        }
    }

    messages.swap(messages_next);
}

void Factor::computeMessagesFixedLam(double eta_damping) {
    using Clock = std::chrono::high_resolution_clock;
    const auto t_all0 = Clock::now();
    g_cmf_calls.fetch_add(1, std::memory_order_relaxed);

    auto add_ns = [](std::atomic<long long>& acc, const Clock::time_point& a, const Clock::time_point& b) {
        acc.fetch_add((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(),
                      std::memory_order_relaxed);
    };

    // 1) one-time init: do full update once
    if (!fixed_lam_valid_) {
        const auto t0 = Clock::now();
        computeMessages(eta_damping);

        // Sync lambdas into ping-pong buffers to avoid stale lambdas after swap
        if (!messages.empty() && messages_next.size() == messages.size()) {
            for (size_t k = 0; k < messages.size(); ++k) {
                messages_next[k].lamRef().noalias() = messages[k].lam();
            }
        }

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

    // Unary
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

    // Old messages (avoid Map-return overhead in the hot path)
    const double* old_eta0_ptr = messages[0].etaData();
    const double* old_eta1_ptr = messages[1].etaData();
    const double* old_lam0_ptr = messages[0].lamData();
    const double* old_lam1_ptr = messages[1].lamData();

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

    // Fixed-lam fast path: 2D-2D
    {
        using Vec2 = Eigen::Vector2d;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix2d;
        using Mat4 = Eigen::Matrix<double, 4, 4>;
        using Stride2 = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

        const auto t_misc0 = Clock::now();
        // Factor blocks via raw pointers
        const double* eta_ptr_f = factor.etaData();
        const double* lam_ptr_f = factor.lamData();
        const Eigen::Map<const Vec4> eta_f0(eta_ptr_f);
        const Eigen::Map<const Mat4> lam_f0(lam_ptr_f);

        // Old messages (maps over raw pointers)
        const Eigen::Map<const Vec2> old0_eta(old_eta0_ptr);
        const Eigen::Map<const Vec2> old1_eta(old_eta1_ptr);
        const Eigen::Map<const Mat2> old0_lam(old_lam0_ptr);
        const Eigen::Map<const Mat2> old1_lam(old_lam1_ptr);

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec2> b0_eta(b0.etaData());
        const Eigen::Map<const Vec2> b1_eta(b1.etaData());

        const auto t_misc1 = Clock::now();
        add_ns(g_cmf_misc_ns, t_misc0, t_misc1);

        // target=0
        {
            const auto ts0 = Clock::now();
            const Vec2 eo  = eta_f0.template segment<2>(0);
            const Vec2 eno = eta_f0.template segment<2>(2) + (b1_eta - old1_eta);
            const Mat2 lono = lam_f0.template block<2, 2>(0, 2);
            const Vec2 y = llt0_.solve(eno);
            const auto ts1 = Clock::now();
            add_ns(g_cmf_solve0_ns, ts0, ts1);

            
            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old0_eta;
            }
            
            const auto tp0 = Clock::now();
            utils::NdimGaussian& outMsg = messages_next[0];
            // Raw pointer write-back (avoid Map-return overhead)
            
            double* p = outMsg.etaData();
            
            p[0] = outEta2[0];
            p[1] = outEta2[1];
            const auto tp1 = Clock::now();
            
            
            add_ns(g_cmf_pack0_ns, tp0, tp1);
        }

        // target=1
        {
            const auto ts0 = Clock::now();
            const Vec2 eo  = eta_f0.template segment<2>(2);
            const Vec2 eno = eta_f0.template segment<2>(0) + (b0_eta - old0_eta);
            const Mat2 lono = lam_f0.template block<2, 2>(2, 0);
            const Vec2 y = llt1_.solve(eno);
            
            const auto ts1 = Clock::now();
            add_ns(g_cmf_solve1_ns, ts0, ts1);


            Vec2 outEta2 = eo - lono * y;


            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old1_eta;
            }

            
            const auto tp0 = Clock::now();
            utils::NdimGaussian& outMsg = messages_next[1];
            double* p = outMsg.etaData();
            p[0] = outEta2[0];
            p[1] = outEta2[1];
            
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
