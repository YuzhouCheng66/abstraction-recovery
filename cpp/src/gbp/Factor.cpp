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

void printComputeFactorProfile() {
    int calls = g_cf_call_count.load();
    if (calls == 0) {
        std::cout << "[computeFactor Profile] No calls recorded.\n";
        return;
    }
    auto toMs = [](long long ns) { return ns / 1e6; };
    std::cout << "\n=== computeFactor Detailed Profile (" << calls << " calls) ===\n";
    std::cout << "  jac_fn:        " << toMs(g_cf_jac_ns.load()) << " ms\n";
    std::cout << "  meas_fn:       " << toMs(g_cf_meas_ns.load()) << " ms\n";
    std::cout << "  loop (lambda): " << toMs(g_cf_loop_lam_ns.load()) << " ms\n";
    std::cout << "  loop (eta):    " << toMs(g_cf_loop_eta_ns.load()) << " ms\n";
    std::cout << "==============================================\n";
}

void resetComputeFactorProfile() {
    g_cf_jac_ns = 0;
    g_cf_meas_ns = 0;
    g_cf_loop_lam_ns = 0;
    g_cf_loop_eta_ns = 0;
    g_cf_call_count = 0;
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

    // Allocate adj_beliefs/messages (fixed per-variable dofs)
    adj_beliefs.reserve(adj_var_nodes.size());
    messages.reserve(adj_var_nodes.size());
    for (auto* v : adj_var_nodes) {
        assert(v != nullptr);
        adj_beliefs.emplace_back(v->dofs);
        messages.emplace_back(v->dofs);
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

    // outputs fixed per target
    new_eta_[0].resize(d0_);
    new_lam_[0].resize(d0_, d0_);
    new_eta_[1].resize(d1_);
    new_lam_[1].resize(d1_, d1_);
}

// ==============================
// sync, factor computation
// ==============================

void Factor::syncAdjBeliefsFromVariables() {
    for (int i = 0; i < (int)adj_var_nodes.size(); ++i) {
        auto* v = adj_var_nodes[i];
        adj_beliefs[i].setEta(v->belief.eta());
        adj_beliefs[i].setLam(v->belief.lam());
    }
}

void Factor::invalidateJacobianCache() {
    jcache_valid_ = false;
    lamcache_set_ = false;
    J_cache_.clear();
    JO_cache_.clear();
    lambda_cache_.resize(0, 0);
}

void Factor::computeFactor(const Eigen::VectorXd& linpoint_in, bool update_self) {
    ++g_cf_call_count;

    linpoint = linpoint_in;

    // meas_fn may depend on current linpoint (e.g. via k), so we evaluate it every call.
    auto t0 = std::chrono::high_resolution_clock::now();
    auto pred = meas_fn(linpoint);
    auto t1 = std::chrono::high_resolution_clock::now();
    g_cf_meas_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    const int D = (int)linpoint.size();

    // Build Jacobian-derived caches lazily (assumes J and measurement_lambda do not change
    // across iterations for the current graph / abstraction).
    if (!jcache_valid_) {
        auto tJ0 = std::chrono::high_resolution_clock::now();
        auto J = jac_fn(linpoint);
        auto tJ1 = std::chrono::high_resolution_clock::now();
        g_cf_jac_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tJ1 - tJ0).count();

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

    auto t2 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < J_cache_.size(); ++i) {
        const Eigen::MatrixXd& Ji = J_cache_[i];
        const Eigen::VectorXd& zi = measurement[i];
        const Eigen::VectorXd& hi = pred[i];

        // ri = Ji * linpoint + zi - hi
        ri_cf_.resize((int)zi.size());
        ri_cf_.noalias() = Ji * linpoint;
        ri_cf_.noalias() += zi;
        ri_cf_.noalias() -= hi;

        // eta += (Ji^T * Oi) * ri  == JO_i * ri
        eta_f_.noalias() += JO_cache_[i] * ri_cf_;

        auto t_eta = std::chrono::high_resolution_clock::now();
        g_cf_loop_eta_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_eta - t2).count();
        t2 = t_eta;
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
        messages[0].setEta(factor.eta());
        messages[0].setLam(factor.lam());
        return;
    }

    // Binary: dimensions are fixed (d0_, d1_, D_)
    // old messages (copies) - unavoidable if messages[k] returns by value internally
    const Eigen::VectorXd old_eta0 = messages[0].eta();
    const Eigen::MatrixXd old_lam0 = messages[0].lam();
    const Eigen::VectorXd old_eta1 = messages[1].eta();
    const Eigen::MatrixXd old_lam1 = messages[1].lam();

    const double a = eta_damping;

    // We compute two directed messages: target=0 => to v0 (eliminate v1), target=1 => to v1 (eliminate v0)
    for (int target = 0; target < 2; ++target) {
        // ------------------------------------------------------------
        // 1) eta_f_, lam_f_ = factor + belief_correction (no resize)
        // ------------------------------------------------------------
        eta_f_.noalias() = factor.eta();
        lam_f_.noalias() = factor.lam();

        if (target == 0) {
            // incorporate v1 belief - old message1
            eta_f_.segment(d0_, d1_).noalias() += (adj_beliefs[1].eta() - old_eta1);
            lam_f_.block(d0_, d0_, d1_, d1_).noalias() += (adj_beliefs[1].lam() - old_lam1);
        } else {
            // incorporate v0 belief - old message0
            eta_f_.segment(0, d0_).noalias() += (adj_beliefs[0].eta() - old_eta0);
            lam_f_.block(0, 0, d0_, d0_).noalias() += (adj_beliefs[0].lam() - old_lam0);
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

        auto Y = Y_.topLeftCorner(d_no, d_o);
        Y.noalias() = llt_.solve(lnoo_view);

        auto y = y_.head(d_no);
        y.noalias() = llt_.solve(eno);

        // ------------------------------------------------------------
        // 5) outLam/outEta: write into fixed buffers (no resize)
        // ------------------------------------------------------------
        Eigen::MatrixXd& outLam = new_lam_[target];
        Eigen::VectorXd& outEta = new_eta_[target];

        // tmpLam = lono*Y (into top-left)
        auto tmpLam = tmpLam_.topLeftCorner(d_o, d_o);
        tmpLam.noalias() = lono_view * Y;

        // outLam = loo - tmpLam
        outLam.noalias() = loo_view;
        outLam.noalias() -= tmpLam;

        // tmpEta = lono*y (into head)
        auto tmpEta = tmpEta_.head(d_o);
        tmpEta.noalias() = lono_view * y;

        // outEta = eo - tmpEta
        outEta.noalias() = eo;
        outEta.noalias() -= tmpEta;

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

    // Commit outputs
    messages[0].setLam(new_lam_[0]);
    messages[0].setEta(new_eta_[0]);
    messages[1].setLam(new_lam_[1]);
    messages[1].setEta(new_eta_[1]);
}

} // namespace gbp
