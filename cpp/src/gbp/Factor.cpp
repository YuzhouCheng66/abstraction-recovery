#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

#include <cassert>
#include <stdexcept>
#include <utility>   // std::move
#include <cstring>   // std::memcpy


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

    if (d0_ == 3 && d1_ == 3 && D_ == 6) {
        eta0_3_ = eta_f_.template segment<3>(0);
        eta1_3_ = eta_f_.template segment<3>(3);

        loo0_3_ = lambda_cache_.template block<3, 3>(0, 0);
        lono0_3_ = lambda_cache_.template block<3, 3>(0, 3);
        lnoo0_3_ = lambda_cache_.template block<3, 3>(3, 0);
        lnono0_3_ = lambda_cache_.template block<3, 3>(3, 3);

        loo1_3_ = lambda_cache_.template block<3, 3>(3, 3);
        lono1_3_ = lambda_cache_.template block<3, 3>(3, 0);
        lnoo1_3_ = lambda_cache_.template block<3, 3>(0, 3);
        lnono1_3_ = lambda_cache_.template block<3, 3>(0, 0);
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
        const auto* eta_ptr_f = factor.etaData();
        const auto* lam_ptr_f = factor.lamData();
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
            const auto eo = eta_f0.template segment<2>(0);
            Vec2 eno = eta_f0.template segment<2>(2);
            eno.noalias() += (b1_eta - old1_eta);

            // --- lam blocks ---
            const auto loo  = lam_f0.template block<2, 2>(0, 0);
            const auto lono = lam_f0.template block<2, 2>(0, 2);
            const auto lnoo = lam_f0.template block<2, 2>(2, 0);

            // lnono: 你要在其上加 (b1_lam-old1_lam) 和 jitter，必须是 owning Mat2（显式拷贝）
            Mat2 lnono = lam_f0.template block<2, 2>(2, 2);
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
            const auto eo   = eta_f0.template segment<2>(2);
            Vec2 eno = eta_f0.template segment<2>(0);
            eno.noalias() += (b0_eta - old0_eta);

            const auto loo   = lam_f0.template block<2, 2>(2, 2);
            const auto lono  = lam_f0.template block<2, 2>(2, 0);
            const auto lnoo  = lam_f0.template block<2, 2>(0, 2);
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

    // ------------------------------------------------------------
    // Fast path: 3D-3D binary factor (synthetic SE2 hot path)
    // ------------------------------------------------------------
    if (d0_ == 3 && d1_ == 3 && D_ == 6) {
        using Vec3 = Eigen::Matrix<double, 3, 1>;
        using Mat3 = Eigen::Matrix<double, 3, 3>;

        const Eigen::Map<const Vec3> old0_eta(messages[0].etaData());
        const Eigen::Map<const Vec3> old1_eta(messages[1].etaData());
        const Eigen::Map<const Mat3> old0_lam(messages[0].lamData());
        const Eigen::Map<const Mat3> old1_lam(messages[1].lamData());

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec3> b0_eta(b0.etaData());
        const Eigen::Map<const Vec3> b1_eta(b1.etaData());
        const Eigen::Map<const Mat3> b0_lam(b0.lamData());
        const Eigen::Map<const Mat3> b1_lam(b1.lamData());

        const double s = 1.0 - a;

        {
            const Vec3& eo = eta0_3_;
            Vec3 eno = eta1_3_;
            eno.noalias() += (b1_eta - old1_eta);

            const Mat3& loo = loo0_3_;
            const Mat3& lono = lono0_3_;
            const Mat3& lnoo = lnoo0_3_;

            Mat3 lnono = lnono0_3_;
            lnono.noalias() += (b1_lam - old1_lam);
            lnono.diagonal().array() += kJitter;

            llt3_0_.compute(lnono);
            if (llt3_0_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (3D fast path, target=0)");
            }

            const Mat3 Y = llt3_0_.solve(lnoo);
            const Vec3 y = llt3_0_.solve(eno);

            Mat3 outLam3 = loo - lono * Y;
            Vec3 outEta3 = eo - lono * y;

            if (a != 0.0) {
                outLam3 *= s;
                outEta3 *= s;
                outLam3.noalias() += a * old0_lam;
                outEta3.noalias() += a * old0_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[0];
            std::memcpy(outMsg.etaData(), outEta3.data(), 3 * sizeof(double));
            std::memcpy(outMsg.lamData(), outLam3.data(), 9 * sizeof(double));
        }

        {
            const Vec3& eo = eta1_3_;
            Vec3 eno = eta0_3_;
            eno.noalias() += (b0_eta - old0_eta);

            const Mat3& loo = loo1_3_;
            const Mat3& lono = lono1_3_;
            const Mat3& lnoo = lnoo1_3_;

            Mat3 lnono = lnono1_3_;
            lnono.noalias() += (b0_lam - old0_lam);
            lnono.diagonal().array() += kJitter;

            llt3_1_.compute(lnono);
            if (llt3_1_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (3D fast path, target=1)");
            }

            const Mat3 Y = llt3_1_.solve(lnoo);
            const Vec3 y = llt3_1_.solve(eno);

            Mat3 outLam3 = loo - lono * Y;
            Vec3 outEta3 = eo - lono * y;

            if (a != 0.0) {
                outLam3 *= s;
                outEta3 *= s;
                outLam3.noalias() += a * old1_lam;
                outEta3.noalias() += a * old1_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[1];
            std::memcpy(outMsg.etaData(), outEta3.data(), 3 * sizeof(double));
            std::memcpy(outMsg.lamData(), outLam3.data(), 9 * sizeof(double));
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
    // 1) one-time init: do full update once
    if (!fixed_lam_valid_) {
        computeMessages(eta_damping);

        // Sync lambdas into ping-pong buffers to avoid stale lambdas after swap
        if (!messages.empty() && messages_next.size() == messages.size()) {
            for (size_t k = 0; k < messages.size(); ++k) {
                messages_next[k].lamRef().noalias() = messages[k].lam();
            }
        }

        fixed_lam_valid_ = true;
        return;
    }

    if (!active) return;

    // Unary
    if (is_unary_) {
        auto& outMsg = messages_next[0];
        outMsg.etaRef().noalias() = factor.eta();
        outMsg.lamRef().noalias() = factor.lam();
        messages.swap(messages_next);
        return;
    }

    // Old messages (avoid Map-return overhead in the hot path)
    const auto* old_eta0_ptr = messages[0].etaData();
    const auto* old_eta1_ptr = messages[1].etaData();
    const auto* old_lam0_ptr = messages[0].lamData();
    const auto* old_lam1_ptr = messages[1].lamData();

    const double a = eta_damping;

    // Only 2D-2D binary factors use the fixed-lam fast path
    if (!(d0_ == 2 && d1_ == 2 && D_ == 4)) {
        computeMessages(eta_damping);
        return;
    }

    // Fixed-lam fast path: 2D-2D
    {
        using Vec2 = Eigen::Vector2d;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix2d;
        using Mat4 = Eigen::Matrix<double, 4, 4>;

        // Factor blocks via raw pointers
        const auto* eta_ptr_f = factor.etaData();
        const auto* lam_ptr_f = factor.lamData();
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

        // target=0
        {
            const auto eo  = eta_f0.template segment<2>(0);
            //const Vec2 eno = eta_f0.template segment<2>(2) + (b1_eta - old1_eta);
            Vec2 eno = eta_f0.template segment<2>(2);
            eno.noalias() += (b1_eta - old1_eta);
            const auto lono = lam_f0.template block<2, 2>(0, 2);

            // llt0_ was factorized from l_no,no in computeFactor()
            const Vec2 y = llt0_.solve(eno);

            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old0_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[0];
            double* p = outMsg.etaData();
            p[0] = outEta2[0];
            p[1] = outEta2[1];
        }

        // target=1
        {
            const auto eo  = eta_f0.template segment<2>(2);
            Vec2 eno = eta_f0.template segment<2>(0);
            eno.noalias() += (b0_eta - old0_eta);
            const auto lono = lam_f0.template block<2, 2>(2, 0);

            const Vec2 y = llt1_.solve(eno);

            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old1_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[1];
            double* p = outMsg.etaData();
            p[0] = outEta2[0];
            p[1] = outEta2[1];
        }

        messages.swap(messages_next);
        return;
    }
}

} // namespace gbp
