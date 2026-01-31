#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <cassert>
#include <functional>
#include <stdexcept>

#include "NdimGaussian.h"


// Profiling utilities for computeFactor analysis
void printComputeFactorProfile();
void resetComputeFactorProfile();

// Profiling utilities for computeMessagesFixedLam (synchronous fixed-lam iteration)
void printComputeMessagesFixedLamProfile();
void resetComputeMessagesFixedLamProfile();

//

namespace gbp {

class VariableNode;

class Factor {
public:
    int factorID = -1;
    bool active = true;

    std::vector<VariableNode*> adj_var_nodes;
    std::vector<int> adj_vIDs;

    // Factor-to-variable messages (ping-pong buffers for C optimization)
    std::vector<utils::NdimGaussian, Eigen::aligned_allocator<utils::NdimGaussian>> messages;
    std::vector<utils::NdimGaussian, Eigen::aligned_allocator<utils::NdimGaussian>> messages_next;

    std::vector<Eigen::VectorXd> measurement;         // z_i
    std::vector<Eigen::MatrixXd> measurement_lambda;  // Λ_i

    std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas_fn;
    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)>  jac_fn;

    utils::NdimGaussian factor;
    Eigen::VectorXd linpoint;

    double eta_damping_local = 0.0;
    bool fixed_lam_valid_ = false;

    Factor(
        int id,
        const std::vector<VariableNode*>& vars,
        const std::vector<Eigen::VectorXd>& z,
        const std::vector<Eigen::MatrixXd>& lambda,
        std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas,
        std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)>  jac
    );

    void computeFactor(const Eigen::VectorXd& linpoint, bool update_self = true);
    // Jacobian/Lambda cache control (use when structure changes)
    void invalidateJacobianCache();
    void computeMessages(double eta_damping);
    void computeMessagesFixedLam(double eta_damping);

private:
    static constexpr double kJitter = 1e-12;

    // ==========================
    // Fixed dimensions (set once)
    // ==========================
    bool is_unary_  = false;
    bool is_binary_ = false;

    int d0_ = 0;   // dofs(var0) if binary, else dofs(var0) for unary
    int d1_ = 0;   // dofs(var1) if binary, else 0
    int D_  = 0;   // total dofs = d0_ + d1_

    // For binary Schur:
    // target==0 (msg to v0): d_o=d0_, d_no=d1_
    // target==1 (msg to v1): d_o=d1_, d_no=d0_
    inline int d_o_(int target)  const { return (target == 0) ? d0_ : d1_; }
    inline int d_no_(int target) const { return (target == 0) ? d1_ : d0_; }

    // ==========================
    // Workspace (allocated once)
    // ==========================
    // Copy of factor (eta, lam) with belief correction applied
    Eigen::VectorXd eta_f_;      // size: D_
    Eigen::MatrixXd lam_f_;      // size: D_ x D_

    // lnono copy (for jitter + factorization), sized to max(d0_, d1_)
    Eigen::MatrixXd lnono_;      // size: max(d0_,d1_) x max(d0_,d1_)

    // Solutions:
    // Y_:  (d_no x d_o)  and y_: (d_no)
    // Allocate at max sizes and use topLeftCorner/head when needed.
    Eigen::MatrixXd Y_;          // size: max_no x max_o
    Eigen::VectorXd y_;          // size: max_no

    Eigen::LLT<Eigen::MatrixXd> llt_;  // reusable factorization object

    // Temporaries (kept for potential parity / debugging; not required by C)
    Eigen::MatrixXd tmpLam_;     // size: max_o x max_o
    Eigen::VectorXd tmpEta_;     // size: max_o

    // ==========================
    // computeFactor workspace
    // ==========================
    Eigen::VectorXd ri_cf_;      // residual per measurement block (m)
    Eigen::VectorXd tmpv_cf_;    // tmp vector: Oi * ri (m)
    Eigen::MatrixXd tmpm_cf_;    // tmp matrix: Oi * Ji (m x D)

    // ==========================
    // Jacobian / Lambda cache (J assumed constant across iterations)
    // ==========================
    bool jcache_valid_ = false;   // whether J/JO/Lambda cache is valid
    bool lamcache_set_ = false;   // whether factor.lam has been set from cache
    std::vector<Eigen::MatrixXd> J_cache_;   // blocks Ji (m x D)
    std::vector<Eigen::MatrixXd> JO_cache_;  // blocks (Ji^T * Oi) (D x m)
    Eigen::MatrixXd lambda_cache_;           // sum_i (Ji^T * Oi * Ji) (D x D)


    // =====================
    // Fixed-lambda cache
    // =====================

    mutable Eigen::LLT<Eigen::Matrix2d> llt0_;  // target=0 消元用
    mutable Eigen::LLT<Eigen::Matrix2d> llt1_;  // target=1 消元用
    bool llt_valid0_ = false;
    bool llt_valid1_ = false;

private:
    void initWorkspace_();
};

} // namespace gbp
