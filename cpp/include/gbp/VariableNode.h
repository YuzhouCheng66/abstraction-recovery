#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <utility>
#include <cstdint>
#include "NdimGaussian.h"

namespace gbp {

class Factor;  // forward declaration

struct AdjFactorRef {
    Factor* factor = nullptr;
    int local_idx = -1; // this variable's index in factor->adj_var_nodes
};

class VariableNode {
public:
    int id = -1;
    int variableID = -1;
    int dofs = 0;
    int dim = 0;  // Alias for dofs (for SLAM compatibility)
    bool active = true;

    // Prior and belief (information form)
    utils::NdimGaussian prior;
    utils::NdimGaussian belief;

    // Derived moment parameters
    Eigen::VectorXd mu;
    Eigen::MatrixXd Sigma;

    // Ground truth (for testing/evaluation)
    Eigen::VectorXd GT;

    // Adjacency: list of (factor*, local_idx)
    std::vector<AdjFactorRef> adj_factors;
    std::vector<Factor*> adj_factors_raw;  // Direct pointers for SLAM graph building

    explicit VariableNode(int id, int dofs_);
    VariableNode();  // Default constructor

    // Main routines
    void updateBelief();          // prior + sum incoming factor->messages[local_idx]

    // Optional performance knob:
    // - Default: keep same semantics as before (compute Sigma every update).
    // - If you don't need Sigma downstream, set false to save a full solve.
    void setComputeSigma(bool on) { compute_sigma_ = on; }
    bool computeSigmaEnabled() const { return compute_sigma_; }

private:
    static constexpr double kJitter = 1e-12;

    // ---- scratch/cache (allocated once, reused; dimensions fixed after construction) ----
    Eigen::VectorXd eta_acc_;     // size dofs: accumulator for eta
    Eigen::MatrixXd lam_acc_;     // size dofs x dofs: accumulator for lambda

    Eigen::MatrixXd lam_work_;    // size dofs x dofs: lam_acc_ + jitter (factorization input)
    Eigen::MatrixXd I_;           // cached Identity(dofs,dofs) for Sigma solve

    Eigen::LLT<Eigen::MatrixXd> llt_; // reused LLT object (stores factorization)

    bool compute_sigma_ = true;   // see setComputeSigma()

private:
    // Ensure caches are initialized to current dofs (safe-guard).
    // In a well-formed graph dofs won't change after ctor, so this is rarely hit.
    inline void ensureCache_() {
        if (dofs <= 0) return;
        if (eta_acc_.size() == dofs) return;

        eta_acc_.setZero(dofs);
        lam_acc_.setZero(dofs, dofs);
        lam_work_.setZero(dofs, dofs);
        I_.setIdentity(dofs, dofs);

        mu.setZero(dofs);
        Sigma.setZero(dofs, dofs);
    }
};

} // namespace gbp
