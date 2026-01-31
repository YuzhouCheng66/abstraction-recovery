#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <utility>
#include <cstdint>
#include "NdimGaussian.h"

namespace gbp {

// Profiling utilities for VariableNode::updateBelief
void printUpdateBeliefProfile();
void resetUpdateBeliefProfile();

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

    // Ground truth (for testing/evaluation)
    Eigen::VectorXd GT;
    // Adjacency: list of (factor*, local_idx)
    std::vector<AdjFactorRef> adj_factors;
    std::vector<Factor*> adj_factors_raw;  // Direct pointers for SLAM graph building

    explicit VariableNode(int id, int dofs_);
    VariableNode();  // Default constructor

    // Main routines
    void updateBelief();          // prior + sum incoming factor->messages[local_idx]


private:
    static constexpr double kJitter = 1e-12;

    // ---- scratch/cache (allocated once, reused; dimensions fixed after construction) ----
    Eigen::VectorXd eta_acc_;     // size dofs: accumulator for eta
    Eigen::MatrixXd lam_acc_;     // size dofs x dofs: accumulator for lambda
    Eigen::MatrixXd lam_work_;    // size dofs x dofs: lam_acc_ + jitter (factorization input)

private:
    // Ensure caches are initialized to current dofs (safe-guard).
    // In a well-formed graph dofs won't change after ctor, so this is rarely hit.
    inline void ensureCache_() {
        if (dofs <= 0) return;
        if (eta_acc_.size() == dofs) return;

        eta_acc_.setZero(dofs);
        lam_acc_.setZero(dofs, dofs);
        lam_work_.setZero(dofs, dofs);

    }
};

} // namespace gbp
