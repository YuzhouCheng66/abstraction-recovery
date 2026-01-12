#pragma once
#include <vector>
#include <queue>
#include <Eigen/Dense>
#include "gbp/VariableNode.h"
#include "gbp/Factor.h"

namespace gbp {

class FactorGraph {
public:
    bool nonlinear_factors = false; // your current phase: false/unused
    double eta_damping = 0.0;

    std::vector<std::unique_ptr<VariableNode>> var_nodes;
    std::vector<std::unique_ptr<Factor>> factors;

    // residual scheduling on variables (by id)
    double residual_eps = 1e-6;
    std::vector<double> var_residual;

    struct HeapEntry {
        double neg_r;
        int varID;
        bool operator<(const HeapEntry& other) const {
            // priority_queue in C++ is max-heap by default on operator<,
            // we want max by residual => compare neg_r (more negative means larger residual).
            // simplest: store (r) and compare r; but keep parity with Python using neg.
            return neg_r > other.neg_r; // reverse for "min-heap behavior"
        }
    };

    std::priority_queue<HeapEntry> var_heap;

    VariableNode* addVariable(int id, int dofs);
    Factor* addFactor(int id, const std::vector<VariableNode*>& vars);

    void connect(Factor* f, VariableNode* v, int local_idx);

    // synchronous iteration (like Python synchronous_iteration)
    void synchronousIteration(bool robustify=false);

    // residual iteration (variable heap) (like residual_iteration_var_heap)
    void residualIterationVarHeap(int max_updates = -1);

    // keep factor.adj_beliefs in sync (like Python: variable sends belief to factors)
    void syncAllFactorAdjBeliefs();
};

} // namespace gbp
