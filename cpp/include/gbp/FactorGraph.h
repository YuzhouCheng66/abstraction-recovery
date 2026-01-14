#pragma once
#include <vector>
#include <queue>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "gbp/VariableNode.h"
#include "gbp/Factor.h"

namespace gbp {

class FactorGraph {
public:
    bool nonlinear_factors = false; // 目前你的phase: false/unused
    double eta_damping = 0.0;

    std::vector<std::unique_ptr<VariableNode>> var_nodes;
    std::vector<std::unique_ptr<Factor>> factors;

    // residual scheduling on variables (by id)
    double residual_eps = 1e-6; // kept for compatibility, but NOT used for early-break (Python parity)
    std::vector<double> var_residual;

    // --- Max-heap on residual (positive) with Python-equivalent tie-break on varID ---
    // Python heap element: (-r, varID, var) in a MIN-heap
    // Equivalent pop order: r desc, varID asc
    struct HeapEntry {
        double r;   // residual (>=0)
        int varID;
        bool operator<(const HeapEntry& other) const {
            if (r != other.r) return r < other.r;  // larger r first
            return varID > other.varID;            // smaller varID first
        }
    };
    std::priority_queue<HeapEntry> var_heap;

    VariableNode* addVariable(int id, int dofs);

    // ===== Python-equivalent Factor construction =====
    // z:                list of measurement blocks
    // measurement_lambda: list of precision blocks
    // meas_fn(x):        list of predicted measurement blocks
    // jac_fn(x):         list of Jacobian blocks
    Factor* addFactor(
        int id,
        const std::vector<VariableNode*>& vars,
        const std::vector<Eigen::VectorXd>& z,
        const std::vector<Eigen::MatrixXd>& measurement_lambda,
        std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas_fn,
        std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)> jac_fn
    );

    void connect(Factor* f, VariableNode* v, int local_idx);

    // synchronous iteration (like Python synchronous_iteration)
    void synchronousIteration(bool robustify=false);

    // residual iteration (variable heap) (like residual_iteration_var_heap)
    void residualIterationVarHeap(int max_updates = -1);

    // keep factor.adj_beliefs in sync (like Python variable->factor copy)
    void syncAllFactorAdjBeliefs();

    // ======================
    // Batch (joint) utilities
    // ======================
    struct JointInfResult {
        Eigen::VectorXd eta;
        Eigen::SparseMatrix<double> lam;
        std::vector<int> var_ix;  // varID -> offset
        int total_dim = 0;
    };

    JointInfResult jointDistributionInfSparse() const;
    Eigen::VectorXd jointMAPSparse(double diag_jitter = 1e-12) const;

private:
    static constexpr double kJitter = 1e-12;
};
} // namespace gbp