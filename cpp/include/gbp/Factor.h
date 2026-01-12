#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <cassert>
#include "NdimGaussian.h"

namespace gbp {

class VariableNode;

class Factor {
public:
    int id = -1;
    int factorID = -1;
    int dim = 0;
    bool active = true;

    // Only 1-ary or 2-ary factor in your project
    std::vector<VariableNode*> adj_var_nodes;   // size 1 or 2
    std::vector<VariableNode*> var_nodes;       // Alias for SLAM compatibility
    std::vector<int> adj_vIDs;                  // cached ids (same size)

    // Incoming beliefs at factor side (optional mirror; keep for parity)
    // In this minimal version, FactorGraph will keep them updated implicitly by reading var->belief if needed.
    // But messages update needs "adj_beliefs - messages" term, so we store adj_beliefs like Python.
    std::vector<utils::NdimGaussian> adj_beliefs;

    // Outgoing messages from factor to each variable (same size as adj_var_nodes)
    std::vector<utils::NdimGaussian> messages;

    // The factor distribution in information form over concatenated variables
    utils::NdimGaussian factor; // dim = dofs(v0)+dofs(v1) or dofs(v0)
    
    // Measurements and precisions (for direct construction without nonlinear factors)
    std::vector<Eigen::VectorXd> measurements;
    std::vector<Eigen::MatrixXd> precisions;

    // For local damping / relinearization (optional)
    double eta_damping_local = 0.0;

    explicit Factor(int id, const std::vector<VariableNode*>& vars);
    Factor();  // Default constructor

    // Call after you connect factor to variables:
    // initialize adj_beliefs from current variable beliefs and messages to zeros.
    void syncAdjBeliefsFromVariables();

    // Compute factor distribution from measurements and linearization point
    void computeFactor();

    // Core GBP kernel
    void computeMessages(double eta_damping);

private:
    static constexpr double kJitter = 1e-12;
};

} // namespace gbp
