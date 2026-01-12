#pragma once
#include <Eigen/Dense>
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

    void updateBelief(); // prior + sum incoming factor->messages[local_idx]
    void sendBeliefToFactors(); // Send updated belief to adjacent factors' adj_beliefs
};

} // namespace gbp
