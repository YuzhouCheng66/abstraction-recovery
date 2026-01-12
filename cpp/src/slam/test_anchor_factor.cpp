#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

int main() {
    std::cout << "=== Test Anchor Factor ===\n\n";
    
    // Create a single variable node
    gbp::VariableNode v0(0, 2);
    v0.GT = Eigen::Vector2d(0.0, 0.0);
    
    // Initialize prior to tiny value
    v0.prior.setLam(Eigen::Matrix2d::Identity() * 1e-12);
    v0.prior.setEta(Eigen::Vector2d::Zero());
    v0.updateBelief();
    
    std::cout << "Variable 0 (before):\n";
    std::cout << "  mu = (" << v0.mu(0) << ", " << v0.mu(1) << ")\n";
    std::cout << "  belief.lam norm = " << v0.belief.lam().norm() << "\n";
    std::cout << "  belief.eta norm = " << v0.belief.eta().norm() << "\n\n";
    
    // Create anchor factor: very strong prior on [0, 0]
    gbp::Factor anchor(0, {&v0});
    anchor.measurements = {Eigen::Vector2d(0.0, 0.0)};
    anchor.precisions = {Eigen::Matrix2d::Identity() / (1e-4 * 1e-4)};
    anchor.computeFactor();
    
    // IMPORTANT: Create the backward relationship - variable needs to know about factor
    v0.adj_factors.push_back(gbp::AdjFactorRef{&anchor, 0});  // factor 0, local_idx 0
    
    std::cout << "Anchor Factor:\n";
    std::cout << "  factor.lam norm = " << anchor.factor.lam().norm() << "\n";
    std::cout << "  factor.eta norm = " << anchor.factor.eta().norm() << "\n";
    std::cout << "  factor.eta = " << anchor.factor.eta().transpose() << "\n";
    std::cout << "  factor.lam:\n" << anchor.factor.lam() << "\n\n";
    
    // Sync belief to factor's adj_beliefs
    anchor.syncAdjBeliefsFromVariables();
    std::cout << "After syncAdjBeliefsFromVariables:\n";
    std::cout << "  adj_beliefs[0].lam norm = " << anchor.adj_beliefs[0].lam().norm() << "\n";
    std::cout << "  adj_beliefs[0].eta norm = " << anchor.adj_beliefs[0].eta().norm() << "\n\n";
    
    // Compute message
    anchor.computeMessages(0.0);
    std::cout << "After computeMessages(0.0):\n";
    std::cout << "  msg[0].lam norm = " << anchor.messages[0].lam().norm() << "\n";
    std::cout << "  msg[0].eta norm = " << anchor.messages[0].eta().norm() << "\n";
    std::cout << "  msg[0].lam:\n" << anchor.messages[0].lam() << "\n";
    std::cout << "  msg[0].eta = " << anchor.messages[0].eta().transpose() << "\n\n";
    
    // Update variable's belief with the message
    v0.updateBelief();
    std::cout << "Variable 0 (after updateBelief with message):\n";
    std::cout << "  mu = (" << v0.mu(0) << ", " << v0.mu(1) << ")\n";
    std::cout << "  belief.lam:\n" << v0.belief.lam() << "\n";
    std::cout << "  belief.eta = " << v0.belief.eta().transpose() << "\n\n";
    
    return 0;
}
