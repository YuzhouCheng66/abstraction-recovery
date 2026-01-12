#include "gbp/VariableNode.h"
#include "gbp/Factor.h"
#include <cassert>

namespace gbp {

VariableNode::VariableNode(int id, int dofs_)
    : id(id),
      variableID(id),
      dofs(dofs_),
      dim(dofs_),
      prior(dofs_),
      belief(dofs_),
      mu(Eigen::VectorXd::Zero(dofs_)),
      Sigma(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      GT(Eigen::VectorXd::Zero(dofs_)) {}

VariableNode::VariableNode()
    : id(-1),
      variableID(-1),
      dofs(0),
      dim(0),
      prior(0),
      belief(0),
      mu(),
      Sigma(),
      GT() {}

void VariableNode::updateBelief() {
    if (!active) return;

    // belief = prior + sum incoming messages
    belief.setEta(prior.eta());
    belief.setLam(prior.lam());

    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;
        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());

        // Accumulate incoming messages
        belief.setEta(belief.eta() + f->messages[k].eta());
        belief.setLam(belief.lam() + f->messages[k].lam());
    }

    // Compute mu, Sigma from belief (SPD expected)
    // Add a tiny jitter for numerical safety (optional)
    Eigen::MatrixXd lam = belief.lam();
    lam.diagonal().array() += 1e-12;

    Eigen::LLT<Eigen::MatrixXd> llt(lam);
    // If you suspect near-singular, switch to LDLT
    mu = llt.solve(belief.eta());
    Sigma = llt.solve(Eigen::MatrixXd::Identity(dofs, dofs));
}

void VariableNode::sendBeliefToFactors() {
    if (!active) return;
    
    // Send the updated belief to each adjacent factor
    for (const auto& aref : adj_factors) {
        Factor* f = aref.factor;
        int k = aref.local_idx;
        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->adj_beliefs.size());
        
        // Update factor's adj_beliefs[k] with this variable's belief
        f->adj_beliefs[k].setEta(belief.eta());
        f->adj_beliefs[k].setLam(belief.lam());
    }
}

} // namespace gbp
