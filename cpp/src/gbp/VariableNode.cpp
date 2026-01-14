#include "gbp/VariableNode.h"
#include "gbp/Factor.h"
#include <cassert>
#include <stdexcept>

namespace gbp {

VariableNode::VariableNode(int id_, int dofs_)
    : id(id_),
      variableID(id_),
      dofs(dofs_),
      dim(dofs_),
      active(true),
      prior(dofs_),
      belief(dofs_),
      mu(Eigen::VectorXd::Zero(dofs_)),
      Sigma(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      GT(Eigen::VectorXd::Zero(dofs_)),
      eta_acc_(Eigen::VectorXd::Zero(dofs_)),
      lam_acc_(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      lam_work_(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      I_(Eigen::MatrixXd::Identity(dofs_, dofs_)),
      llt_(),
      compute_sigma_(true)
{
    // Nothing else
}

VariableNode::VariableNode()
    : id(-1),
      variableID(-1),
      dofs(0),
      dim(0),
      active(true),
      prior(0),
      belief(0),
      mu(),
      Sigma(),
      GT(),
      eta_acc_(),
      lam_acc_(),
      lam_work_(),
      I_(),
      llt_(),
      compute_sigma_(true)
{
    // Nothing else
}

void VariableNode::updateBelief() {
    if (!active) return;
    if (dofs <= 0) return;

    // Ensure caches are allocated once and match dofs
    ensureCache_();

    // -------------------------
    // Accumulate belief in scratch (information form):
    //   eta_acc = prior.eta + sum_k msg_k.eta
    //   lam_acc = prior.lam + sum_k msg_k.lam
    // -------------------------
    eta_acc_.noalias() = prior.eta();
    lam_acc_.noalias() = prior.lam();

    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;

        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());

        // Accumulate incoming messages
        eta_acc_.noalias() += f->messages[k].eta();
        lam_acc_.noalias() += f->messages[k].lam();
    }

    // Write back to belief (keep external semantics identical)
    belief.setEta(eta_acc_);
    belief.setLam(lam_acc_);

    // -------------------------
    // Compute mu, Sigma from belief (SPD expected):
    //   (lam + jitter*I) * mu    = eta
    //   (lam + jitter*I) * Sigma = I
    // -------------------------
    lam_work_.noalias() = lam_acc_;
    lam_work_.diagonal().array() += kJitter;

    llt_.compute(lam_work_);
    if (llt_.info() != Eigen::Success) {
        throw std::runtime_error("VariableNode::updateBelief: LLT failed (matrix not SPD?)");
    }

    // mu = inv(lam)*eta
    mu.noalias() = llt_.solve(eta_acc_);

    // Sigma = inv(lam)*I (optional)
    if (compute_sigma_) {
        Sigma.noalias() = llt_.solve(I_);
    }
}

void VariableNode::sendBeliefToFactors() {
    if (!active) return;

    for (const auto& aref : adj_factors) {
        Factor* f = aref.factor;
        const int k = aref.local_idx;

        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->adj_beliefs.size());

        // Push current belief to factor cache
        f->adj_beliefs[k].setEta(belief.eta());
        f->adj_beliefs[k].setLam(belief.lam());
    }
}

} // namespace gbp
