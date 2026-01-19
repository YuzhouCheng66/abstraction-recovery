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

    // ------------------------------------------------------------
    // Fast path: 2D variables (fixed-size Eigen kernels)
    //   - Avoid dynamic-size Eigen overhead for the common dofs==2 case
    //   - Still uses LLT.solve (no explicit inverse)
    // ------------------------------------------------------------
    if (dofs == 2) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;

        Vec2 eta2 = prior.eta().head<2>();
        Mat2 lam2 = prior.lam().topLeftCorner<2, 2>();

        for (const auto& aref : adj_factors) {
            const Factor* f = aref.factor;
            const int k = aref.local_idx;

            assert(f != nullptr);
            assert(k >= 0 && k < (int)f->messages.size());

            eta2.noalias() += f->messages[k].eta().head<2>();
            lam2.noalias() += f->messages[k].lam().topLeftCorner<2, 2>();
        }

        // Write belief (information form) without extra copies
        belief.etaRef().head<2>() = eta2;
        belief.lamRef().topLeftCorner<2, 2>() = lam2;

        // Solve for mu (and optionally Sigma) using fixed-size LLT
        Eigen::LLT<Mat2> llt2;
        llt2.compute(lam2);
        if (llt2.info() != Eigen::Success) {
            lam2(0, 0) += kJitter;
            lam2(1, 1) += kJitter;
            llt2.compute(lam2);
            if (llt2.info() != Eigen::Success) {
                throw std::runtime_error("VariableNode::updateBelief (2D): LLT failed (matrix not SPD?)");
            }
        }

        mu.head<2>().noalias() = llt2.solve(eta2);

        if (compute_sigma_) {
            const Mat2 I2 = Mat2::Identity();
            Sigma.topLeftCorner<2, 2>().noalias() = llt2.solve(I2);
        }
        return;
    }

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

    // ---- VA: write belief without setEta/setLam overhead (no extra copies) ----
    belief.etaRef().noalias() = eta_acc_;
    belief.lamRef().noalias() = lam_acc_;

    // -------------------------
    // Compute mu, Sigma from belief (SPD expected):
    //   (lam + jitter*I) * mu    = eta
    //   (lam + jitter*I) * Sigma = I
    // -------------------------
    // ---- VB: avoid lam_work_ copy unless LLT fails ----
    // Try factorization without jitter first (common case: SPD already)
    llt_.compute(lam_acc_);
    if (llt_.info() != Eigen::Success) {
        lam_work_.noalias() = lam_acc_;
        lam_work_.diagonal().array() += kJitter;
        llt_.compute(lam_work_);
        if (llt_.info() != Eigen::Success) {
            throw std::runtime_error("VariableNode::updateBelief: LLT failed (matrix not SPD?)");
        }
    }


    // mu = inv(lam)*eta
    mu.noalias() = llt_.solve(eta_acc_);

    // Sigma = inv(lam)*I (optional)
    if (compute_sigma_) {
        Sigma.noalias() = llt_.solve(I_);
    }

}

} // namespace gbp
