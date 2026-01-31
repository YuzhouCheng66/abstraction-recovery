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
      GT(Eigen::VectorXd::Zero(dofs_)),
      mu2(Eigen::Vector2d::Zero()),
      mu(Eigen::VectorXd::Zero(dofs_)),
      eta_acc_(Eigen::VectorXd::Zero(dofs_)),
      lam_acc_(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      lam_work_(Eigen::MatrixXd::Zero(dofs_, dofs_))
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
      GT(),
      mu2(Eigen::Vector2d::Zero()),
      mu(Eigen::VectorXd::Zero(0)),
      eta_acc_(),
      lam_acc_(),
      lam_work_()
{
    // Nothing else
}

void VariableNode::updateBelief() {
    if (!active) return;
    if (dofs <= 0) return;

    ensureCache_();

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

        belief.etaRef().head<2>() = eta2;
        belief.lamRef().topLeftCorner<2, 2>() = lam2;
        return;
    }

    eta_acc_.noalias() = prior.eta();
    lam_acc_.noalias() = prior.lam();

    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;
        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());
        eta_acc_.noalias() += f->messages[k].eta();
        lam_acc_.noalias() += f->messages[k].lam();
    }

    belief.etaRef().noalias() = eta_acc_;
    belief.lamRef().noalias() = lam_acc_;
}

} // namespace gbp