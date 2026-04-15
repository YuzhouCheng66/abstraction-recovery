#include "gbp/VariableNode.h"
#include "gbp/Factor.h"
#include <cassert>
#include <cstring>
#include <stdexcept>



namespace gbp {

void VariableNode::updateBeliefImpl_(bool update_mu) {
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
        if (update_mu) {
            mu2 = belief.mu().head<2>();
            mu = mu2;
            markMuCurrent();
        } else {
            mu_valid_ = false;
        }
        return;
    }

    if (dofs == 3) {
        using Vec3 = Eigen::Matrix<double, 3, 1>;
        using Mat3 = Eigen::Matrix<double, 3, 3>;

        const Eigen::Map<const Vec3> prior_eta(prior.etaData());
        const Eigen::Map<const Mat3> prior_lam(prior.lamData());

        Vec3 eta3 = prior_eta;
        Mat3 lam3 = prior_lam;

        for (const auto& aref : adj_factors) {
            const Factor* f = aref.factor;
            const int k = aref.local_idx;
            assert(f != nullptr);
            assert(k >= 0 && k < (int)f->messages.size());
            eta3.noalias() += Eigen::Map<const Vec3>(f->messages[k].etaData());
            lam3.noalias() += Eigen::Map<const Mat3>(f->messages[k].lamData());
        }

        belief.lamRef();
        std::memcpy(belief.etaData(), eta3.data(), 3 * sizeof(double));
        std::memcpy(belief.lamData(), lam3.data(), 9 * sizeof(double));

        if (update_mu) {
            llt3_.compute(lam3);
            if (llt3_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in VariableNode::updateBelief (3D fast path)");
            }
            mu = llt3_.solve(eta3);
            markMuCurrent();
        } else {
            mu_valid_ = false;
        }
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
    if (update_mu) {
        mu = belief.mu();
        markMuCurrent();
    } else {
        mu_valid_ = false;
    }
}

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
    updateBeliefImpl_(true);
}

void VariableNode::updateBeliefNoMu() {
    updateBeliefImpl_(false);
}

void VariableNode::refreshMu() {
    if (mu_valid_) {
        return;
    }

    if (dofs == 2) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;
        const Eigen::Map<const Vec2> eta2(belief.etaData());
        const Eigen::Map<const Mat2> lam2(belief.lamData());
        Eigen::LLT<Mat2> llt;
        llt.compute(lam2);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("LLT failed in VariableNode::refreshMu (2D)");
        }
        mu2 = llt.solve(eta2);
        mu = mu2;
        mu_valid_ = true;
        return;
    }

    if (dofs == 3) {
        using Vec3 = Eigen::Matrix<double, 3, 1>;
        using Mat3 = Eigen::Matrix<double, 3, 3>;
        const Eigen::Map<const Vec3> eta3(belief.etaData());
        const Eigen::Map<const Mat3> lam3(belief.lamData());
        llt3_.compute(lam3);
        if (llt3_.info() != Eigen::Success) {
            throw std::runtime_error("LLT failed in VariableNode::refreshMu (3D)");
        }
        mu = llt3_.solve(eta3);
        mu_valid_ = true;
        return;
    }

    mu = belief.mu();
    mu_valid_ = true;
}

} // namespace gbp
