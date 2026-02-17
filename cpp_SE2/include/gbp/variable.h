#pragma once
#include <gbp/key.h>
#include <gbp/message.h>
#include <iostream>
#include <manif/SE2.h>
#include <map>

struct Variable {
  Key key_;
  manif::SE2d state_;
  Eigen::MatrixXd Lambda_;
  Mailbox inbox_, outbox_;
  bool merge_inbox;

  Variable(Key key, const manif::SE2d &state)
      : key_(key), state_(state), Lambda_(Eigen::Matrix3d::Zero()), merge_inbox(true) {}

  Mailbox update_belief(const Mailbox &inbox) {
    this->inbox_ = this->merge_inbox ? merge_mails(inbox, this->inbox_) : inbox;
    this->outbox_.clear();
    if (this->inbox_.size() == 0) {
      return {};
    }
    Eigen::Matrix3d Lambda_all = Eigen::Matrix3d::Zero();
    Eigen::Vector3d eta_all = Eigen::Vector3d::Zero();
    for (auto [_, msg] : this->inbox_) {
      auto [X_msg, Lam_msg] = msg.as<manif::SE2d>();
      manif::SE2Tangentd tau = X_msg - this->state_;
      Eigen::Matrix3d L = tau.rjac().transpose() * Lam_msg * tau.rjac();
      eta_all += L * tau.coeffs();
      Lambda_all += L;
    }

    if (Lambda_all.isApprox(Eigen::Matrix3d::Zero())) {
      return {};
    }

    // Compute the outgoing message
    for (auto [k, msg] : this->inbox_) {
      auto [X_msg, Lam_msg] = msg.as<manif::SE2d>();
      manif::SE2Tangentd tau_msg = X_msg - this->state_;
      Eigen::Matrix3d L_msg_a = tau_msg.rjac().transpose() * Lam_msg * tau_msg.rjac();
      Eigen::Vector3d eta_msg_a = L_msg_a * tau_msg.coeffs();

      Eigen::Vector3d eta_a = eta_all - eta_msg_a;
      Eigen::Matrix3d L_a = Lambda_all - L_msg_a;
      manif::SE2Tangentd tau_a(L_a.colPivHouseholderQr().solve(eta_a));
      if (tau_a.coeffs().allFinite()) {
        this->outbox_.insert({k, Message(this->state_ + tau_a,
                                         tau_a.rjacinv().transpose() * L_a * tau_a.rjacinv())});
      } else {
        this->outbox_.insert({k, Message(this->state_, Eigen::Matrix3d::Zero())});
      }
    }

    // Update belief
    manif::SE2Tangentd tau(Lambda_all.colPivHouseholderQr().solve(eta_all));
    this->state_ += tau;
    this->Lambda_ = tau.rjacinv().transpose() * Lambda_all * tau.rjacinv();
    //}
    return this->outbox_;
  }
};
