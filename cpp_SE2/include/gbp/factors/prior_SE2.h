#pragma once
#include "gbp/prior.h"
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include <gbp/factor.h>
#include <gbp/factor_type.h>
#include <gbp/key.h>
#include <gbp/maths.h>
#include <gbp/message.h>
#include <manif/SE2.h>

struct PriorSE2 : public Factor {
  manif::SE2d measurement_;
  Eigen::Matrix<double, 3, 3> measurement_Lam_;

  PriorSE2(Key key, const std::vector<Key> &neighbors, const manif::SE2d &measurement,
           const Eigen::Matrix<double, 3, 3> &measurement_Lam)
      : Factor(key, neighbors, true, FactorType::PRIOR2D), measurement_(measurement),
        measurement_Lam_(measurement_Lam) {}

  bool update_factor(const std::map<Key, std::any> &X0, int32_t lin_freq) override {
    bool relinearize = receive_msg_and_check_relin(X0, lin_freq);
    if (!relinearize) {
      return false;
    }
    auto [h, J] = residual_prior(this->X0_vec_.at(0), this->measurement_);
    set_factor_eta_Lam(h, J, this->measurement_Lam_);
    return true;
  }

  std::map<Key, Message> send_message(const std::map<Key, Message> &inbox) override {
    this->inbox_ = merge_mails(inbox, this->inbox_);
    this->outbox_.clear();
    if (this->X0_.size() != this->neighbors_.size()) {
      return this->outbox_;
    }
    Eigen::VectorXd eta_a = this->factor_eta_;
    Eigen::MatrixXd Lam_a = this->factor_Lam_;
    // Aggregate the information from all neighbors
    for (size_t i = 0; i < this->neighbors_.size(); ++i) {
      Key k = this->neighbors_.at(i);
      update_one(i, eta_a, Lam_a);
    }
    // Compute outgoing messages
    for (size_t i = 0; i < this->neighbors_.size(); ++i) {
      compute_outgoing_msg(i, eta_a, Lam_a);
    }
    return this->outbox_;
  }

  double error() override {
    if (this->X0_.size() != this->neighbors_.size()) {
      return 0.f;
    }
    auto [h, J] = residual_prior(this->X0_vec_.at(0), this->measurement_);
    return h.transpose() * this->measurement_Lam_ * h;
  }
};
