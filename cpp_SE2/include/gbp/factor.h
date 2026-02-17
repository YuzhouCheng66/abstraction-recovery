#pragma once
#include <iostream>
#include <map>
#include <math.h>
#include <vector>

#include <Eigen/Dense>
#include <functional>
#include <gbp/factor_type.h>
#include <gbp/key.h>
#include <gbp/maths.h>
#include <gbp/message.h>
#include <manif/SE2.h>

enum class Loss { DCS, HUBER, NONE };

struct Factor {
  Key key_;
  Eigen::MatrixXd factor_eta_;
  Eigen::MatrixXd factor_Lam_;

  std::vector<Key> neighbors_;
  std::map<Key, manif::SE2d> X0_, X0_buffer_;
  std::vector<manif::SE2d> X0_vec_;
  std::map<Key, std::pair<Eigen::VectorXd, Eigen::MatrixXd>> last_outbox_;

  Mailbox inbox_, outbox_;
  bool is_prior_, active_;
  double damping_, Phi_, scale_;
  Loss loss_;
  FactorType type_;
  int32_t iter_, min_variable_id_;

  Factor(Key key, const std::vector<Key> &neighbors, bool is_prior, FactorType type)
      : key_(key), neighbors_(neighbors), is_prior_(is_prior), type_(type), damping_(0.2),
        Phi_(10.f), loss_(Loss::NONE), scale_(1.f), iter_(0), active_(false) {
    min_variable_id_ = std::numeric_limits<int32_t>::max();
    for (auto &n : neighbors_) {
      if (n.graph_id_ == key.graph_id_) {
        min_variable_id_ = std::min(min_variable_id_, n.node_id_);
      }
    }
  }

  bool is_robust() const { return this->loss_ != Loss::NONE; }

  bool receive_msg_and_check_relin(const std::map<Key, std::any> &X0, int32_t lin_freq) {
    // Always store the information given
    std::map<Key, manif::SE2d> tmp;
    for (auto &[k, v] : X0) {
      tmp.insert({k, std::any_cast<manif::SE2d>(v)});
    }
    // tmp has the new information, X0_buffer_ has the old information
    tmp.insert(this->X0_buffer_.begin(), this->X0_buffer_.end());
    this->X0_buffer_ = tmp;

    this->iter_++;
    if (this->X0_buffer_.size() != this->neighbors_.size()) {
      return false;
    }
    if (this->iter_ % lin_freq != 0) {
      return false;
    }
    // Ready to relinearize. Copy the buffer to the main X0, X0_vec_
    this->X0_ = this->X0_buffer_;
    this->X0_vec_.clear();
    for (auto n : this->neighbors_) {
      this->X0_vec_.push_back(this->X0_.at(n));
    }
    return true;
  }

  void set_factor_eta_Lam(const Eigen::VectorXd &h, const Eigen::MatrixXd &J,
                          const Eigen::MatrixXd &measurement_Lam) {
    this->scale_ = 1.f;
    if (this->is_robust()) {
      double error = h.transpose() * measurement_Lam * h;
      this->scale_ = this->robust_kernel(error);
    }
    // 1e-6 for numerical stability
    Eigen::MatrixXd Lam = this->scale_ * J.transpose() * measurement_Lam * J +
                          Eigen::MatrixXd::Identity(J.cols(), J.cols()) * 1e-6;
    Eigen::VectorXd eta = this->scale_ * J.transpose() * measurement_Lam * -h;
    // Clear out the outbox as linearlization point changed
    this->last_outbox_.clear();

    this->factor_eta_ = eta;
    this->factor_Lam_ = Lam;
  }

  void update_one(int i, Eigen::VectorXd &eta_c, Eigen::MatrixXd &Lam_c) {
    const int DoF = manif::SE2d::DoF;
    Key k = this->neighbors_.at(i);
    auto [Xj, Lj] = this->inbox_.at(k).as<manif::SE2d>();
    manif::SE2Tangentd tau = Xj - this->X0_.at(k);
    Eigen::Matrix3d Lam = tau.rjac().transpose() * Lj * tau.rjac();
    Eigen::Vector3d eta = Lam * tau.coeffs();
    eta_c.segment<DoF>(i * DoF) += eta;
    Lam_c.block<DoF, DoF>(i * DoF, i * DoF) += Lam;
  }

  void downdate_one(int i, Eigen::VectorXd &eta_c, Eigen::MatrixXd &Lam_c) {
    const int DoF = manif::SE2d::DoF;
    Key k = this->neighbors_.at(i);
    auto [Xj, Lj] = this->inbox_.at(k).as<manif::SE2d>();
    manif::SE2Tangentd tau = Xj - this->X0_.at(k);
    Eigen::Matrix3d Lam = tau.rjac().transpose() * Lj * tau.rjac();
    Eigen::Vector3d eta = Lam * tau.coeffs();
    eta_c.segment<DoF>(i * DoF) -= eta;
    Lam_c.block<DoF, DoF>(i * DoF, i * DoF) -= Lam;
  }

  void compute_outgoing_msg(int i, const Eigen::VectorXd &eta_all, const Eigen::MatrixXd &Lam_all) {
    Key k1 = this->neighbors_.at(i);
    Eigen::VectorXd eta_c = eta_all;
    Eigen::MatrixXd Lam_c = Lam_all;
    downdate_one(i, eta_c, Lam_c);

    if (!this->is_prior_ && Lam_c.isApprox(this->factor_Lam_)) {
      return;
    }
    auto [eta, Lam] = marginalize(eta_c, Lam_c, i * 3, (i + 1) * 3 - 1);
    if (!this->is_prior_ && this->last_outbox_.count(k1)) {
      auto [last_eta, last_Lam] = this->last_outbox_.at(k1);
      eta = (1.f - this->damping_) * eta + this->damping_ * last_eta;
      Lam = (1.f - this->damping_) * Lam + this->damping_ * last_Lam;
    }
    this->last_outbox_.insert_or_assign(k1, std::make_pair(eta, Lam));

    manif::SE2Tangentd tau(Lam.colPivHouseholderQr().solve(eta));
    manif::SE2d X = this->X0_.at(k1) + tau;
    Eigen::Matrix3d L = tau.rjacinv().transpose() * Lam * tau.rjacinv();
    this->outbox_.insert({k1, Message(X, L)});
  }

  virtual bool update_factor(const std::map<Key, std::any> &X0, int32_t lin_freq) = 0;
  virtual std::map<Key, Message> send_message(const std::map<Key, Message> &inbox) = 0;
  virtual double error() = 0;

  double robust_kernel(double error) const {
    double scale = 1.f;
    if (this->loss_ == Loss::DCS) {
      if (error > this->Phi_) {
        double s = 2.f * this->Phi_ / (this->Phi_ + error);
        scale = s * s;
      }
    }
    if (this->loss_ == Loss::HUBER) {
      double e = std::sqrt(error);
      if (e > this->Phi_) {
        scale = this->Phi_ / e;
      }
    }
    return scale;
  }
};