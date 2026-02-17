#pragma once
#include <set>

#include <gbp/key.h>

struct OptOptions {
  int32_t window_;
  int32_t lin_every_;
  int32_t iter_before_motion_;
  int32_t n_samples_;
  int32_t sample_every_;

  double damping_;
  double Phi_;
  bool is_robust_;
  double comm_dist_;
  bool add_prior_noise_;
  int32_t prior_noise_level_;
  int32_t sensor_noise_level_;
  int32_t n_obs_;
  std::set<int32_t> not_ready_;
  OptOptions()
      : window_(10), lin_every_(5), iter_before_motion_(20), n_samples_(1), sample_every_(1),
        damping_(0.2), Phi_(10.f), is_robust_(true), comm_dist_(-1), add_prior_noise_(false),
        prior_noise_level_(0), sensor_noise_level_(0), n_obs_(-1) {}
};
