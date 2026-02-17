// src/test.cpp
#include <Eigen/Dense>
#include <manif/SE2.h>

#include <gbp/factor_graph.h>
#include <gbp/multi_factor_graph.h>
#include <gbp/opt_options.h>
#include <gbp/optimizer.h>

#include <spdlog/spdlog.h>

#include <iostream>
#include <map>
#include <vector>

static inline double theta_from_se2(const manif::SE2d &X) {
  // robust way: use SO2 log -> 1D tangent
  return X.angle();
}


static inline void print_states(const FactorGraph &fg, const std::string &tag) {
  std::cout << tag << "\n";
  for (auto &[k, vptr] : fg.variables_) {
    const auto &X = vptr->state_;
    auto t = X.translation();
    double th = theta_from_se2(X);
    std::cout << "  v(" << k.node_id_ << "): x=" << t.x() << " y=" << t.y()
              << " th=" << th << "\n";
  }
}

int main() {
  spdlog::set_level(spdlog::level::info);

  // ---------- 1) Build the minimal graph (single graph_id = 0) ----------
  std::map<int32_t, FactorGraph> factor_graphs;
  factor_graphs.emplace(0, FactorGraph(0));
  FactorGraph &fg = factor_graphs.at(0);

  // Variables: same initial guesses as your python snippet
  Key k1(0, 0), k2(0, 1), k3(0, 2);

  fg.add_variable(k1, manif::SE2d(0.0, 0.0, 0.0));
  fg.add_variable(k2, manif::SE2d(0.98, 0.2, 0.01));
  fg.add_variable(k3, manif::SE2d(1.0, 1.11, M_PI / 2.0 - 0.01));

  // Initial Lambda on variables: diag([1,1,100]) * 1e-8
  Eigen::Matrix3d Lam0 = Eigen::Matrix3d::Zero();
  Lam0(0, 0) = 1e-8;
  Lam0(1, 1) = 1e-8;
  Lam0(2, 2) = 100.0 * 1e-8;

  fg.variables_.at(k1)->Lambda_ = Lam0;
  fg.variables_.at(k2)->Lambda_ = Lam0;
  fg.variables_.at(k3)->Lambda_ = Lam0;

  // Factors:
  // Prior: measurement_lambda = 1e6 * diag([1,1,100])
  Eigen::Matrix3d priorLam = Eigen::Matrix3d::Zero();
  priorLam(0, 0) = 1e6;
  priorLam(1, 1) = 1e6;
  priorLam(2, 2) = 1e6 * 100.0;
  fg.add_prior(k1, manif::SE2d(0.0, 0.0, 0.0), priorLam);

  // Between: measurement_lambda = diag([1,1,100])
  Eigen::Matrix3d betweenLam = Eigen::Matrix3d::Zero();
  betweenLam(0, 0) = 1.0;
  betweenLam(1, 1) = 1.0;
  betweenLam(2, 2) = 100.0;

  // Use numpy(seed=0) results directly:
  // meas2 = [1.17640523, 0.04001572, 0.00978738]
  // meas3 = [0.22408932, 1.18675580, 1.56102355]
  fg.add_between(k1, k2, manif::SE2d(1.17640523, 0.04001572, 0.00978738), betweenLam);
  fg.add_between(k2, k3, manif::SE2d(0.22408932, 1.18675580, 1.56102355), betweenLam);

  // ---------- 2) Options / Message filter ----------
  OptOptions options;
  options.window_ = 0;       // no sliding window, keep all factors active
  options.lin_every_ = 1;    // relinearize every iteration (needed for first iteration to send msgs)
  options.is_robust_ = false;

  // Make sure active_set_ contains all factors (not just newly added ones)
  fg.filter_active_set(options, /*full_reset=*/true);

  MessageFilter filter(FilterType::RANDOM);

  // resample(n_samples=0) makes samples_[0] = {0} (self graph only),
  // which is required because filter() uses samples_.at(graph_id).
  std::map<int32_t, std::vector<manif::SE2d>> groundtruth;
  groundtruth[0] = {
      fg.variables_.at(k1)->state_,
      fg.variables_.at(k2)->state_,
      fg.variables_.at(k3)->state_,
  };
  filter.resample(/*n_samples=*/0, factor_graphs, groundtruth);

  // ---------- 3) Run GBP iterations ----------
  print_states(fg, "Initial:");

  const int iters = 20;
  for (int it = 1; it <= iters; ++it) {
    // keep factors active (for safety)
    fg.filter_active_set(options, /*full_reset=*/true);

    factor_iteration(it, factor_graphs, filter, options);
    variable_iteration(factor_graphs, filter, options);

    if (it == 1 || it % 5 == 0) {
      print_states(fg, "After iter " + std::to_string(it) + ":");
    }
  }

  return 0;
}
