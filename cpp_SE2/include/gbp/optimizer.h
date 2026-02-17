#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <gbp/factor.h>
#include <gbp/factor_graph.h>
#include <gbp/maths.h>
#include <gbp/message.h>
#include <gbp/multi_factor_graph.h>
#include <gbp/opt_options.h>
#include <gbp/random_generator.h>
#include <gbp/variable.h>

#include <Eigen/Dense>
#include <manif/Rn.h>
#include <manif/SE2.h>
#include <manif/SO2.h>

inline void factor_iteration(int32_t iter, std::map<int32_t, FactorGraph> &factor_graphs,
                             MessageFilter &message_filter, OptOptions &options) {
#pragma omp parallel for
  for (size_t i = 0; i < factor_graphs.size(); ++i) {
    auto it = factor_graphs.begin();
    std::advance(it, i);
    auto &[graph_id, factor_graph] = *it;
    int32_t n_variables = factor_graph.variables_.size();
    // Iterate over the active factors
    for (auto &k : factor_graph.active_set_) {
      auto f = factor_graph.factors_.at(k);
      auto state_vector =
          message_filter.filter(make_state_vector(factor_graphs, f.get()), graph_id);
      auto inbox = message_filter.filter(make_factor_inbox(factor_graphs, f.get()), graph_id);
      f->update_factor(state_vector, options.lin_every_);
      f->send_message(inbox);
    }
  }
}

inline void variable_iteration(std::map<int32_t, FactorGraph> &factor_graphs,
                               MessageFilter &message_filter, OptOptions &options) {

  spdlog::stopwatch sw;
  int32_t N = 0;
  std::map<Key, Mailbox> inboxes;
  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &key_f : factor_graph.active_set_) {
      auto &f = factor_graph.factors_.at(key_f);
      for (auto &[key_v, msg] : f->outbox_) {
        if (inboxes.count(key_v) == 0) {
          inboxes.insert({key_v, {}});
        }
        inboxes.at(key_v).insert({key_f, msg});
        N++;
      }
    }
  }
  spdlog::info("    Make inbox: {}, size: {}", sw, N);
  sw.reset();

#pragma omp parallel for
  for (size_t i = 0; i < inboxes.size(); ++i) {
    auto it = inboxes.begin();
    std::advance(it, i);
    auto &[key, inbox] = *it;
    auto &factor_graph = factor_graphs.at(key.graph_id_);
    auto &v = factor_graph.variables_.at(key);
    inbox = message_filter.filter(inbox, key.graph_id_);
    v->update_belief(inbox);
    if (options.not_ready_.count(key.graph_id_)) {
      v->Lambda_ = Eigen::Matrix3d::Zero();
      for (auto &[_, msg] : v->outbox_) {
        msg.Lambda_ = Eigen::Matrix3d::Zero();
      }
      v->inbox_.clear();
    }
  }
  spdlog::info("    Update Belief: {}", sw);
}

