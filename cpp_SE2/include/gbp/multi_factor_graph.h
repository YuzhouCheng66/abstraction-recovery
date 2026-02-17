#pragma once
#include <map>
#include <set>
#include <vector>

#include <gbp/factor.h>
#include <gbp/factor_graph.h>
#include <gbp/key.h>
#include <gbp/message.h>
#include <gbp/random_generator.h>
#include <gbp/variable.h>

inline std::map<Key, std::any>
make_state_vector(const std::map<int32_t, FactorGraph> &factor_graphs, Factor *f) {
  std::map<Key, std::any> states;
  for (Key key : f->neighbors_) {
    auto &v = factor_graphs.at(key.graph_id_).variables_.at(key);
    states.insert({key, v->state_});
  }
  return states;
}
inline Mailbox make_factor_inbox(const std::map<int32_t, FactorGraph> &factor_graphs,
                                 const Factor *f) {
  Mailbox inbox;
  for (auto key : f->neighbors_) {
    auto &v = factor_graphs.at(key.graph_id_).variables_.at(key);
    if (v->outbox_.count(f->key_)) {
      inbox.insert({key, v->outbox_.at(f->key_)});
    } else {
      inbox.insert({key, Message(v->state_, v->Lambda_)});
    }
  }
  return inbox;
}

inline Mailbox make_variable_inbox(const std::map<int32_t, FactorGraph> &factor_graphs,
                                   const Variable *v) {
  Mailbox inbox;
  // TODO: This is inefficient but oh well
  for (auto &[_, factor_graph] : factor_graphs) {
    for (auto &[key, factor] : factor_graph.factors_) {
      if (factor->outbox_.count(v->key_) == 0) {
        continue;
      }
      inbox.insert({key, factor->outbox_.at(v->key_)});
    }
  }
  return inbox;
}

enum class FilterType { RANDOM, CLOSEST_RANDOM };
struct MessageFilter {
  FilterType type_;
  std::map<int32_t, std::set<int32_t>> samples_;
  double dropout_percentage_;
  std::atomic_int32_t n_msgs_;

  MessageFilter(FilterType type) : type_(type), dropout_percentage_(-1), n_msgs_(0) {}

  void resample(size_t n_samples, const std::map<int32_t, FactorGraph> &factor_graphs,
                const std::map<int32_t, std::vector<manif::SE2d>> &groundtruth,
                double max_dist = -1) {
    if (n_samples == 0) {
      std::set<int32_t> graph_ids;
      for (auto &[id, _] : groundtruth) {
        graph_ids.insert(id);
      }
      for (auto &[id, _] : groundtruth) {
        samples_.insert_or_assign(id, graph_ids);
      }
      return;
    }
    if (n_samples > groundtruth.size() - 1) {
      n_samples = groundtruth.size() - 1;
    }

    auto dist_matrix = inverse_distance2_matrix(groundtruth, max_dist);
    this->samples_.clear();
    for (auto &[i, _] : groundtruth) {
      std::vector<double> cul;
      double acc = 0.f;
      std::vector<int32_t> gt_idx;
      for (auto &[j, _] : groundtruth) {
        acc += dist_matrix.at(i).at(j);
        cul.push_back(acc);
        gt_idx.push_back(j);
      }
      this->samples_.insert({i, {}});
      while (this->samples_.at(i).size() < n_samples) {
        double needle = RGen::uniform01(RGen::Type::SAMPLE) * acc;
        for (size_t j = 0; j < groundtruth.size(); ++j) {
          if (needle <= cul.at(j)) {
            // Use insert to make sure we select n_samples.
            this->samples_.at(i).insert(gt_idx.at(j));
            break;
          }
        }
      }
    }
  }

  Mailbox filter(const Mailbox &mailbox, int32_t graph_id) {
    Mailbox filtered;
    for (auto &[k, v] : mailbox) {
      if (k.graph_id_ == graph_id || this->samples_.at(graph_id).count(k.graph_id_)) {
        // Only drop out external communication
        if (k.graph_id_ != graph_id && dropout_percentage_ > 0 &&
            RGen::uniform01(RGen::Type::SAMPLE) < dropout_percentage_) {
          continue;
        }
        filtered.insert({k, v});
      }
    }
    return filtered;
  }

  std::map<Key, std::any> filter(const std::map<Key, std::any> &states, int32_t graph_id) {
    std::map<Key, std::any> filtered;
    for (auto &[k, v] : states) {
      if (k.graph_id_ == graph_id || this->samples_.at(graph_id).count(k.graph_id_)) {
        // Only drop out external communication
        if (k.graph_id_ != graph_id && dropout_percentage_ > 0 &&
            RGen::uniform01(RGen::Type::SAMPLE) < dropout_percentage_) {
          continue;
        }
        if (k.graph_id_ != graph_id) {
          this->n_msgs_++;
        }
        filtered.insert({k, v});
      }
    }
    return filtered;
  }

  std::map<int32_t, std::map<int32_t, double>>
  inverse_distance2_matrix(const std::map<int32_t, std::vector<manif::SE2d>> &poses,
                           double max_dist) {
    std::map<int32_t, std::map<int32_t, double>> dist_matrix;
    for (auto &[i, poses_i] : poses) {
      for (auto &[j, poses_j] : poses) {
        if (i == j) {
          dist_matrix[i][j] = 0;
          continue;
        }
        manif::SE2d pi = poses_i.back();
        manif::SE2d pj = poses_j.back();
        double d =
            (pi.translation() - pj.translation()).norm() + RGen::normal(RGen::Type::SAMPLE) * 0.1;
        dist_matrix[i][j] = 1 / (d * d);
        if (max_dist > 0 && d > max_dist) {
          // If the maximum distance is set, and we exceed it, we set it to zero to make sure that
          // we never sample this point (i.e. it's infinitely far away)
          dist_matrix[i][j] = 0;
        }
      }
    }
    return dist_matrix;
  }
};
