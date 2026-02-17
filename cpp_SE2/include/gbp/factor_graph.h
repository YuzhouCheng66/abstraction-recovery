#pragma once

#include <map>
#include <memory>
#include <set>

#include <manif/SE2.h>
#include <manif/SO2.h>

#include <gbp/between.h>
#include <gbp/factor.h>
#include <gbp/factor_type.h>
#include <gbp/factors/between_SE2.h>
#include <gbp/factors/prior_SE2.h>
#include <gbp/factors/range_bearing_SE2.h>
#include <gbp/key.h>
#include <gbp/message.h>
#include <gbp/opt_options.h>
#include <gbp/prior.h>
#include <gbp/range_bearing.h>
#include <gbp/variable.h>

struct FactorGraph {
  int32_t id_;
  std::map<Key, std::shared_ptr<Factor>> factors_;
  std::map<Key, std::shared_ptr<Variable>> variables_;
  std::set<Key> active_set_;

  FactorGraph(int32_t id) : id_(id) {}

  std::vector<manif::SE2d> make_state_vector(const Factor *f) {
    std::vector<manif::SE2d> state_vector;
    for (auto key : f->neighbors_) {
      state_vector.push_back(this->variables_.at(key)->state_);
    }
    return state_vector;
  };

  std::map<Key, Message> make_factor_inbox(const Factor *f) {
    std::map<Key, Message> inbox;
    for (auto key : f->neighbors_) {
      auto &v = this->variables_.at(key);
      inbox.insert({key, Message(v->state_, v->Lambda_)});
    }
    return inbox;
  };

  std::map<Key, Message> make_variable_inbox(const Variable *v) {
    std::map<Key, Message> inbox;
    for (auto &[key, factor] : this->factors_) {
      if (factor->outbox_.count(v->key_) == 0) {
        continue;
      }
      inbox.insert({key, factor->outbox_.at(v->key_)});
    }
    return inbox;
  };

  Variable *add_variable(size_t variable_id, const manif::SE2d &initial) {
    Key key = Key(this->id_, variable_id);
    return this->add_variable(key, initial);
  }

  Variable *add_variable(Key key, const manif::SE2d &initial) {
    this->variables_.insert_or_assign(key, std::make_shared<Variable>(key, initial));
    return this->variables_.at(key).get();
  }

  Factor *add_prior(Key v_key, const manif::SE2d &p, const Eigen::Matrix3d &Lam) {
    Key f_key = this->avaiable_factor_key();
    this->factors_.insert({f_key, std::make_shared<PriorSE2>(PriorSE2(f_key, {v_key}, p, Lam))});

    // Add variable automatically if it does not exit
    if (this->variables_.count(v_key) == 0) {
      this->add_variable(v_key, p);
    }
    active_set_.insert(f_key);
    return this->factors_.at(f_key).get();
  }

  Factor *add_between(Key v_key1, Key v_key2, const manif::SE2d &m_between,
                      const Eigen::Matrix3d &Lam) {
    Key f_key = this->avaiable_factor_key();
    this->factors_.insert(
        {f_key, std::make_shared<BetweenSE2>(BetweenSE2(f_key, {v_key1, v_key2}, m_between, Lam))});
    active_set_.insert(f_key);
    return this->factors_.at(f_key).get();
  }

  Factor *add_range_bearing(Key v_key1, Key v_key2,
                            const std::pair<double, manif::SO2d> &m_range_bearing,
                            const Eigen::Matrix2d &Lam) {
    Key f_key = this->avaiable_factor_key();
    active_set_.insert(f_key);
    this->factors_.insert({f_key, std::make_shared<RangeBearingSE2>(RangeBearingSE2(
                                      f_key, {v_key1, v_key2}, m_range_bearing, Lam))});
    return this->factors_.at(f_key).get();
  }

  void filter_active_set(const OptOptions &options, bool full_reset = false) {
    std::vector<Key> iterate_over = this->active_keys(full_reset);
    int32_t n_variables = this->variables_.size();
    this->active_set_.clear();
    for (auto &k : iterate_over) {
      auto f = this->factors_.at(k);
      f->active_ = false;
      if (options.not_ready_.count(this->id_)) {
        if (options.window_ == 0 || f->iter_ < options.window_ * options.iter_before_motion_ ||
            f->is_prior_) {
          f->active_ = true;
          this->active_set_.insert(k);
        }
        continue;
      }
      if (options.window_ == 0 || f->min_variable_id_ > n_variables - options.window_) {
        f->active_ = true;
        this->active_set_.insert(k);
      }
    }
  }

  std::vector<Key> active_keys(bool full_reset) const {
    std::vector<Key> keys;
    if (full_reset) {
      for (auto &[k, _] : this->factors_) {
        keys.push_back(k);
      }
      return keys;
    }
    return std::vector<Key>(active_set_.begin(), active_set_.end());
  }

  Key avaiable_factor_key() const { return Key(this->id_, this->factors_.size()); }
};
