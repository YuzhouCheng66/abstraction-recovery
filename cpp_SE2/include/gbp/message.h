#pragma once
#include <Eigen/Dense>
#include <any>
#include <map>

#include <gbp/key.h>

struct Message {
  std::any state_;
  Eigen::MatrixXd Lambda_;

  Message(std::any state, const Eigen::MatrixXd &Lambda) : state_(state), Lambda_(Lambda) {}
  template <typename T> std::pair<T, Eigen::MatrixXd> as() const {
    return std::make_pair(std::any_cast<T>(state_), Lambda_);
  }
};

using Mailbox = std::map<Key, Message>;
inline Mailbox merge_mails(const Mailbox &preferred, const Mailbox &other) {
  Mailbox tmp;
  tmp.insert(preferred.begin(), preferred.end());
  tmp.insert(other.begin(), other.end());
  return tmp;
}
