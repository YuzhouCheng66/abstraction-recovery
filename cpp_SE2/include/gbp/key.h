#pragma once
#include <stdint.h>
struct Key {
  int32_t graph_id_;
  int32_t node_id_;
  Key(int32_t graph_id, int32_t node_id)
      : graph_id_(graph_id), node_id_(node_id) {}

  bool operator==(const Key &k) const {
    return this->graph_id_ == k.graph_id_ && this->node_id_ == k.node_id_;
  }

  bool operator<(const Key &k) const {
    return this->graph_id_ < k.graph_id_ ||
           (this->graph_id_ == k.graph_id_ && this->node_id_ < k.node_id_);
  }
};
