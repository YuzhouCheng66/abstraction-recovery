#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>
#include <string>

#include <Eigen/Dense>

#include "gbp/FactorGraph.h"
#include "gbp/VariableNode.h"
#include "gbp/Factor.h"

// Profiling utilities for bottomUp analysis
void printBottomUpProfile();
void resetBottomUpProfile();

namespace gbp {

// 用于 unordered_map<pair<int,int>, ...>
struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const noexcept {
        // stable-ish hash combine
        return (static_cast<size_t>(p.first) * 1315423911u) ^ static_cast<size_t>(p.second);
    }
};

struct SuperLayer {
    std::shared_ptr<FactorGraph> graph;
    int group_size = 1;

    // base var id -> super id (sid = bid / group_size)
    std::vector<int> node_map;

    // sid -> list of base ids
    std::vector<std::vector<int>> groups;

    // sid -> (bid -> (offset, dofs))
    std::vector<std::unordered_map<int, std::pair<int,int>>> local_idx;

    // sid -> total dofs of the super variable
    std::vector<int> total_dofs;
};

struct AbsLayer {
    std::shared_ptr<FactorGraph> graph;

    // sid -> B_reduced (d_sup x r)
    std::unordered_map<int, Eigen::MatrixXd> Bs;

    // sid -> k (d_sup)
    std::unordered_map<int, Eigen::VectorXd> ks;

    // kept for parity with your Python signature (currently unused in Python too)
    std::unordered_map<int, Eigen::VectorXd> k2s;
};

class HierarchyGBP {
public:
    explicit HierarchyGBP(int group_size);

    // Build super from base
    std::shared_ptr<SuperLayer>
    buildSuperFromBase(const std::shared_ptr<FactorGraph>& base, double eta_damping_super);

    // Update super values only (no new graph/factors)
    void bottomUpUpdateSuper(
        const std::shared_ptr<FactorGraph>& base,
        const std::shared_ptr<SuperLayer>& super);

    // ======================
    // Abs layer (match your Python build_abs_graph / bottom_up_modify_abs_graph)
    // ======================
    std::shared_ptr<AbsLayer>
    buildAbsFromSuper(
        const std::shared_ptr<FactorGraph>& sup_fg,
        int r_reduced = 2,
        double eta_damping_abs = 0.4
    );

    void bottomUpUpdateAbs(
        const std::shared_ptr<FactorGraph>& sup_fg,
        const std::shared_ptr<AbsLayer>& abs,
        int r_reduced = 2,
        double eta_damping_abs = 0.4
    );

// ======================
// Top-down projection (match your Python top_down_modify_* functions)
// ======================

// From super down to base: split super.mu into base vars, update beliefs, and
// adjust adjacent factors' adj_beliefs/messages (Python parity).
void topDownModifyBaseFromSuper(
    const std::shared_ptr<FactorGraph>& base_fg,
    const std::shared_ptr<SuperLayer>& super
);

// From abs down to super: lift abs mu back to super, update beliefs, and
// adjust adjacent factors' adj_beliefs/messages (Python parity).
void topDownModifySuperFromAbs(
    const std::shared_ptr<FactorGraph>& sup_fg,
    const std::shared_ptr<AbsLayer>& abs
);


struct VLayerEntry {
    std::string name;
    std::shared_ptr<FactorGraph> graph;      // convenience pointer to the active graph at this layer
    std::shared_ptr<SuperLayer> super;       // non-null iff this is a super layer
    std::shared_ptr<AbsLayer> abs;           // non-null iff this is an abs layer
};

// Simplified V-cycle (Python VGraph.vloop parity):
// 1) bottom-up: (re)build/modify super & abs layers from the previous layer, then one GBP iteration per layer
// 2) top-down:  one GBP iteration per layer, then project mu downward (abs->super, super->base)
// Notes:
// - layers[0] must be the base layer with a valid `graph` pointer.
// - For super layers, if `super` is null it will be built; otherwise updated in-place.
// - For abs layers, if `abs` is null it will be built; otherwise updated in-place.
void vLoop(
    std::vector<VLayerEntry>& layers,
    int r_reduced = 2,
    double eta_damping = 0.4
);

private:
    int k_; // group size

    std::vector<int> makeOrderNodeMap_(int n_base) const;
};

} // namespace gbp
