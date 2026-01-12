#include "gbp/FactorGraph.h"
#include <cassert>
#include <cmath>

namespace gbp {

VariableNode* FactorGraph::addVariable(int id, int dofs) {
    if ((int)var_nodes.size() <= id) {
        var_nodes.resize(id + 1);
    }
    var_nodes[id] = std::make_unique<VariableNode>(id, dofs);

    if ((int)var_residual.size() <= id) {
        var_residual.resize(id + 1, 0.0);
    }
    return var_nodes[id].get();
}

Factor* FactorGraph::addFactor(int id, const std::vector<VariableNode*>& vars) {
    factors.push_back(std::make_unique<Factor>(id, vars));
    return factors.back().get();
}

void FactorGraph::connect(Factor* f, VariableNode* v, int local_idx) {
    assert(f && v);
    v->adj_factors.push_back(AdjFactorRef{f, local_idx});
}

void FactorGraph::syncAllFactorAdjBeliefs() {
    for (auto& fptr : factors) {
        fptr->syncAdjBeliefsFromVariables();
    }
}

void FactorGraph::synchronousIteration(bool /*robustify*/) {
    // 1) factor messages
    for (auto& fptr : factors) {
        if (!fptr->active) continue;
        fptr->computeMessages(eta_damping);
    }

    // 2) variable beliefs
    for (auto& vptr : var_nodes) {
        if (!vptr) continue;
        vptr->updateBelief();
    }

    // 3) variables send beliefs to factors (maintain adj_beliefs)
    syncAllFactorAdjBeliefs();
}

void FactorGraph::residualIterationVarHeap(int max_updates) {
    const int n_vars = (int)var_nodes.size();
    if (max_updates < 0) max_updates = n_vars;

    // init heap once (like Python: if len(self.var_heap) == 0)
    if (var_heap.empty()) {
        for (int vid = 0; vid < n_vars; ++vid) {
            if (!var_nodes[vid]) continue;
            var_residual[vid] = 0.0;
            var_heap.push(HeapEntry{-0.0, vid});
        }
    }

    int n_updates = 0;

    while (!var_heap.empty() && n_updates < max_updates) {
        HeapEntry e = var_heap.top();
        var_heap.pop();

        const int vid = e.varID;
        VariableNode* v = var_nodes[vid].get();
        if (!v || !v->active) continue;

        const double r_v = -e.neg_r;
        const double cur_r = var_residual[vid];

        // stale entry check (parity with Python)
        if (std::abs(r_v - cur_r) > 1e-12) continue;
        if (r_v < residual_eps) break;

        // diffuse through neighbors
        for (const auto& aref : v->adj_factors) {
            Factor* f = aref.factor;
            if (!f || !f->active) continue;

            // Update factor adj beliefs before message update (important for correctness)
            // because factor.computeMessages uses adj_beliefs - messages.
            f->syncAdjBeliefsFromVariables();
            f->computeMessages(eta_damping);

            // Update all vars connected to this factor
            for (auto* u : f->adj_var_nodes) {
                if (!u || !u->active) continue;

                Eigen::VectorXd old_eta = u->belief.eta();
                u->updateBelief();
                Eigen::VectorXd new_eta = u->belief.eta();

                // Update factor-side beliefs after u updated
                // (we do a full sync at the end of outer iteration for simplicity,
                //  but local sync keeps things closer to Python behavior)
                // We'll do local sync: update the entry for u inside this factor
                // by calling f->syncAdjBeliefsFromVariables() later.
                // For now, compute residual:
                double est_r_u = (new_eta - old_eta).norm();
                int uid = u->variableID;
                if (uid >= 0 && uid < (int)var_residual.size()) {
                    double old_r_u = var_residual[uid];
                    if (est_r_u > old_r_u + 1e-12) {
                        var_residual[uid] = est_r_u;
                        var_heap.push(HeapEntry{-est_r_u, uid});
                    }
                }
            }

            // sync beliefs for this factor after updates
            f->syncAdjBeliefsFromVariables();
        }

        // clear residual for v after diffusion
        var_residual[vid] = 0.0;
        var_heap.push(HeapEntry{-0.0, vid}); // keep it in heap for future rounds
        n_updates += 1;
    }
}

} // namespace gbp




