#include "gbp/FactorGraph.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>

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

Factor* FactorGraph::addFactor(
    int id,
    const std::vector<VariableNode*>& vars,
    const std::vector<Eigen::VectorXd>& z,
    const std::vector<Eigen::MatrixXd>& measurement_lambda,
    std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas_fn,
    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)> jac_fn
) {
    factors.push_back(std::make_unique<Factor>(
        id, vars, z, measurement_lambda, std::move(meas_fn), std::move(jac_fn)
    ));
    return factors.back().get();
}


void FactorGraph::connect(Factor* f, VariableNode* v, int local_idx) {
    assert(f && v);
    v->adj_factors.push_back(AdjFactorRef{f, local_idx});
}

void FactorGraph::syncAllFactorAdjBeliefs() {
    #pragma omp parallel for
    for (int i = 0; i < (int)factors.size(); ++i) {
        auto& fptr = factors[i];
        fptr->syncAdjBeliefsFromVariables();
    }
}

void FactorGraph::synchronousIteration(bool /*robustify*/) {
    // --- Profiling: measure each step ---
    static double t_msg = 0, t_belief = 0, t_sync = 0;
    static int t_count = 0;
    // #ifdef _OPENMP
    // printf("[OpenMP] max threads: %d\n", omp_get_max_threads());
    // #endif
    auto t0 = std::chrono::high_resolution_clock::now();
    // 1) factor messages (OpenMP并行)
    #pragma omp parallel for
    for (int i = 0; i < (int)factors.size(); ++i) {
        auto& fptr = factors[i];
        if (!fptr->active) continue;
        fptr->computeMessages(eta_damping);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    t_msg += std::chrono::duration<double>(t1 - t0).count();
    // 2) variable beliefs (OpenMP并行)
    #pragma omp parallel for
    for (int i = 0; i < (int)var_nodes.size(); ++i) {
        auto& vptr = var_nodes[i];
        if (!vptr) continue;
        vptr->updateBelief();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    t_belief += std::chrono::duration<double>(t2 - t1).count();
    // 3) variables send beliefs to factors (maintain adj_beliefs)
    syncAllFactorAdjBeliefs();
    auto t3 = std::chrono::high_resolution_clock::now();
    t_sync += std::chrono::duration<double>(t3 - t2).count();
    ++t_count;
    // Only print at the end of all iterations (called from test_large_scale_slam)
    if (t_count == 100) {
        printf("[Profiling] Avg computeMessages: %.4f ms, updateBelief: %.4f ms, syncAllFactorAdjBeliefs: %.4f ms\n",
            1000.0 * t_msg / t_count, 1000.0 * t_belief / t_count, 1000.0 * t_sync / t_count);
    }
}


void FactorGraph::residualIterationVarHeap(int max_updates) {
    constexpr double eps = 1e-12;

    // Python len(self.var_nodes): actual variable list length
    // Here var_nodes may have holes; count non-null as Python-like size
    int n_vars_list = 0;
    for (auto& vptr : var_nodes) if (vptr) ++n_vars_list;
    if (n_vars_list == 0) return;

    // Ensure var_residual sized
    if ((int)var_residual.size() < (int)var_nodes.size()) {
        var_residual.resize(var_nodes.size(), 0.0);
    }

    // init heap once (Python: if len(self.var_heap) == 0)
    if (var_heap.empty()) {
        for (int vid = 0; vid < (int)var_nodes.size(); ++vid) {
            if (!var_nodes[vid]) continue;
            var_residual[vid] = 0.0;
            // Python pushes (-0, vid, v) into min-heap.
            // Our max-heap stores r=0, vid; tie-break makes smaller vid pop first if all r equal.
            var_heap.push(HeapEntry{0.0, vid});
        }
    }

    int n_updates = 0;

    while (!var_heap.empty()) {
        // Python loop guard: n_updates < min(len(heap), len(var_nodes))
        // len(heap) is dynamic; re-evaluate each iteration
        const int heap_cap = std::min((int)var_heap.size(), n_vars_list);

        // If caller provided max_updates >= 0, cap further; otherwise exact Python parity
        const int cap = (max_updates < 0) ? heap_cap : std::min(max_updates, heap_cap);

        if (n_updates >= cap) break;

        HeapEntry e = var_heap.top();
        var_heap.pop();

        const int vid = e.varID;
        VariableNode* v = (vid >= 0 && vid < (int)var_nodes.size()) ? var_nodes[vid].get() : nullptr;
        if (!v) continue;
        if (!v->active) continue; // keep if your Python has active flag; else you can remove

        const double r_v  = e.r;
        const double cur_r = var_residual[vid];

        // stale entry check (Python: only accept if equal to current record)
        if (std::abs(r_v - cur_r) > eps) continue;

        // NOTE: do NOT early-break on residual_eps; Python does not have this rule.
        // if (r_v < residual_eps) break;  // removed for parity

        // diffuse through neighbors
        for (const auto& aref : v->adj_factors) {
            Factor* f = aref.factor;
            if (!f || !f->active) continue;

            // If factor uses cached adj beliefs, sync before computing messages
            f->syncAdjBeliefsFromVariables();
            f->computeMessages(eta_damping);

            // Update all vars connected to this factor
            for (auto* u : f->adj_var_nodes) {
                if (!u || !u->active) continue;

                const int uid = u->variableID;
                if (uid < 0 || uid >= (int)var_residual.size()) continue;

                Eigen::VectorXd old_eta = u->belief.eta();
                u->updateBelief();
                Eigen::VectorXd new_eta = u->belief.eta();

                const double est_r_u = (new_eta - old_eta).norm();
                const double old_r_u = var_residual[uid];

                // Python: only update if new residual is larger
                if (est_r_u > old_r_u + eps) {
                    var_residual[uid] = est_r_u;
                    var_heap.push(HeapEntry{est_r_u, uid});
                }
            }

            // optional: keep factor cache consistent after updates
            f->syncAdjBeliefsFromVariables();
        }

        // clear residual for v after diffusion (Python does this) -- and DO NOT push v back
        var_residual[vid] = 0.0;
        n_updates += 1;
    }
}

gbp::FactorGraph::JointInfResult gbp::FactorGraph::jointDistributionInfSparse() const {
    // ---- compute total dim + max varID ----
    int total = 0;
    int max_id = -1;
    for (const auto& v : var_nodes) {
        total += v->dofs;
        max_id = std::max(max_id, v->variableID);
    }

    JointInfResult out;
    out.total_dim = total;
    out.eta = Eigen::VectorXd::Zero(total);
    out.var_ix.assign(max_id + 1, -1);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(total) * 10);

    // ---- priors: match Python exactly ----
    int offset = 0;
    for (const auto& v : var_nodes) {
        const int id = v->variableID;
        const int m  = v->dofs;

        out.var_ix[id] = offset;

        out.eta.segment(offset, m) += v->prior.eta();

        const Eigen::MatrixXd& L = v->prior.lam();
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                const double val = L(r, c);
                if (val != 0.0) trips.emplace_back(offset + r, offset + c, val);
            }
        }

        offset += m;
    }

    // ---- factors: match Python loops (factor_ix / other_factor_ix) ----
    for (const auto& fptr : factors) {
        const Factor& f = *fptr;
        if (!f.active) continue;

        const int k = static_cast<int>(f.adj_var_nodes.size());
        if (k == 0) continue;

        const Eigen::VectorXd& f_eta = f.factor.eta();
        const Eigen::MatrixXd& f_lam = f.factor.lam();

        int factor_ix = 0;
        for (int a = 0; a < k; ++a) {
            const VariableNode* va = f.adj_var_nodes[a];
            const int ida = va->variableID;
            const int da  = va->dofs;

            const int oa = (ida >= 0 && ida < (int)out.var_ix.size()) ? out.var_ix[ida] : -1;
            if (oa < 0) throw std::runtime_error("jointDistributionInfSparse: var_ix missing ida");

            // eta diag block
            out.eta.segment(oa, da) += f_eta.segment(factor_ix, da);

            // lam diag block
            for (int r = 0; r < da; ++r) {
                for (int c = 0; c < da; ++c) {
                    const double val = f_lam(factor_ix + r, factor_ix + c);
                    if (val != 0.0) trips.emplace_back(oa + r, oa + c, val);
                }
            }

            int other_factor_ix = 0;
            for (int b = 0; b < k; ++b) {
                const VariableNode* vb = f.adj_var_nodes[b];
                const int idb = vb->variableID;
                const int db  = vb->dofs;

                const int ob = (idb >= 0 && idb < (int)out.var_ix.size()) ? out.var_ix[idb] : -1;
                if (ob < 0) throw std::runtime_error("jointDistributionInfSparse: var_ix missing idb");

                // Python: if other_adj_var_node.variableID > adj_var_node.variableID
                if (idb > ida) {
                    for (int r = 0; r < da; ++r) {
                        for (int c = 0; c < db; ++c) {
                            const double val = f_lam(factor_ix + r, other_factor_ix + c);
                            if (val != 0.0) {
                                trips.emplace_back(oa + r, ob + c, val);
                                trips.emplace_back(ob + c, oa + r, val); // symmetric
                            }
                        }
                    }
                }

                other_factor_ix += db;
            }

            factor_ix += da;
        }
    }

    out.lam.resize(total, total);
    out.lam.setFromTriplets(trips.begin(), trips.end());
    out.lam.makeCompressed();
    return out;
}

Eigen::VectorXd gbp::FactorGraph::jointMAPSparse(double diag_jitter) const {
    JointInfResult J = jointDistributionInfSparse();

    Eigen::SparseMatrix<double> A = J.lam;
    for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += diag_jitter;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("jointMAPSparse: factorization failed");
    }

    Eigen::VectorXd mu = solver.solve(J.eta);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("jointMAPSparse: solve failed");
    }
    return mu;
}

} // namespace gbp



