#include "gbp/FactorGraph.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstring>   // std::memcpy
#include <stdexcept>
#include <random>
#include <atomic>
#include <omp.h>
#include <unordered_map>

namespace gbp {

// ===== Profiling counters for synchronousIterationFixedLam (wall-clock) =====
// These counters capture the wall time spent inside the two main phases of
// FactorGraph::synchronousIterationFixedLam():
//   (1) factor->computeMessagesFixedLam(...)
//   (2) var->updateBelief()
// The counters are process-wide and thread-safe.
static std::atomic<long long> g_sync_fl_calls{0};
static std::atomic<long long> g_sync_fl_msgs_ns{0};
static std::atomic<long long> g_sync_fl_belief_ns{0};
static std::atomic<long long> g_sync_fl_total_ns{0};

void resetSyncFixedLamProfile() {
    g_sync_fl_calls.store(0, std::memory_order_relaxed);
    g_sync_fl_msgs_ns.store(0, std::memory_order_relaxed);
    g_sync_fl_belief_ns.store(0, std::memory_order_relaxed);
    g_sync_fl_total_ns.store(0, std::memory_order_relaxed);
}

void printSyncFixedLamProfile() {
    const long long calls = g_sync_fl_calls.load(std::memory_order_relaxed);
    const long long total_ns = g_sync_fl_total_ns.load(std::memory_order_relaxed);
    const long long msgs_ns  = g_sync_fl_msgs_ns.load(std::memory_order_relaxed);
    const long long bel_ns   = g_sync_fl_belief_ns.load(std::memory_order_relaxed);

    if (calls <= 0) {
        std::cout << "[synchronousIterationFixedLam Profile] No calls.\n";
        return;
    }

    const double total_ms = (double)total_ns * 1e-6;
    const double msgs_ms  = (double)msgs_ns  * 1e-6;
    const double bel_ms   = (double)bel_ns   * 1e-6;
    const double avg_total = total_ms / (double)calls;
    const double avg_msgs  = msgs_ms  / (double)calls;
    const double avg_bel   = bel_ms   / (double)calls;

    std::cout
        << "[synchronousIterationFixedLam Profile] calls=" << calls
        << " total=" << total_ms << " ms (avg " << avg_total << " ms/call)\n"
        << "  - computeMessagesFixedLam: " << msgs_ms
        << " ms (avg " << avg_msgs << " ms/call)\n"
        << "  - updateBelief:            " << bel_ms
        << " ms (avg " << avg_bel << " ms/call)\n";
}

FactorGraph::~FactorGraph() {
    if (!locks_initialized) return;
    for (auto& lk : var_locks) omp_destroy_lock(&lk);
    for (auto& lk : fac_locks) omp_destroy_lock(&lk);
    locks_initialized = false;
}

static inline void ensure_locks(
    std::vector<omp_lock_t>& var_locks,
    std::vector<omp_lock_t>& fac_locks,
    bool& initialized,
    int need_vars,
    int need_facs
) {
    if (!initialized) {
        var_locks.resize(need_vars);
        fac_locks.resize(need_facs);
        for (int i = 0; i < need_vars; ++i) omp_init_lock(&var_locks[i]);
        for (int i = 0; i < need_facs; ++i) omp_init_lock(&fac_locks[i]);
        initialized = true;
        return;
    }

    const int old_vars = (int)var_locks.size();
    const int old_facs = (int)fac_locks.size();
    if (old_vars < need_vars) {
        var_locks.resize(need_vars);
        for (int i = old_vars; i < need_vars; ++i) omp_init_lock(&var_locks[i]);
    }
    if (old_facs < need_facs) {
        fac_locks.resize(need_facs);
        for (int i = old_facs; i < need_facs; ++i) omp_init_lock(&fac_locks[i]);
    }
}

// Atomic max for double (returns true if updated).
static inline bool atomic_max_double(std::atomic<double>& a, double v) noexcept {
    double cur = a.load(std::memory_order_relaxed);
    while (v > cur) {
        if (a.compare_exchange_weak(
                cur, v,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
        // on failure, cur is updated with the latest value
    }
    return false;
}

VariableNode* FactorGraph::addVariable(int id, int dofs) {
    if ((int)var_nodes.size() <= id) {
        var_nodes.resize(id + 1);
    }
    var_nodes[id] = std::make_unique<VariableNode>(id, dofs);

    if ((int)var_residual.size() <= id) {
        var_residual.resize(id + 1, 0.0);
    }

    // Keep scheduler atomics sized with variables.
    if ((int)var_residual_a.size() <= id) {
        const int old = (int)var_residual_a.size();
        var_residual_a.resize(id + 1);
        for (int i = old; i <= id; ++i) {
            var_residual_a[i].store(0.0, std::memory_order_relaxed);
        }
    }
    if ((int)var_ver.size() <= id) {
        const int old = (int)var_ver.size();
        var_ver.resize(id + 1);
        for (int i = old; i <= id; ++i) {
            var_ver[i].store(0u, std::memory_order_relaxed);
        }
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

void FactorGraph::synchronousIteration(bool /*robustify*/) {
    // 1) factor messages (OpenMP并行)
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < (int)factors.size(); ++i) {
        auto& fptr = factors[i];
        if (!fptr->active) continue;
        fptr->computeMessages(eta_damping);
    }

    // 2) variable beliefs (OpenMP并行)
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < (int)var_nodes.size(); ++i) {
        auto& vptr = var_nodes[i];
        if (!vptr) continue;
        vptr->updateBelief();
    }

}

void FactorGraph::synchronousIterationFixedLam(bool /*robustify*/) {
    const auto t_all0 = std::chrono::high_resolution_clock::now();

    // 1) factor messages (OpenMP并行)
    const auto t_msg0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < (int)factors.size(); ++i) {
        auto& fptr = factors[i];
        if (!fptr || !fptr->active) continue;
        fptr->computeMessagesFixedLam(eta_damping);
    }
    const auto t_msg1 = std::chrono::high_resolution_clock::now();

    // 2) variable beliefs (OpenMP并行)
    const auto t_bel0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < (int)var_nodes.size(); ++i) {
        auto& vptr = var_nodes[i];
        if (!vptr) continue;
        vptr->updateBelief();
    }
    const auto t_bel1 = std::chrono::high_resolution_clock::now();

    const auto t_all1 = std::chrono::high_resolution_clock::now();

    const long long msg_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_msg1 - t_msg0).count();
    const long long bel_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_bel1 - t_bel0).count();
    const long long all_ns = (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t_all1 - t_all0).count();

    g_sync_fl_calls.fetch_add(1, std::memory_order_relaxed);
    g_sync_fl_msgs_ns.fetch_add(msg_ns, std::memory_order_relaxed);
    g_sync_fl_belief_ns.fetch_add(bel_ns, std::memory_order_relaxed);
    g_sync_fl_total_ns.fetch_add(all_ns, std::memory_order_relaxed);
}



void FactorGraph::relinearizeAllFactors() {
    // Explicit Gauss-Newton style re-linearization.
    // For each factor, build the linearization point by concatenating the
    // current means (mu) of its adjacent variables, then recompute the factor.
    for (auto& fup : factors) {
        Factor* f = fup.get();
        if (!f || !f->active) continue;

        int total = 0;
        for (auto* vn : f->adj_var_nodes) {
            if (!vn) continue;
            total += vn->dofs;
        }
        if (total <= 0) continue;

        Eigen::VectorXd linpoint(total);
        int off = 0;
        for (auto* vn : f->adj_var_nodes) {
            if (!vn) continue;
            linpoint.segment(off, vn->dofs) = vn->mu;
            off += vn->dofs;
        }

        // update_self=true: refresh cached quadratic + reset outgoing messages
        // (matches how you use computeFactor elsewhere).
        f->computeFactor(linpoint, true);
    }
}

void FactorGraph::residualIterationVarHeap(int max_updates) {
    constexpr double eps = 1e-12;
    std::atomic<int> n_updates{0};

    int n_vars_list = 0;
    for (auto& vptr : var_nodes) if (vptr) ++n_vars_list;
    if (n_vars_list == 0) return;

    // Ensure scheduler arrays are sized.
    const int n_all = (int)var_nodes.size();
    if ((int)var_residual.size() < n_all) var_residual.resize(n_all, 0.0);
    if ((int)var_residual_a.size() < n_all) {
        const int old = (int)var_residual_a.size();
        var_residual_a.resize(n_all);
        for (int i = old; i < n_all; ++i) var_residual_a[i].store(0.0, std::memory_order_relaxed);
    }
    if ((int)var_ver.size() < n_all) {
        const int old = (int)var_ver.size();
        var_ver.resize(n_all);
        for (int i = old; i < n_all; ++i) var_ver[i].store(0u, std::memory_order_relaxed);
    }

    // Lazily construct MultiQueue scheduler.
    if (!var_mq) {
        const unsigned T = std::max(1u, std::thread::hardware_concurrency());
        const int Q = std::max(1, (int)(4u * T));
        var_mq = std::make_unique<mqfast::MultiQueueFast<HeapEntry, HeapEntryKey>>(Q, HeapEntryKey{});
        // Conservative default. You can tune this later based on profiling.
        var_mq->reserve_per_queue(256);
        var_mq_initialized = false;
    }

    if (!var_mq_initialized) {
        for (int vid = 0; vid < n_all; ++vid) {
            if (!var_nodes[vid]) continue;
            var_residual[vid] = 0.0; // legacy mirror
            var_residual_a[vid].store(0.0, std::memory_order_relaxed);
            const uint32_t v0 = var_ver[vid].fetch_add(1u, std::memory_order_relaxed) + 1u;
            var_mq->push(HeapEntry{0.0, vid, v0});
        }
        var_mq_initialized = true;
    }

    // Ensure per-variable/per-factor locks for safe parallel worker loop.
    ensure_locks(var_locks, fac_locks, locks_initialized, n_all, (int)factors.size());

    // Build a fast pointer->index map for factor locks (Factor IDs are not guaranteed contiguous).
    std::unordered_map<Factor*, int> fac_to_idx;
    fac_to_idx.reserve(factors.size() * 2 + 1);
    for (int i = 0; i < (int)factors.size(); ++i) fac_to_idx.emplace(factors[i].get(), i);

    // With MultiQueue, size/empty are not reliable; drive the loop by try_pop.
    const int cap = (max_updates < 0) ? n_vars_list : std::min(max_updates, n_vars_list);

    // Parallel worker loop: each thread continuously pops a (near-)max entry and processes it.
    // We use locks to avoid data races inside Factor::computeMessages and VariableNode::updateBelief.
    #pragma omp parallel
    {
        // Scratch to avoid repeated allocations.
        std::vector<int> fac_idx_buf;
        std::vector<int> var_idx_buf;
        fac_idx_buf.reserve(16);
        var_idx_buf.reserve(8);

        while (true) {
            if (n_updates.load(std::memory_order_relaxed) >= cap) break;

            HeapEntry e;
            if (!var_mq->try_pop(e)) break;

            const int vid = e.varID;
            VariableNode* v = (vid >= 0 && vid < n_all) ? var_nodes[vid].get() : nullptr;
            if (!v) continue;
            if (!v->active) continue;

            const double r_v   = e.r;
            const uint32_t ver_e = e.ver;

            const uint32_t cur_ver = var_ver[vid].load(std::memory_order_acquire);
            if (cur_ver != ver_e) continue;

            const double cur_r = var_residual_a[vid].load(std::memory_order_acquire);
            if (std::abs(r_v - cur_r) > eps) continue;

            // ----- process all adjacent factors -----
            for (const auto& aref : v->adj_factors) {
                Factor* f = aref.factor;
                if (!f || !f->active) continue;

                auto itf = fac_to_idx.find(f);
                if (itf == fac_to_idx.end()) continue;
                const int fid = itf->second;

                // Lock all vars in this factor (sorted) to prevent concurrent belief writes
                // while computing messages.
                var_idx_buf.clear();
                var_idx_buf.reserve(f->adj_var_nodes.size());
                for (auto* u : f->adj_var_nodes) {
                    if (!u || !u->active) continue;
                    const int uid = u->variableID;
                    if (uid < 0 || uid >= n_all) continue;
                    var_idx_buf.push_back(uid);
                }
                std::sort(var_idx_buf.begin(), var_idx_buf.end());
                var_idx_buf.erase(std::unique(var_idx_buf.begin(), var_idx_buf.end()), var_idx_buf.end());
                for (int uid : var_idx_buf) omp_set_lock(&var_locks[uid]);

                // Lock factor to protect its internal message buffers.
                omp_set_lock(&fac_locks[fid]);
                f->computeMessages(eta_damping);
                omp_unset_lock(&fac_locks[fid]);

                for (int uid : var_idx_buf) omp_unset_lock(&var_locks[uid]);

                // Update beliefs for variables touched by this factor.
                for (auto* u : f->adj_var_nodes) {
                    if (!u || !u->active) continue;
                    const int uid = u->variableID;
                    if (uid < 0 || uid >= n_all) continue;

                    // Lock variable belief updates.
                    omp_set_lock(&var_locks[uid]);

                    // Also lock all adjacent factors of this variable while updateBelief reads messages.
                    fac_idx_buf.clear();
                    for (const auto& arf2 : u->adj_factors) {
                        Factor* f2 = arf2.factor;
                        if (!f2 || !f2->active) continue;
                        auto it2 = fac_to_idx.find(f2);
                        if (it2 != fac_to_idx.end()) fac_idx_buf.push_back(it2->second);
                    }
                    std::sort(fac_idx_buf.begin(), fac_idx_buf.end());
                    fac_idx_buf.erase(std::unique(fac_idx_buf.begin(), fac_idx_buf.end()), fac_idx_buf.end());
                    for (int fidx : fac_idx_buf) omp_set_lock(&fac_locks[fidx]);

                    Eigen::VectorXd old_eta = u->belief.eta();
                    u->updateBelief();
                    Eigen::VectorXd new_eta = u->belief.eta();

                    for (auto rit = fac_idx_buf.rbegin(); rit != fac_idx_buf.rend(); ++rit) {
                        omp_unset_lock(&fac_locks[*rit]);
                    }
                    omp_unset_lock(&var_locks[uid]);

                    const double est_r_u = (new_eta - old_eta).norm();
                    if (est_r_u > eps) {
                        if (atomic_max_double(var_residual_a[uid], est_r_u)) {
                            const uint32_t newv = var_ver[uid].fetch_add(1u, std::memory_order_acq_rel) + 1u;
                            var_mq->push(HeapEntry{est_r_u, uid, newv});
                        }
                    }
                }
            }

            // Reset residual for the processed variable and invalidate any stale entries.
            var_residual_a[vid].store(0.0, std::memory_order_release);
            (void)var_ver[vid].fetch_add(1u, std::memory_order_acq_rel);

            n_updates.fetch_add(1, std::memory_order_relaxed);
        }
    }

}

gbp::FactorGraph::JointInfResult gbp::FactorGraph::jointDistributionInfSparse() const {
    int total = 0;
    int max_id = -1;
    for (const auto& v : var_nodes) {
        if (!v) continue;
        total += v->dofs;
        max_id = std::max(max_id, v->variableID);
    }

    JointInfResult out;
    out.total_dim = total;
    out.eta = Eigen::VectorXd::Zero(total);
    out.var_ix.assign(max_id + 1, -1);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(total) * 10);

    int offset = 0;
    for (const auto& v : var_nodes) {
        if (!v) continue;
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

            out.eta.segment(oa, da) += f_eta.segment(factor_ix, da);

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

                if (idb > ida) {
                    for (int r = 0; r < da; ++r) {
                        for (int c = 0; c < db; ++c) {
                            const double val = f_lam(factor_ix + r, other_factor_ix + c);
                            if (val != 0.0) {
                                trips.emplace_back(oa + r, ob + c, val);
                                trips.emplace_back(ob + c, oa + r, val);
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


/*
// ------------------------------
// CHOLMOD helpers
// ------------------------------
static inline const char* cholmodItypeName(int itype) {
    if (itype == CHOLMOD_INT)  return "CHOLMOD_INT(32)";
    if (itype == CHOLMOD_LONG) return "CHOLMOD_LONG(64)";
    return "UNKNOWN";
}

static cholmod_sparse* eigen_to_cholmod_lower_sym(
    const Eigen::SparseMatrix<double>& A,
    cholmod_common* c
) {
    if (A.rows() != A.cols()) {
        throw std::runtime_error("CHOLMOD: A must be square");
    }
    if (!A.isCompressed()) {
        throw std::runtime_error("CHOLMOD: Eigen matrix must be compressed");
    }

    const int n = (int)A.rows();
    const int* Ap = A.outerIndexPtr();   // size n+1 (int in Eigen by default)
    const int* Ai = A.innerIndexPtr();   // size nnz
    const double* Ax = A.valuePtr();     // size nnz

    // count nnz in lower triangle (row >= col)
    SuiteSparse_long nnzL = 0;
    for (int col = 0; col < n; ++col) {
        for (int k = Ap[col]; k < Ap[col + 1]; ++k) {
            const int row = Ai[k];
            if (row >= col) ++nnzL;
        }
    }

    // IMPORTANT:
    // sorted MUST be 0 here (do NOT claim it's sorted).
    // packed must be 1 (CSC compressed).
    cholmod_sparse* M = cholmod_allocate_sparse(
        (size_t)n, (size_t)n, (size_t)nnzL,
        0,
        1,
        -1,              // symmetric, store LOWER triangle
        CHOLMOD_REAL,
        c
    );
    if (!M) return nullptr;

    // Fill p/i with correct index type depending on c->itype
    SuiteSparse_long pos = 0;

    if (c->itype == CHOLMOD_LONG) {
        auto* Mp = static_cast<SuiteSparse_long*>(M->p);
        auto* Mi = static_cast<SuiteSparse_long*>(M->i);
        auto* Mx = static_cast<double*>(M->x);

        Mp[0] = 0;
        for (int col = 0; col < n; ++col) {
            for (int k = Ap[col]; k < Ap[col + 1]; ++k) {
                const int row = Ai[k];
                if (row < col) continue;
                Mi[pos] = (SuiteSparse_long)row;
                Mx[pos] = Ax[k];
                ++pos;
            }
            Mp[col + 1] = pos;
        }
    } else if (c->itype == CHOLMOD_INT) {
        auto* Mp = static_cast<int*>(M->p);
        auto* Mi = static_cast<int*>(M->i);
        auto* Mx = static_cast<double*>(M->x);

        Mp[0] = 0;
        for (int col = 0; col < n; ++col) {
            for (int k = Ap[col]; k < Ap[col + 1]; ++k) {
                const int row = Ai[k];
                if (row < col) continue;
                Mi[(int)pos] = row;
                Mx[(int)pos] = Ax[k];
                ++pos;
            }
            Mp[col + 1] = (int)pos;
        }
    } else {
        cholmod_free_sparse(&M, c);
        throw std::runtime_error("CHOLMOD: unsupported itype");
    }

    return M;
}

// 一次性 CHOLMOD: analyze + factorize + solve，并返回解
static Eigen::VectorXd cholmod_cholesky_solve_once(
    Eigen::SparseMatrix<double> A,      // 传值：允许加 jitter、压缩
    const Eigen::VectorXd& b,
    double diag_jitter
) {
    if (A.rows() != A.cols()) throw std::runtime_error("CHOLMOD: A must be square");
    if (A.rows() != b.size()) throw std::runtime_error("CHOLMOD: dimension mismatch");

    if (diag_jitter != 0.0) {
        for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += diag_jitter;
    }
    A.makeCompressed();

    cholmod_common c;
    cholmod_start(&c);
    c.supernodal = CHOLMOD_SUPERNODAL;

    cholmod_sparse* Ac = eigen_to_cholmod_lower_sym(A, &c);
    if (!Ac) {
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: eigen_to_cholmod_lower_sym returned null");
    }
    cholmod_sort(Ac, &c);

    cholmod_dense* bc = cholmod_allocate_dense(
        (size_t)b.size(), 1, (size_t)b.size(), CHOLMOD_REAL, &c
    );
    if (!bc) {
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: allocate_dense(b) failed");
    }
    std::memcpy(bc->x, b.data(), (size_t)b.size() * sizeof(double));

    cholmod_factor* L = cholmod_analyze(Ac, &c);
    int ok = cholmod_factorize(Ac, L, &c);
    if (!ok || c.status != CHOLMOD_OK) {
        cholmod_free_factor(&L, &c);
        cholmod_free_dense(&bc, &c);
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: factorization failed (try larger diag_jitter)");
    }

    cholmod_dense* xc = cholmod_solve(CHOLMOD_A, L, bc, &c);
    if (!xc || c.status != CHOLMOD_OK) {
        if (xc) cholmod_free_dense(&xc, &c);
        cholmod_free_factor(&L, &c);
        cholmod_free_dense(&bc, &c);
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: solve failed");
    }

    Eigen::VectorXd x(b.size());
    std::memcpy(x.data(), xc->x, (size_t)b.size() * sizeof(double));

    cholmod_free_dense(&xc, &c);
    cholmod_free_factor(&L, &c);
    cholmod_free_dense(&bc, &c);
    cholmod_free_sparse(&Ac, &c);
    cholmod_finish(&c);

    return x;
}
*/

#include <random>   // 必须：mt19937 / normal_distribution

Eigen::VectorXd gbp::FactorGraph::jointMAPSparse(double diag_jitter) const {
    JointInfResult J = jointDistributionInfSparse();

    // ---------------------------
    // 1) Eigen baseline
    // ---------------------------
    Eigen::SparseMatrix<double> A = J.lam;
    if (diag_jitter != 0.0) {
        for (int i = 0; i < A.rows(); ++i) A.coeffRef(i, i) += diag_jitter;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    A.makeCompressed();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("jointMAPSparse: Eigen factorization failed");
    }

    Eigen::VectorXd mu;
    {
        auto t2 = std::chrono::high_resolution_clock::now();
        mu = solver.solve(J.eta);
        auto t3 = std::chrono::high_resolution_clock::now();
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("jointMAPSparse: Eigen solve failed");
        }
        const double factor_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double solve_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "[SimplicialLDLT] factor=" << factor_ms << " ms, solve=" << solve_ms << " ms\n";
    }

    /*
    // ---------------------------
    // 2) CHOLMOD: supernodal check (quiet) + optional grid test
    // ---------------------------

    // Toggle this to test a matrix that is much more likely to create supernodes
    // than a near-chain pose graph.
    constexpr bool kUseGridForCholmodTest = false; // <-- set true to use grid Laplacian

    auto build_grid_laplacian = [](int side, double jitter) -> Eigen::SparseMatrix<double> {
        // SPD: L + jitter*I
        const int n = side * side;
        std::vector<Eigen::Triplet<double>> T;
        T.reserve((size_t)n * 5);

        auto id = [side](int r, int c) { return r * side + c; };

        for (int r = 0; r < side; ++r) {
            for (int c = 0; c < side; ++c) {
                const int p = id(r, c);

                int deg = 0;
                // 4-neighbors
                if (r > 0)        { T.emplace_back(p, id(r - 1, c), -1.0); ++deg; }
                if (r + 1 < side) { T.emplace_back(p, id(r + 1, c), -1.0); ++deg; }
                if (c > 0)        { T.emplace_back(p, id(r, c - 1), -1.0); ++deg; }
                if (c + 1 < side) { T.emplace_back(p, id(r, c + 1), -1.0); ++deg; }

                T.emplace_back(p, p, (double)deg + jitter);
            }
        }

        Eigen::SparseMatrix<double> L(n, n);
        L.setFromTriplets(T.begin(), T.end());
        L.makeCompressed();
        return L;
    };

    Eigen::SparseMatrix<double> A_chol;
    Eigen::VectorXd b_chol;

    if constexpr (kUseGridForCholmodTest) {
        // Pick a size large enough to show typical sparse behavior
        // (e.g. 120 -> 14400 vars). You can tune this.
        const int side = 120;
        const double jitter = (diag_jitter != 0.0) ? diag_jitter : 1e-3;

        A_chol = build_grid_laplacian(side, jitter);
        b_chol = Eigen::VectorXd::Ones(A_chol.rows());
    } else {
        // Use your actual normal equations / information matrix
        A_chol = A;
        b_chol = J.eta;
    }

    cholmod_common c;
    cholmod_start(&c);
    std::cout << "[CHOLMOD] macros: SIMP=" << CHOLMOD_SIMPLICIAL
            << " AUTO=" << CHOLMOD_AUTO
            << " SUPER=" << CHOLMOD_SUPERNODAL << "\n";

    c.supernodal = 2;  // 期望是 2
    c.supernodal_switch = 0;            // 避免任何“切回 simplicial”的启发式
    c.final_super = 1;
    c.final_ll    = 1;


    // Keep CHOLMOD quiet
    c.print = 0;
    // Request supernodal
    c.supernodal = 0;

    // Convert Eigen -> CHOLMOD (lower symmetric)
    // IMPORTANT: ensure eigen_to_cholmod_lower_sym uses sorted=0 OR we sort here.
    cholmod_sparse* Ac = eigen_to_cholmod_lower_sym(A_chol, &c);
    if (!Ac || c.status != CHOLMOD_OK) {
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: eigen_to_cholmod_lower_sym failed");
    }

    // Force sorting to be safe
    cholmod_sort(Ac, &c);
    if (c.status != CHOLMOD_OK) {
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: cholmod_sort failed");
    }

    // RHS
    cholmod_dense* bc = cholmod_allocate_dense(
        (size_t)b_chol.size(), 1, (size_t)b_chol.size(), CHOLMOD_REAL, &c
    );
    if (!bc || c.status != CHOLMOD_OK) {
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: allocate_dense(b) failed");
    }
    std::memcpy(bc->x, b_chol.data(), sizeof(double) * (size_t)b_chol.size());

    // Analyze
    cholmod_factor* L = cholmod_analyze(Ac, &c);
    if (!L || c.status != CHOLMOD_OK) {
        cholmod_free_dense(&bc, &c);
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: analyze failed");
    }

    // Factorize + Solve timing
    auto tc0 = std::chrono::high_resolution_clock::now();

    const int ok = cholmod_factorize(Ac, L, &c);
    if (!ok || c.status != CHOLMOD_OK) {
        cholmod_free_factor(&L, &c);
        cholmod_free_dense(&bc, &c);
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: factorize failed (non-SPD or other issue)");
    }

    cholmod_dense* xc = cholmod_solve(CHOLMOD_A, L, bc, &c);

    auto tc1 = std::chrono::high_resolution_clock::now();

    if (!xc || c.status != CHOLMOD_OK) {
        if (xc) cholmod_free_dense(&xc, &c);
        cholmod_free_factor(&L, &c);
        cholmod_free_dense(&bc, &c);
        cholmod_free_sparse(&Ac, &c);
        cholmod_finish(&c);
        throw std::runtime_error("CHOLMOD: solve failed");
    }

    const double chol_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

    // Minimal supernodal verdict
    std::cout
        << "[CHOLMOD] requested=SUPERNODAL, is_super(after factorize)=" << L->is_super
        << ", factorize+solve=" << chol_ms << " ms\n";

    // (Optional) retrieve solution if you want to validate:
    // Eigen::VectorXd mu_chol(b_chol.size());
    // std::memcpy(mu_chol.data(), xc->x, sizeof(double) * (size_t)b_chol.size());

    // Free
    cholmod_free_dense(&xc, &c);
    cholmod_free_factor(&L, &c);
    cholmod_free_dense(&bc, &c);
    cholmod_free_sparse(&Ac, &c);
    cholmod_finish(&c);

    */
    // Keep behavior unchanged: return Eigen solution.
    // If you want to validate CHOLMOD result, return mu_chol instead.
    return mu;
}



} // namespace gbp
