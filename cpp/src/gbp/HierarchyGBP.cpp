#include "gbp/HierarchyGBP.h"

#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <iostream>
#include <Eigen/Eigenvalues>

#ifdef _OPENMP
#include <omp.h>
#endif

// ===== Profiling counters for bottomUp analysis =====
static std::atomic<long long> g_bottomUp_belief_copy_ns{0};
static std::atomic<long long> g_bottomUp_setLamEta_ns{0};
static std::atomic<long long> g_bottomUp_computeFactor_ns{0};
static std::atomic<long long> g_bottomUp_lp_build_ns{0};
static std::atomic<long long> g_bottomUpAbs_proj_ns{0};
static std::atomic<long long> g_bottomUpAbs_llt_ns{0};
static std::atomic<long long> g_bottomUpAbs_setbelief_ns{0};
static std::atomic<int> g_bottomUp_call_count{0};

void printBottomUpProfile() {
    int calls = g_bottomUp_call_count.load();
    if (calls == 0) {
        std::cout << "[BottomUp Profile] No calls recorded.\n";
        return;
    }
    auto toMs = [](long long ns) { return ns / 1e6; };
    std::cout << "\n=== BottomUp Detailed Profile (" << calls << " calls total) ===\n";
    std::cout << "  [Super] belief copy (mu/Sigma/eta/lam build): " << toMs(g_bottomUp_belief_copy_ns.load()) << " ms\n";
    std::cout << "  [Super] setLam/setEta:                        " << toMs(g_bottomUp_setLamEta_ns.load()) << " ms\n";
    std::cout << "  [Super] lp build (linpoint concat):           " << toMs(g_bottomUp_lp_build_ns.load()) << " ms\n";
    std::cout << "  [Super] computeFactor (meas+jac+linearize):   " << toMs(g_bottomUp_computeFactor_ns.load()) << " ms\n";
    std::cout << "  [Abs]   projection (B'*mu, B'*Sigma*B, k):    " << toMs(g_bottomUpAbs_proj_ns.load()) << " ms\n";
    std::cout << "  [Abs]   LLT + solve (lam=inv, eta=lam*mu):    " << toMs(g_bottomUpAbs_llt_ns.load()) << " ms\n";
    std::cout << "  [Abs]   set belief/mu/Sigma:                  " << toMs(g_bottomUpAbs_setbelief_ns.load()) << " ms\n";
    std::cout << "==========================================\n";
}

void resetBottomUpProfile() {
    g_bottomUp_belief_copy_ns = 0;
    g_bottomUp_setLamEta_ns = 0;
    g_bottomUp_computeFactor_ns = 0;
    g_bottomUp_lp_build_ns = 0;
    g_bottomUpAbs_proj_ns = 0;
    g_bottomUpAbs_llt_ns = 0;
    g_bottomUpAbs_setbelief_ns = 0;
    g_bottomUp_call_count = 0;
}

namespace gbp {

HierarchyGBP::HierarchyGBP(int group_size)
: k_(group_size <= 0 ? 1 : group_size) {}

std::vector<int> HierarchyGBP::makeOrderNodeMap_(int n_base) const {
    // order grouping: sid = bid / k
    std::vector<int> m(n_base, 0);
    for (int bid = 0; bid < n_base; ++bid) {
        m[bid] = bid / k_;
    }
    return m;
}

std::shared_ptr<SuperLayer>
HierarchyGBP::buildSuperFromBase(const std::shared_ptr<FactorGraph>& base, double eta_damping_super) {
    auto layer = std::make_shared<SuperLayer>();
    layer->group_size = k_;

    const int n_base = static_cast<int>(base->var_nodes.size());
    if (n_base <= 0) {
        layer->graph = std::make_shared<FactorGraph>();
        layer->graph->eta_damping = eta_damping_super;
        return layer;
    }

    layer->node_map = makeOrderNodeMap_(n_base);
    const int n_super = (n_base + k_ - 1) / k_;

    layer->groups.resize(n_super);
    layer->local_idx.resize(n_super);
    layer->total_dofs.assign(n_super, 0);

    // ---- invert groups ----
    for (int bid = 0; bid < n_base; ++bid) {
        const int sid = layer->node_map[bid];
        layer->groups[sid].push_back(bid);
    }

    // ---- build local_idx and total dofs per super ----
    for (int sid = 0; sid < n_super; ++sid) {
        int off = 0;
        for (int bid : layer->groups[sid]) {
            const auto& bv = *base->var_nodes[bid];
            layer->local_idx[sid][bid] = {off, bv.dofs};
            off += bv.dofs;
        }
        layer->total_dofs[sid] = off;
    }

    // ---- create super graph ----
    layer->graph = std::make_shared<FactorGraph>();
    layer->graph->nonlinear_factors = base->nonlinear_factors;
    layer->graph->eta_damping = eta_damping_super;

    // ---- create super variables (and initialize mu/Sigma/belief/prior like Python) ----
    for (int sid = 0; sid < n_super; ++sid) {
        const int dofs = layer->total_dofs[sid];
        auto* sv = layer->graph->addVariable(sid, dofs);

        // mu_super = concat(base mu), Sigma_super = block_diag(base Sigma)
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(dofs);
        Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(dofs, dofs);
        sv->GT = Eigen::VectorXd::Zero(dofs);

        for (int bid : layer->groups[sid]) {
            const auto& bv = *base->var_nodes[bid];
            const auto [off, d] = layer->local_idx[sid].at(bid);

            mu.segment(off, d) = bv.mu;
            Sigma.block(off, off, d, d) = bv.Sigma;
            sv->GT.segment(off, d) = bv.GT.head(d);
        }

        // lam = inv(Sigma), eta = lam * mu  (Python parity)
        // 注意：如果 Sigma 奇异（例如没有 computeSigma），这里会失败。
        // 你的 pipeline 中 VariableNode 默认 compute Sigma，所以这里通常可行。
        Eigen::MatrixXd lam = Sigma.inverse();
        Eigen::VectorXd eta = lam * mu;

        sv->mu = mu;
        sv->Sigma = Sigma;

        sv->belief = utils::NdimGaussian(dofs, eta, lam);
        // prior very weak: 1e-12 * lam/eta (Python)
        sv->prior = utils::NdimGaussian(dofs);
        sv->prior.setLam(1e-12 * lam);
        sv->prior.setEta(1e-12 * eta);
    }

    // ============================================================
    // Precompute: in-group factors and cross-group factors
    // ============================================================
    std::unordered_map<int, std::vector<Factor*>> in_group; // sid -> base factors
    std::unordered_map<std::pair<int,int>, std::vector<Factor*>, PairHash> cross_group; // (sidA,sidB) -> base factors

    for (auto& f_up : base->factors) {
        Factor& f = *f_up;

        const int arity = static_cast<int>(f.adj_var_nodes.size());
        if (arity == 1) {
            const int i = f.adj_var_nodes[0]->variableID;
            const int si = layer->node_map[i];
            in_group[si].push_back(&f);
        } else if (arity == 2) {
            const int i = f.adj_var_nodes[0]->variableID;
            const int j = f.adj_var_nodes[1]->variableID;
            int si = layer->node_map[i];
            int sj = layer->node_map[j];
            if (si == sj) {
                in_group[si].push_back(&f);
            } else {
                if (si > sj) std::swap(si, sj);
                cross_group[{si, sj}].push_back(&f);
            }
        } else {
            // 目前你的 C++ Factor 只支持 unary/binary
        }
    }

    // ============================================================
    // 1) Super prior factors: all-in group (unary + intra-group binary collapsed)
    // ============================================================
    for (auto& kv : in_group) {
        const int sid = kv.first;
        const auto base_factors = kv.second; // copy vector<Factor*>

        const auto& idx_map = layer->local_idx[sid];
        const int ncols = layer->total_dofs[sid];

        // z_super, lambda_super: concat all base measurement blocks
        std::vector<Eigen::VectorXd> z_super;
        std::vector<Eigen::MatrixXd> lam_super;
        for (auto* bf : base_factors) {
            z_super.insert(z_super.end(), bf->measurement.begin(), bf->measurement.end());
            lam_super.insert(lam_super.end(), bf->measurement_lambda.begin(), bf->measurement_lambda.end());
        }

        // meas_fn_super_prior(x_super) -> list[Vector]
        auto meas_fn = [base_factors, idx_map](const Eigen::VectorXd& x_super) {
            std::vector<Eigen::VectorXd> out;
            for (auto* bf : base_factors) {
                // x_loc = concat slices in the SAME order as bf->adj_var_nodes
                int total = 0;
                for (auto* v : bf->adj_var_nodes) total += v->dofs;

                Eigen::VectorXd x_loc(total);
                int c0 = 0;
                for (auto* v : bf->adj_var_nodes) {
                    const int vid = v->variableID;
                    const auto it = idx_map.find(vid);
                    // 组内 factor，理论上一定找到
                    const auto [st, d] = it->second;
                    x_loc.segment(c0, d) = x_super.segment(st, d);
                    c0 += d;
                }

                auto blocks = bf->meas_fn(x_loc);
                out.insert(out.end(), blocks.begin(), blocks.end());
            }
            return out;
        };

        // jac_fn_super_prior(x_super) -> list[Matrix(rows, ncols)]
        auto jac_fn = [base_factors, idx_map, ncols](const Eigen::VectorXd& x_super) {
            std::vector<Eigen::MatrixXd> out;

            for (auto* bf : base_factors) {
                // build x_loc and dims
                int total = 0;
                std::vector<int> vids;
                std::vector<int> dims;
                vids.reserve(bf->adj_var_nodes.size());
                dims.reserve(bf->adj_var_nodes.size());

                for (auto* v : bf->adj_var_nodes) {
                    vids.push_back(v->variableID);
                    dims.push_back(v->dofs);
                    total += v->dofs;
                }

                Eigen::VectorXd x_loc(total);
                int c0 = 0;
                for (size_t t = 0; t < vids.size(); ++t) {
                    const int vid = vids[t];
                    const int d = dims[t];
                    const auto [st, _] = idx_map.at(vid);
                    x_loc.segment(c0, d) = x_super.segment(st, d);
                    c0 += d;
                }

                auto Jloc = bf->jac_fn(x_loc); // list of blocks
                for (auto& J : Jloc) {
                    Eigen::MatrixXd Js = Eigen::MatrixXd::Zero(J.rows(), ncols);
                    int col = 0;
                    for (size_t t = 0; t < vids.size(); ++t) {
                        const int vid = vids[t];
                        const int d = dims[t];
                        const auto [st, _] = idx_map.at(vid);
                        Js.block(0, st, J.rows(), d) = J.block(0, col, J.rows(), d);
                        col += d;
                    }
                    out.push_back(std::move(Js));
                }
            }

            return out;
        };

        // add factor + connect
        auto* v0 = layer->graph->var_nodes[sid].get();
        auto* sf = layer->graph->addFactor(
            static_cast<int>(layer->graph->factors.size()),
            std::vector<VariableNode*>{v0},
            z_super,
            lam_super,
            meas_fn,
            jac_fn
        );
        sf->invalidateJacobianCache();
        layer->graph->connect(sf, v0, 0);
    }

    // ============================================================
    // 2) Super between factors: cross-group binaries
    // ============================================================
    for (auto& kv : cross_group) {
        const int sidA = kv.first.first;
        const int sidB = kv.first.second;
        const auto base_factors = kv.second; // copy

        const auto& idxA = layer->local_idx[sidA];
        const auto& idxB = layer->local_idx[sidB];
        const int nA = layer->total_dofs[sidA];
        const int nB = layer->total_dofs[sidB];

        // z_super, lambda_super
        std::vector<Eigen::VectorXd> z_super;
        std::vector<Eigen::MatrixXd> lam_super;
        for (auto* bf : base_factors) {
            z_super.insert(z_super.end(), bf->measurement.begin(), bf->measurement.end());
            lam_super.insert(lam_super.end(), bf->measurement_lambda.begin(), bf->measurement_lambda.end());
        }

        // meas_fn_super_between(xAB) where xAB = [xA; xB]
        auto meas_fn = [base_factors, idxA, idxB, nA, nB, sidA, sidB, node_map = layer->node_map](const Eigen::VectorXd& xAB) {
            const Eigen::VectorXd xA = xAB.head(nA);
            const Eigen::VectorXd xB = xAB.tail(nB);

            std::vector<Eigen::VectorXd> out;
            for (auto* bf : base_factors) {
                const int i = bf->adj_var_nodes[0]->variableID;
                const int j = bf->adj_var_nodes[1]->variableID;

                // 按 bf 的变量顺序构造 x_loc = [xi; xj]
                Eigen::VectorXd xi, xj;
                int di = bf->adj_var_nodes[0]->dofs;
                int dj = bf->adj_var_nodes[1]->dofs;

                Eigen::VectorXd x_loc(di + dj);

                const int si = node_map[i];
                const int sj = node_map[j];

                if (si == sidA && sj == sidB) {
                    const auto [oi, diA] = idxA.at(i);
                    const auto [oj, djB] = idxB.at(j);
                    x_loc.segment(0, di)     = xA.segment(oi, di);
                    x_loc.segment(di, dj)    = xB.segment(oj, dj);
                } else {
                    // swapped
                    const auto [oi, diB] = idxB.at(i);
                    const auto [oj, djA] = idxA.at(j);
                    x_loc.segment(0, di)     = xB.segment(oi, di);
                    x_loc.segment(di, dj)    = xA.segment(oj, dj);
                }

                auto blocks = bf->meas_fn(x_loc);
                out.insert(out.end(), blocks.begin(), blocks.end());
            }
            return out;
        };

        // jac_fn_super_between(xAB) -> list[Matrix(rows, nA+nB)]
        auto jac_fn = [base_factors, idxA, idxB, nA, nB, sidA, sidB, node_map = layer->node_map](const Eigen::VectorXd& xAB) {
            const Eigen::VectorXd xA = xAB.head(nA);
            const Eigen::VectorXd xB = xAB.tail(nB);

            std::vector<Eigen::MatrixXd> out;
            for (auto* bf : base_factors) {
                const int i = bf->adj_var_nodes[0]->variableID;
                const int j = bf->adj_var_nodes[1]->variableID;

                const int di = bf->adj_var_nodes[0]->dofs;
                const int dj = bf->adj_var_nodes[1]->dofs;

                Eigen::VectorXd x_loc(di + dj);

                const int si = node_map[i];
                const int sj = node_map[j];

                bool normal = (si == sidA && sj == sidB);

                if (normal) {
                    const auto [oi, diA] = idxA.at(i);
                    const auto [oj, djB] = idxB.at(j);
                    x_loc.segment(0, di)  = xA.segment(oi, di);
                    x_loc.segment(di, dj) = xB.segment(oj, dj);
                } else {
                    const auto [oi, diB] = idxB.at(i);
                    const auto [oj, djA] = idxA.at(j);
                    x_loc.segment(0, di)  = xB.segment(oi, di);
                    x_loc.segment(di, dj) = xA.segment(oj, dj);
                }

                auto Jloc = bf->jac_fn(x_loc);
                for (auto& J : Jloc) {
                    Eigen::MatrixXd Js = Eigen::MatrixXd::Zero(J.rows(), nA + nB);

                    if (normal) {
                        const auto [oi, diA] = idxA.at(i);
                        const auto [oj, djB] = idxB.at(j);
                        Js.block(0, oi,        J.rows(), di) = J.block(0, 0,  J.rows(), di);
                        Js.block(0, nA + oj,   J.rows(), dj) = J.block(0, di, J.rows(), dj);
                    } else {
                        // i in B, j in A
                        const auto [oi, diB] = idxB.at(i);
                        const auto [oj, djA] = idxA.at(j);
                        Js.block(0, nA + oi,   J.rows(), di) = J.block(0, 0,  J.rows(), di);
                        Js.block(0, oj,        J.rows(), dj) = J.block(0, di, J.rows(), dj);
                    }

                    out.push_back(std::move(Js));
                }
            }
            return out;
        };

        auto* vA = layer->graph->var_nodes[sidA].get();
        auto* vB = layer->graph->var_nodes[sidB].get();
        auto* sf = layer->graph->addFactor(
            static_cast<int>(layer->graph->factors.size()),
            std::vector<VariableNode*>{vA, vB},
            z_super,
            lam_super,
            meas_fn,
            jac_fn
        );
        sf->invalidateJacobianCache();
        layer->graph->connect(sf, vA, 0);
        layer->graph->connect(sf, vB, 1);
    }

    return layer;
}

void HierarchyGBP::bottomUpUpdateSuper(
    const std::shared_ptr<FactorGraph>& base,
    const std::shared_ptr<SuperLayer>& super) {

    ++g_bottomUp_call_count;

    // 1) belief: block-diag copy base -> super
    const int n_super = static_cast<int>(super->groups.size());
    #pragma omp parallel for schedule(dynamic)
    for (int sid = 0; sid < n_super; ++sid) {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto& sv = *super->graph->var_nodes[sid];
        const int D = super->total_dofs[sid];

        Eigen::VectorXd eta = Eigen::VectorXd::Zero(D);
        Eigen::MatrixXd lam = Eigen::MatrixXd::Zero(D, D);

        // Keep moments in sync without forcing LLT/inverse in NdimGaussian.
        sv.mu.resize(D);
        sv.Sigma.setZero(D, D);

        for (int bid : super->groups[sid]) {
            auto& bv = *base->var_nodes[bid];
            const auto [off, d] = super->local_idx[sid].at(bid);

            eta.segment(off, d) = bv.belief.eta();
            lam.block(off, off, d, d) = bv.belief.lam();

            // moments: block-diag copy
            sv.mu.segment(off, d) = bv.mu;
            sv.Sigma.block(off, off, d, d) = bv.Sigma;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        g_bottomUp_belief_copy_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        // belief set (lam first to reset factorization)
        sv.belief.setLam(lam);
        sv.belief.setEta(eta);

        auto t2 = std::chrono::high_resolution_clock::now();
        g_bottomUp_setLamEta_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }

    // 2) relinearize all super factors at current concatenated mu
    const int n_f = static_cast<int>(super->graph->factors.size());
    #pragma omp parallel for schedule(dynamic)
    for (int fi = 0; fi < n_f; ++fi) {
        auto& fptr = super->graph->factors[fi];
        if (!fptr) continue;
        Factor& f = *fptr;
        if (!f.active) continue;

        auto t0 = std::chrono::high_resolution_clock::now();

        // linpoint = concat mu of adj vars in order
        int total = 0;
        for (auto* v : f.adj_var_nodes) total += v->dofs;

        static thread_local Eigen::VectorXd lp;
        lp.resize(total);

        int c0 = 0;
        for (auto* v : f.adj_var_nodes) {
            lp.segment(c0, v->dofs) = v->mu;
            c0 += v->dofs;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        g_bottomUp_lp_build_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        f.computeFactor(lp, true);

        auto t2 = std::chrono::high_resolution_clock::now();
        g_bottomUp_computeFactor_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    }
}



// ======================
// Abs layer implementation (matches Python build_abs_graph / bottom_up_modify_abs_graph)
// ======================

static inline Eigen::MatrixXd topEigenvectorsDescending(const Eigen::MatrixXd& Sigma, int r) {
    // Python: eigvals, eigvecs = np.linalg.eigh(Sigma); sort eigvals desc; take eigvecs[:, :r]
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sigma);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("Abs: Eigen decomposition failed in topEigenvectorsDescending()");
    }
    const Eigen::VectorXd evals = es.eigenvalues();      // ascending
    const Eigen::MatrixXd evecs = es.eigenvectors();     // columns correspond to evals

    // indices sorted by eval desc
    std::vector<int> idx(evals.size());
    for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return evals[a] > evals[b]; });

    Eigen::MatrixXd B(Sigma.rows(), r);
    for (int k = 0; k < r; ++k) {
        B.col(k) = evecs.col(idx[k]);
    }
    return B;
}

std::shared_ptr<AbsLayer>
HierarchyGBP::buildAbsFromSuper(
    const std::shared_ptr<FactorGraph>& sup_fg,
    int r_reduced,
    double eta_damping_abs
) {
    auto abs = std::make_shared<AbsLayer>();
    abs->graph = std::make_shared<FactorGraph>();
    abs->graph->nonlinear_factors = false;         // Python: nonlinear_factors=False
    abs->graph->eta_damping = eta_damping_abs;

    // === 1) Build abstraction variables ===
    for (const auto& sn_uptr : sup_fg->var_nodes) {
        if (!sn_uptr) continue;
        const auto& sn = *sn_uptr;

        const int sid = sn.variableID;
        const int r = (sn.dofs <= r_reduced) ? sn.dofs : r_reduced;

        const Eigen::VectorXd mu_sup = sn.mu;
        const Eigen::MatrixXd Sigma_sup = sn.Sigma;

        // eig decomposition + take top-r eigenvectors (descending)
        Eigen::MatrixXd B_reduced = topEigenvectorsDescending(Sigma_sup, r);
        abs->Bs[sid] = B_reduced;

        // projection
        Eigen::VectorXd mu_abs = B_reduced.transpose() * mu_sup;
        Eigen::MatrixXd Sigma_abs = B_reduced.transpose() * Sigma_sup * B_reduced;

        Eigen::VectorXd k = mu_sup - B_reduced * mu_abs;
        abs->ks[sid] = k;

        Eigen::MatrixXd lam_abs = Sigma_abs.inverse();
        Eigen::VectorXd eta_abs = lam_abs * mu_abs;

        // create variable
        auto* v = abs->graph->addVariable(sid, r);
        v->GT = sn.GT;
        v->mu = mu_abs;
        v->Sigma = Sigma_abs;

        // belief set (lam first to reset factorization) - match style used elsewhere
        v->belief.setLam(lam_abs);
        v->belief.setEta(eta_abs);
    }

    // === 2) Build all abs factors once ===
    for (const auto& fptr : sup_fg->factors) {
        if (!fptr) continue;
        const Factor* sup_f = fptr.get();
        if (!sup_f->active) continue;

        const int k_adj = (int)sup_f->adj_var_nodes.size();
        if (k_adj == 1) {
            // prior
            const int sid = sup_f->adj_var_nodes[0]->variableID;

            auto meas_fn = [abs, sup_f, sid](const Eigen::VectorXd& x_abs) -> std::vector<Eigen::VectorXd> {
                const Eigen::MatrixXd& B = abs->Bs.at(sid);
                const Eigen::VectorXd& k = abs->ks.at(sid);
                Eigen::VectorXd x_sup = B * x_abs + k;
                return sup_f->meas_fn(x_sup);
            };

            auto jac_fn = [abs, sup_f, sid](const Eigen::VectorXd& x_abs) -> std::vector<Eigen::MatrixXd> {
                const Eigen::MatrixXd& B = abs->Bs.at(sid);
                const Eigen::VectorXd& k = abs->ks.at(sid);
                Eigen::VectorXd x_sup = B * x_abs + k;

                auto Jloc = sup_f->jac_fn(x_sup);
                std::vector<Eigen::MatrixXd> out;
                out.reserve(Jloc.size());
                for (const auto& J : Jloc) {
                    out.push_back(J * B); // blockwise == vstack(Jloc) @ B then split
                }
                return out;
            };

            auto* v0 = abs->graph->var_nodes[sid].get();
            auto* af = abs->graph->addFactor(
                sup_f->factorID,
                std::vector<VariableNode*>{v0},
                sup_f->measurement,
                sup_f->measurement_lambda,
                std::move(meas_fn),
                std::move(jac_fn)
            );
            af->invalidateJacobianCache();
            abs->graph->connect(af, v0, 0);

            // initial linearization
            af->computeFactor(v0->mu, true);
        } else if (k_adj == 2) {
            // between
            const int i = sup_f->adj_var_nodes[0]->variableID;
            const int j = sup_f->adj_var_nodes[1]->variableID;

            const int ni = abs->graph->var_nodes[i]->dofs; // abs dofs for i

            auto meas_fn = [abs, sup_f, i, j, ni](const Eigen::VectorXd& xij_abs) -> std::vector<Eigen::VectorXd> {
                const Eigen::VectorXd xi = xij_abs.head(ni);
                const Eigen::VectorXd xj = xij_abs.tail(xij_abs.size() - ni);

                const Eigen::MatrixXd& Bi = abs->Bs.at(i);
                const Eigen::MatrixXd& Bj = abs->Bs.at(j);
                const Eigen::VectorXd& ki = abs->ks.at(i);
                const Eigen::VectorXd& kj = abs->ks.at(j);

                Eigen::VectorXd x_sup(Bi.rows() + Bj.rows());
                x_sup.head(Bi.rows()) = Bi * xi + ki;
                x_sup.tail(Bj.rows()) = Bj * xj + kj;

                return sup_f->meas_fn(x_sup);
            };

            auto jac_fn = [abs, sup_f, i, j, ni](const Eigen::VectorXd& xij_abs) -> std::vector<Eigen::MatrixXd> {
                const Eigen::VectorXd xi = xij_abs.head(ni);
                const Eigen::VectorXd xj = xij_abs.tail(xij_abs.size() - ni);

                const Eigen::MatrixXd& Bi = abs->Bs.at(i);
                const Eigen::MatrixXd& Bj = abs->Bs.at(j);
                const Eigen::VectorXd& ki = abs->ks.at(i);
                const Eigen::VectorXd& kj = abs->ks.at(j);

                const int di_sup = (int)Bi.rows();
                const int dj_sup = (int)Bj.rows();
                const int nj = (int)xj.size();

                Eigen::VectorXd x_sup(di_sup + dj_sup);
                x_sup.head(di_sup) = Bi * xi + ki;
                x_sup.tail(dj_sup) = Bj * xj + kj;

                auto Jsup_blocks = sup_f->jac_fn(x_sup);
                std::vector<Eigen::MatrixXd> out;
                out.reserve(Jsup_blocks.size());

                for (const auto& Jsup : Jsup_blocks) {
                    // Jsup: rows x (di_sup + dj_sup)
                    Eigen::MatrixXd Jabs = Eigen::MatrixXd::Zero(Jsup.rows(), ni + nj);
                    Jabs.leftCols(ni)  = Jsup.leftCols(di_sup)  * Bi;
                    Jabs.rightCols(nj) = Jsup.rightCols(dj_sup) * Bj;
                    out.push_back(std::move(Jabs));
                }
                return out;
            };

            auto* vi = abs->graph->var_nodes[i].get();
            auto* vj = abs->graph->var_nodes[j].get();
            auto* af = abs->graph->addFactor(
                sup_f->factorID,
                std::vector<VariableNode*>{vi, vj},
                sup_f->measurement,
                sup_f->measurement_lambda,
                std::move(meas_fn),
                std::move(jac_fn)
            );
            af->invalidateJacobianCache();
            abs->graph->connect(af, vi, 0);
            abs->graph->connect(af, vj, 1);

            Eigen::VectorXd lin0(vi->dofs + vj->dofs);
            lin0.head(vi->dofs) = vi->mu;
            lin0.tail(vj->dofs) = vj->mu;
            af->computeFactor(lin0, true);
        }
        // ignore higher-order factors (Python doesn't have them here)
    }

    return abs;
}

void HierarchyGBP::bottomUpUpdateAbs(
    const std::shared_ptr<FactorGraph>& sup_fg,
    const std::shared_ptr<AbsLayer>& abs,
    int r_reduced,
    double eta_damping_abs
) {
    abs->graph->eta_damping = eta_damping_abs;

    // Stage updated k vectors to avoid concurrent writes into unordered_map.
    std::vector<Eigen::VectorXd> new_ks;
    new_ks.resize(sup_fg->var_nodes.size());

    // === 1) Update each abs variable (B fixed; match Python path A) ===
    const int n_sup_vars = static_cast<int>(sup_fg->var_nodes.size());
    #pragma omp parallel for schedule(dynamic)
    for (int vi = 0; vi < n_sup_vars; ++vi) {
        const auto& sn_uptr = sup_fg->var_nodes[vi];
        if (!sn_uptr) continue;
        const auto& sn = *sn_uptr;

        const int sid = sn.variableID;
        if (sid < 0) continue;
        if (sid >= static_cast<int>(abs->graph->var_nodes.size()) || !abs->graph->var_nodes[sid]) continue;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Fixed projection
        const Eigen::MatrixXd& B = abs->Bs.at(sid);

        const Eigen::VectorXd& mu_sup = sn.mu;
        const Eigen::MatrixXd& Sigma_sup = sn.Sigma;

        Eigen::VectorXd mu_abs = B.transpose() * mu_sup;
        Eigen::MatrixXd Sigma_abs = B.transpose() * Sigma_sup * B;

        // k := mu_sup - B * mu_abs  (store after parallel)
        new_ks[sid] = mu_sup - B * mu_abs;

        auto t1 = std::chrono::high_resolution_clock::now();
        g_bottomUpAbs_proj_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        // Convert to information form without explicit inverse
        // lam_abs = Sigma_abs^{-1}, eta_abs = lam_abs * mu_abs
        Eigen::LLT<Eigen::MatrixXd> llt(Sigma_abs);
        static thread_local Eigen::MatrixXd I;
        I.resize(Sigma_abs.rows(), Sigma_abs.cols());
        I.setIdentity();

        Eigen::MatrixXd lam_abs = llt.solve(I);
        Eigen::VectorXd eta_abs = lam_abs * mu_abs;

        auto t2 = std::chrono::high_resolution_clock::now();
        g_bottomUpAbs_llt_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();

        auto* v = abs->graph->var_nodes[sid].get();
        v->mu = mu_abs;
        v->Sigma = Sigma_abs;

        v->belief.setLam(lam_abs);
        v->belief.setEta(eta_abs);

        auto t3 = std::chrono::high_resolution_clock::now();
        g_bottomUpAbs_setbelief_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    }

    // Commit ks sequentially (thread-safe)
    for (int sid = 0; sid < static_cast<int>(new_ks.size()); ++sid) {
        if (new_ks[sid].size() == 0) continue;
        abs->ks[sid] = std::move(new_ks[sid]);
    }

    // === 2) Re-linearize all abs factors ===
    const int n_abs_factors = static_cast<int>(abs->graph->factors.size());
    #pragma omp parallel for schedule(dynamic)
    for (int fi = 0; fi < n_abs_factors; ++fi) {
        auto& fptr = abs->graph->factors[fi];
        if (!fptr) continue;
        auto& f = *fptr;
        if (!f.active) continue;

        int total = 0;
        for (auto* v : f.adj_var_nodes) total += v->dofs;

        static thread_local Eigen::VectorXd lp;
        lp.resize(total);

        int c0 = 0;
        for (auto* v : f.adj_var_nodes) {
            lp.segment(c0, v->dofs) = v->mu;
            c0 += v->dofs;
        }
        f.computeFactor(lp, true);
    }
}


// ============================================================================
// Top-down: super -> base  (Python: top_down_modify_base_and_abs_graph)
// ============================================================================
void HierarchyGBP::topDownModifyBaseFromSuper(
    const std::shared_ptr<FactorGraph>& base_fg,
    const std::shared_ptr<SuperLayer>& super)
{
    if (!base_fg || !super || !super->graph) return;

    auto& super_graph = *super->graph;

    // Build variableID -> ptr lookup for base (thread-safe read-only in OMP region).
    int max_vid = -1;
    for (const auto& up : base_fg->var_nodes) {
        if (up) max_vid = std::max(max_vid, up->variableID);
    }
    std::vector<VariableNode*> id2var_base((max_vid >= 0) ? (max_vid + 1) : 0, nullptr);
    for (auto& up : base_fg->var_nodes) {
        if (!up) continue;
        const int vid = up->variableID;
        if (vid >= 0 && vid < static_cast<int>(id2var_base.size())) {
            id2var_base[vid] = up.get();
        }
    }

    // sid -> base ids is available as super->groups
    const int n_super_vars = static_cast<int>(super_graph.var_nodes.size());
    #pragma omp parallel for schedule(dynamic)
    for (int svi = 0; svi < n_super_vars; ++svi) {
        const auto& sv_up = super_graph.var_nodes[svi];
        if (!sv_up) continue;
        const VariableNode& s_var = *sv_up;
        const int sid = s_var.variableID;

        if (sid < 0 || sid >= static_cast<int>(super->groups.size())) continue;
        const auto& base_ids = super->groups[sid];
        if (base_ids.empty()) continue;

        const Eigen::VectorXd& mu_super = s_var.mu;

        // split mu_super into base vars in group order
        int off = 0;
        for (int bid : base_ids) {
            VariableNode* v = nullptr;

            if (bid >= 0 && bid < static_cast<int>(id2var_base.size())) {
                v = id2var_base[bid];
            }
            // fallback: treat bid as index if base graph stores contiguous ids
            if (!v && bid >= 0 && bid < static_cast<int>(base_fg->var_nodes.size()) && base_fg->var_nodes[bid]) {
                v = base_fg->var_nodes[bid].get();
            }
            if (!v) continue;

            const int d = v->dofs;
            if (d <= 0) continue;
            if (off + d > mu_super.size()) break;

            const Eigen::VectorXd mu_child = mu_super.segment(off, d);
            off += d;

            // delta formulation: lam unchanged, so d_eta = lam * d_mu
            const Eigen::VectorXd mu_old = v->mu;
            v->mu = mu_child;

            const Eigen::VectorXd d_mu = v->mu - mu_old;
            if (d_mu.squaredNorm() == 0.0) {
                // still keep weak prior in sync with current belief
                const Eigen::VectorXd eta_now = v->belief.lam() * v->mu;
                v->belief.setEta(eta_now);
                v->prior.setEta(1e-10 * eta_now);
                v->prior.setLam(1e-10 * v->belief.lam());
                continue;
            }

            const Eigen::MatrixXd& lam = v->belief.lam();
            const Eigen::VectorXd d_eta = lam * d_mu;

            // update belief eta (lam unchanged)
            Eigen::VectorXd eta_new = v->belief.eta();
            eta_new.noalias() += d_eta;
            v->belief.setEta(eta_new);

            // weak prior (Python: 1e-10 * ...)
            v->prior.setEta(1e-10 * eta_new);
            v->prior.setLam(1e-10 * lam);

            // sync to adjacent factors + correct messages (eta only; lam unchanged)
            const int n_adj = static_cast<int>(v->adj_factors.size());
            if (n_adj > 0) {
                const Eigen::VectorXd d_eta_share = d_eta / static_cast<double>(n_adj);

                for (const auto& ref : v->adj_factors) {
                    Factor* f = ref.factor;
                    const int idx = ref.local_idx;
                    if (!f) continue;
                    if (idx < 0 || idx >= static_cast<int>(f->adj_beliefs.size())) continue;
                    if (idx < 0 || idx >= static_cast<int>(f->messages.size())) continue;

                    // update cached adj_beliefs
                    f->adj_beliefs[idx].setEta(v->belief.eta());
                    f->adj_beliefs[idx].setLam(v->belief.lam());

                    // correct corresponding message (eta only)
                    Eigen::VectorXd msg_eta = f->messages[idx].eta();
                    msg_eta.noalias() += d_eta_share;
                    f->messages[idx].setEta(msg_eta);
                }
            }
        }
    }
}

// ============================================================================
// Top-down: abs -> super  (Python: top_down_modify_super_graph)
// ============================================================================
void HierarchyGBP::topDownModifySuperFromAbs(
    const std::shared_ptr<FactorGraph>& sup_fg,
    const std::shared_ptr<AbsLayer>& abs)
{
    if (!sup_fg || !abs || !abs->graph) return;

    auto& abs_graph = *abs->graph;

    const int n_sup_vars = static_cast<int>(sup_fg->var_nodes.size());
    #pragma omp parallel for schedule(dynamic)
    for (int svi = 0; svi < n_sup_vars; ++svi) {
        auto& sn_up = sup_fg->var_nodes[svi];
        if (!sn_up) continue;
        VariableNode* sn = sn_up.get();
        const int sid = sn->variableID;

        auto itB = abs->Bs.find(sid);
        auto itk = abs->ks.find(sid);
        if (itB == abs->Bs.end() || itk == abs->ks.end()) continue;

        const Eigen::MatrixXd& B = itB->second; // (d_s x r)
        const Eigen::VectorXd& k = itk->second; // (d_s)

        // locate corresponding abs variable (same sid)
        VariableNode* an = nullptr;
        if (sid >= 0 && sid < static_cast<int>(abs_graph.var_nodes.size()) && abs_graph.var_nodes[sid]) {
            an = abs_graph.var_nodes[sid].get();
        } else {
            // fallback search
            for (auto& up : abs_graph.var_nodes) {
                if (up && up->variableID == sid) { an = up.get(); break; }
            }
        }
        if (!an) continue;

        // x_s = B x_a + k
        const Eigen::VectorXd mu_new = B * an->mu + k;

        // delta formulation: lam unchanged, so d_eta = lam * d_mu
        const Eigen::VectorXd mu_old = sn->mu;
        sn->mu = mu_new;

        const Eigen::VectorXd d_mu = sn->mu - mu_old;
        if (d_mu.squaredNorm() == 0.0) {
            const Eigen::VectorXd eta_now = sn->belief.lam() * sn->mu;
            sn->belief.setEta(eta_now);
            sn->prior.setEta(1e-10 * eta_now);
            sn->prior.setLam(1e-10 * sn->belief.lam());
            continue;
        }

        const Eigen::MatrixXd& lam = sn->belief.lam();
        const Eigen::VectorXd d_eta = lam * d_mu;

        // update belief eta (lam unchanged)
        Eigen::VectorXd eta_new = sn->belief.eta();
        eta_new.noalias() += d_eta;
        sn->belief.setEta(eta_new);

        // weak prior
        sn->prior.setEta(1e-10 * eta_new);
        sn->prior.setLam(1e-10 * lam);

        // sync to adjacent factors + correct messages (eta only; lam unchanged)
        const int n_adj = static_cast<int>(sn->adj_factors.size());
        if (n_adj > 0) {
            const Eigen::VectorXd d_eta_share = d_eta / static_cast<double>(n_adj);

            for (const auto& ref : sn->adj_factors) {
                Factor* f = ref.factor;
                const int idx = ref.local_idx;
                if (!f) continue;
                if (idx < 0 || idx >= static_cast<int>(f->adj_beliefs.size())) continue;
                if (idx < 0 || idx >= static_cast<int>(f->messages.size())) continue;

                f->adj_beliefs[idx].setEta(sn->belief.eta());
                f->adj_beliefs[idx].setLam(sn->belief.lam());

                Eigen::VectorXd msg_eta = f->messages[idx].eta();
                msg_eta.noalias() += d_eta_share;
                f->messages[idx].setEta(msg_eta);
            }
        }
    }
}


void HierarchyGBP::vLoop(
    std::vector<VLayerEntry>& layers,
    int r_reduced,
    double eta_damping
) {
    if (layers.empty() || !layers[0].graph) return;

    auto startsWith = [](const std::string& s, const char* prefix) -> bool {
        return s.rfind(prefix, 0) == 0;
    };

    // ---- bottom-up ----
    // Python: for i in range(1, len(layers)):
    for (size_t i = 1; i < layers.size(); ++i) {
        const std::string& name = layers[i].name;

        // mirror Python:
        // if name.startswith("super1"): pass
        if (startsWith(name, "super1")) {
            // do nothing (no bottom-up modify)
            // but we still run one iteration below if graph exists
        }
        // elif name.startswith("super"): bottom_up_modify_super_graph(...)
        else if (startsWith(name, "super")) {
            // vLoop assumes layers are already built outside.
            // So we only "modify/update", not build.
            if (layers[i].super && layers[i - 1].graph) {
                bottomUpUpdateSuper(layers[i - 1].graph, layers[i].super);
                layers[i].graph = layers[i].super->graph;
            }
        }
        // elif name.startswith("abs"): bottom_up_modify_abs_graph(...)
        else if (startsWith(name, "abs")) {
            if (layers[i].abs && layers[i - 1].graph) {
                bottomUpUpdateAbs(layers[i - 1].graph, layers[i].abs, r_reduced, eta_damping);
                layers[i].graph = layers[i].abs->graph;
            }
        }

        // After build/modify, one iteration per layer (non-base)
        if (layers[i].graph) {
            layers[i].graph->eta_damping = eta_damping;
            layers[i].graph->synchronousIteration(false);
        }
    }

    // ---- top-down ----
    // Python: for i in range(len(layers)-1, 0, -1):
    for (size_t ii = layers.size(); ii-- > 1; ) {
        // After one iteration per layer, reproject
        if (layers[ii].graph) {
            layers[ii].graph->eta_damping = eta_damping;
            layers[ii].graph->synchronousIteration(false);
        }

        const std::string& name = layers[ii].name;

        // Python:
        // if name.startswith("super"): top_down_modify_base_and_abs_graph(...)
        if (startsWith(name, "super")) {
            if (layers[ii].super && layers[ii - 1].graph) {
                topDownModifyBaseFromSuper(layers[ii - 1].graph, layers[ii].super);
            }
        }
        // elif name.startswith("abs"): top_down_modify_super_graph(...)
        else if (startsWith(name, "abs")) {
            if (layers[ii].abs && layers[ii - 1].graph) {
                topDownModifySuperFromAbs(layers[ii - 1].graph, layers[ii].abs);
            }
        }
    }
}


} // namespace gbp
