#include "slam/SyntheticSE2MG.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <omp.h>

#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

namespace slam {

namespace {

using PoseVec = std::vector<Eigen::Vector3d>;
using SteadyClock = std::chrono::steady_clock;

double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

Eigen::Matrix2d rot2(double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
    return R;
}

Eigen::Vector3d se2Compose(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    out.head<2>() = a.head<2>() + rot2(a(2)) * b.head<2>();
    out(2) = wrapAngle(a(2) + b(2));
    return out;
}

Eigen::Vector3d se2Inverse(const Eigen::Vector3d& a) {
    const Eigen::Matrix2d RT = rot2(a(2)).transpose();
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    out.head<2>() = -(RT * a.head<2>());
    out(2) = wrapAngle(-a(2));
    return out;
}

Eigen::Vector3d se2Between(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return se2Compose(se2Inverse(a), b);
}

Eigen::Vector3d se2Exp(const Eigen::Vector3d& xi) {
    const double vx = xi(0);
    const double vy = xi(1);
    const double w = xi(2);
    if (std::abs(w) < 1e-12) {
        return Eigen::Vector3d(vx, vy, 0.0);
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    Eigen::Matrix2d V;
    V << a, -b,
         b,  a;
    const Eigen::Vector2d t = V * Eigen::Vector2d(vx, vy);
    return Eigen::Vector3d(t(0), t(1), wrapAngle(w));
}

Eigen::Vector3d se2Log(const Eigen::Vector3d& pose) {
    const double tx = pose(0);
    const double ty = pose(1);
    const double w = pose(2);
    if (std::abs(w) < 1e-12) {
        return Eigen::Vector3d(tx, ty, 0.0);
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double denom = a * a + b * b;
    Eigen::Matrix2d V_inv;
    V_inv <<  a, b,
             -b, a;
    V_inv /= denom;
    const Eigen::Vector2d v = V_inv * Eigen::Vector2d(tx, ty);
    return Eigen::Vector3d(v(0), v(1), wrapAngle(w));
}

Eigen::Vector3d se2Plus(const Eigen::Vector3d& base_pose, const Eigen::Vector3d& delta) {
    return se2Compose(base_pose, se2Exp(delta));
}

Eigen::Matrix3d jacobianExpSE2(const Eigen::Vector3d& xi) {
    const double vx = xi(0);
    const double vy = xi(1);
    const double w = xi(2);

    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    if (std::abs(w) < 1e-8) {
        J.setIdentity();
        J(0, 2) = -0.5 * vy;
        J(1, 2) = 0.5 * vx;
        return J;
    }

    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double da = (w * std::cos(w) - std::sin(w)) / (w * w);
    const double db = (w * std::sin(w) - (1.0 - std::cos(w))) / (w * w);

    J(0, 0) = a;
    J(0, 1) = -b;
    J(1, 0) = b;
    J(1, 1) = a;
    J(0, 2) = da * vx - db * vy;
    J(1, 2) = db * vx + da * vy;
    J(2, 2) = 1.0;
    return J;
}

Eigen::Matrix3d jacobianPlusSE2(const Eigen::Vector3d& base_pose, const Eigen::Vector3d& delta) {
    const double c = std::cos(base_pose(2));
    const double s = std::sin(base_pose(2));
    Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
    G << c, -s, 0.0,
         s,  c, 0.0,
         0.0, 0.0, 1.0;
    return G * jacobianExpSE2(delta);
}

Eigen::Matrix<double, 3, 6> jacobianBetweenAbsolute(const Eigen::Vector3d& xi, const Eigen::Vector3d& xj) {
    const double thi = xi(2);
    const double c = std::cos(thi);
    const double s = std::sin(thi);
    Eigen::Matrix2d RT;
    RT <<  c, s,
          -s, c;

    const Eigen::Vector2d dp = xj.head<2>() - xi.head<2>();
    const Eigen::Vector2d r = RT * dp;
    const Eigen::Vector2d dr_dthi(r(1), -r(0));

    Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
    J.block<2, 2>(0, 0) = -RT;
    J.block<2, 1>(0, 2) = dr_dthi;
    J.block<2, 2>(0, 3) = RT;
    J(2, 2) = -1.0;
    J(2, 5) = 1.0;
    return J;
}

Eigen::Matrix3d jacobianComposeInvConstant(const Eigen::Vector3d& z) {
    const double c = std::cos(z(2));
    const double s = std::sin(z(2));
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    J(0, 0) = c;
    J(0, 1) = s;
    J(1, 0) = -s;
    J(1, 1) = c;
    J(2, 2) = 1.0;
    return J;
}

Eigen::Matrix3d jacobianLogSE2(const Eigen::Vector3d& pose) {
    const double tx = pose(0);
    const double ty = pose(1);
    const double w = pose(2);

    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    if (std::abs(w) < 1e-8) {
        J.setIdentity();
        J(0, 2) = 0.5 * ty;
        J(1, 2) = -0.5 * tx;
        return J;
    }

    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double den = a * a + b * b;
    const double da = (w * std::cos(w) - std::sin(w)) / (w * w);
    const double db = (w * std::sin(w) - (1.0 - std::cos(w))) / (w * w);
    const double dden = 2.0 * (a * da + b * db);

    const double c = a / den;
    const double d = b / den;
    const double dc = (da * den - a * dden) / (den * den);
    const double dd = (db * den - b * dden) / (den * den);

    J(0, 0) = c;
    J(0, 1) = d;
    J(1, 0) = -d;
    J(1, 1) = c;
    J(0, 2) = dc * tx + dd * ty;
    J(1, 2) = -dd * tx + dc * ty;
    J(2, 2) = 1.0;
    return J;
}

Eigen::Matrix<double, 3, 6> analyticEdgeResidualJacobian(
    const Eigen::Vector3d& base_i,
    const Eigen::Vector3d& base_j,
    const Eigen::Vector3d& z,
    const Eigen::Vector3d& ei,
    const Eigen::Vector3d& ej
) {
    const Eigen::Vector3d xi = se2Plus(base_i, ei);
    const Eigen::Vector3d xj = se2Plus(base_j, ej);
    const Eigen::Vector3d pred = se2Between(xi, xj);
    const Eigen::Vector3d err_pose = se2Compose(se2Inverse(z), pred);

    const Eigen::Matrix<double, 3, 6> J_between_abs = jacobianBetweenAbsolute(xi, xj);
    const Eigen::Matrix3d J_compose = jacobianComposeInvConstant(z);
    const Eigen::Matrix3d J_log = jacobianLogSE2(err_pose);

    Eigen::Matrix<double, 6, 6> J_plus = Eigen::Matrix<double, 6, 6>::Zero();
    J_plus.block<3, 3>(0, 0) = jacobianPlusSE2(base_i, ei);
    J_plus.block<3, 3>(3, 3) = jacobianPlusSE2(base_j, ej);

    return J_log * J_compose * J_between_abs * J_plus;
}

Eigen::Matrix3d analyticAnchorResidualJacobian(
    const Eigen::Vector3d& base_anchor,
    const Eigen::Vector3d& anchor_pose,
    const Eigen::Vector3d& ei
) {
    const Eigen::Vector3d xi = se2Plus(base_anchor, ei);
    const Eigen::Vector3d err_pose = se2Compose(se2Inverse(anchor_pose), xi);
    return jacobianLogSE2(err_pose) * jacobianComposeInvConstant(anchor_pose) * jacobianPlusSE2(base_anchor, ei);
}

Eigen::VectorXd stackedMeanVector(gbp::FactorGraph& graph) {
    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(total_dim);
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        vup->refreshMu();
        out.segment(offset, vup->dofs) = vup->mu;
        offset += vup->dofs;
    }
    return out;
}

void injectCorrectionKeepMessages(gbp::FactorGraph& graph, const Eigen::VectorXd& delta) {
    const Eigen::VectorXd x_new = stackedMeanVector(graph) + delta;
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        gbp::VariableNode& var = *vup;
        var.mu = x_new.segment(offset, var.dofs);
        var.belief.setEta(var.belief.lam() * var.mu);
        var.markMuCurrent();
        offset += var.dofs;
    }
}

Eigen::SparseMatrix<double> symmetrizeSparse(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> sym = 0.5 * (A + Eigen::SparseMatrix<double>(A.transpose()));
    sym.makeCompressed();
    return sym;
}

Eigen::VectorXd solveSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    const Eigen::VectorXd& eta,
    double ridge
) {
    if (lam.rows() == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    Eigen::SparseMatrix<double> A = symmetrizeSparse(lam);
    if (ridge != 0.0) {
        for (int i = 0; i < A.rows(); ++i) {
            A.coeffRef(i, i) += ridge;
        }
    }
    A.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }

    const Eigen::VectorXd x = solver.solve(eta);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
    return x;
}

std::vector<std::vector<int>> orderedGroups(int n_vars, int group_size) {
    if (group_size <= 0) {
        throw std::runtime_error("group_size must be positive");
    }
    std::vector<std::vector<int>> groups;
    int start = 0;
    while (start + 2 * group_size <= n_vars) {
        std::vector<int> group;
        group.reserve(group_size);
        for (int id = start; id < start + group_size; ++id) {
            group.push_back(id);
        }
        groups.push_back(std::move(group));
        start += group_size;
    }
    if (start < n_vars) {
        std::vector<int> tail;
        tail.reserve(n_vars - start);
        for (int id = start; id < n_vars; ++id) {
            tail.push_back(id);
        }
        groups.push_back(std::move(tail));
    }
    return groups;
}

Eigen::MatrixXd buildGroupMessageConditionedInformation(
    const gbp::FactorGraph& graph,
    const std::vector<int>& group_var_ids
) {
    std::unordered_map<int, int> local_offset;
    int total_dim = 0;
    for (int var_id : group_var_ids) {
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        local_offset[var_id] = total_dim;
        total_dim += var->dofs;
    }

    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(total_dim, total_dim);
    std::set<int> handled_internal_factors;

    for (int var_id : group_var_ids) {
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        const int off = local_offset.at(var_id);
        info.block(off, off, var->dofs, var->dofs).noalias() +=
            0.5 * (var->prior.lam() + var->prior.lam().transpose());

        for (const auto& aref : var->adj_factors) {
            const gbp::Factor* factor = aref.factor;
            if (!factor || !factor->active) {
                continue;
            }

            bool factor_inside = true;
            for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                if (local_offset.find(adj_var->variableID) == local_offset.end()) {
                    factor_inside = false;
                    break;
                }
            }

            if (factor_inside) {
                if (!handled_internal_factors.insert(factor->factorID).second) {
                    continue;
                }
                int factor_row = 0;
                for (const gbp::VariableNode* adj_i : factor->adj_var_nodes) {
                    const int oi = local_offset.at(adj_i->variableID);
                    int factor_col = 0;
                    for (const gbp::VariableNode* adj_j : factor->adj_var_nodes) {
                        const int oj = local_offset.at(adj_j->variableID);
                        info.block(oi, oj, adj_i->dofs, adj_j->dofs).noalias() +=
                            factor->factor.lam().block(factor_row, factor_col, adj_i->dofs, adj_j->dofs);
                        factor_col += adj_j->dofs;
                    }
                    factor_row += adj_i->dofs;
                }
            } else {
                const int msg_ix = aref.local_idx;
                info.block(off, off, var->dofs, var->dofs).noalias() +=
                    0.5 * (factor->messages[msg_ix].lam() + factor->messages[msg_ix].lam().transpose());
            }
        }
    }

    return 0.5 * (info + info.transpose());
}

struct BasisData {
    std::vector<std::vector<int>> groups;
    std::vector<std::vector<int>> full_indices_per_group;
    std::vector<Eigen::MatrixXd> local_bases;
    std::vector<int> coarse_offsets;
    std::vector<int> var_to_group;
    std::vector<int> var_to_local_offset;
    int total_dim = 0;
    int coarse_dim = 0;
};

BasisData buildMessageConditionedBasis(
    const gbp::FactorGraph& graph,
    int group_size,
    int r_reduced
) {
    BasisData basis;
    basis.groups = orderedGroups(static_cast<int>(graph.var_nodes.size()), group_size);

    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    basis.total_dim = total_dim;

    basis.coarse_offsets.push_back(0);
    basis.var_to_group.assign(graph.var_nodes.size(), -1);
    basis.var_to_local_offset.assign(graph.var_nodes.size(), -1);
    int reduced_offset = 0;

    for (const std::vector<int>& group : basis.groups) {
        std::vector<int> full_indices;
        full_indices.reserve(group.size() * 3);
        int local_scalar_offset = 0;
        for (int var_id : group) {
            const int base = 3 * var_id;
            full_indices.push_back(base + 0);
            full_indices.push_back(base + 1);
            full_indices.push_back(base + 2);
            basis.var_to_group[var_id] = static_cast<int>(basis.full_indices_per_group.size());
            basis.var_to_local_offset[var_id] = local_scalar_offset;
            local_scalar_offset += 3;
        }

        const Eigen::MatrixXd block = buildGroupMessageConditionedInformation(graph, group);
        const int block_dim = static_cast<int>(full_indices.size());
        const int r_local = std::min(r_reduced, block_dim);

        Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
        if (r_local < block_dim) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(block);
            if (es.info() != Eigen::Success) {
                throw std::runtime_error("Failed to eigendecompose conditioned information block");
            }
            const Eigen::MatrixXd evecs = es.eigenvectors();
            local_basis.resize(block_dim, r_local);
            for (int col = 0; col < r_local; ++col) {
                local_basis.col(col) = evecs.col(col);
            }
        }

        basis.full_indices_per_group.push_back(full_indices);
        basis.local_bases.push_back(local_basis);
        reduced_offset += r_local;
        basis.coarse_offsets.push_back(reduced_offset);
    }

    basis.coarse_dim = reduced_offset;
    return basis;
}

Eigen::VectorXd restrictToCoarse(const BasisData& basis, const Eigen::VectorXd& fine_vec) {
    Eigen::VectorXd coarse = Eigen::VectorXd::Zero(basis.coarse_dim);
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const std::vector<int>& rows = basis.full_indices_per_group[g];
        Eigen::VectorXd fine_local(rows.size());
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            fine_local(i) = fine_vec(rows[i]);
        }
        const int coarse_offset = basis.coarse_offsets[g];
        const int coarse_dim = basis.local_bases[g].cols();
        coarse.segment(coarse_offset, coarse_dim) = basis.local_bases[g].transpose() * fine_local;
    }
    return coarse;
}

Eigen::VectorXd prolongToFine(const BasisData& basis, const Eigen::VectorXd& coarse_vec) {
    Eigen::VectorXd fine = Eigen::VectorXd::Zero(basis.total_dim);
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const std::vector<int>& rows = basis.full_indices_per_group[g];
        const int coarse_offset = basis.coarse_offsets[g];
        const int coarse_dim = basis.local_bases[g].cols();
        const Eigen::VectorXd fine_local =
            basis.local_bases[g] * coarse_vec.segment(coarse_offset, coarse_dim);
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            fine(rows[i]) = fine_local(i);
        }
    }
    return fine;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirect(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    bool use_parallel,
    int num_threads
) {
    const int num_groups = static_cast<int>(basis.local_bases.size());
    const int num_block_pairs = num_groups * num_groups;

    std::vector<Eigen::MatrixXd> blocks(num_block_pairs);
    std::vector<char> active(num_block_pairs, 0);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int group = basis.var_to_group[var.variableID];
        if (group < 0) {
            continue;
        }
        const int local_offset = basis.var_to_local_offset[var.variableID];
        const Eigen::MatrixXd basis_block = basis.local_bases[group].middleRows(local_offset, var.dofs);
        const int block_index = group * num_groups + group;
        if (!active[block_index]) {
            blocks[block_index] = Eigen::MatrixXd::Zero(basis_block.cols(), basis_block.cols());
            active[block_index] = 1;
        }
        blocks[block_index].noalias() += basis_block.transpose() * var.prior.lam() * basis_block;
    }

    auto accumulate_factor = [&](const gbp::Factor& factor, std::vector<Eigen::MatrixXd>& target_blocks, std::vector<char>& target_active) {
        if (!factor.active) {
            return;
        }

        std::vector<int> local_factor_offsets(factor.adj_var_nodes.size(), 0);
        int factor_offset = 0;
        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            local_factor_offsets[a] = factor_offset;
            factor_offset += factor.adj_var_nodes[a]->dofs;
        }

        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int ga = basis.var_to_group[va->variableID];
            const int la = basis.var_to_local_offset[va->variableID];
            const int da = va->dofs;
            const Eigen::MatrixXd Ba = basis.local_bases[ga].middleRows(la, da);
            const int off_a = local_factor_offsets[a];

            for (int b = 0; b < static_cast<int>(factor.adj_var_nodes.size()); ++b) {
                const gbp::VariableNode* vb = factor.adj_var_nodes[b];
                const int gb = basis.var_to_group[vb->variableID];
                const int lb = basis.var_to_local_offset[vb->variableID];
                const int db = vb->dofs;
                const Eigen::MatrixXd Bb = basis.local_bases[gb].middleRows(lb, db);
                const int off_b = local_factor_offsets[b];
                const int block_index = ga * num_groups + gb;

                if (!target_active[block_index]) {
                    target_blocks[block_index] = Eigen::MatrixXd::Zero(Ba.cols(), Bb.cols());
                    target_active[block_index] = 1;
                }

                target_blocks[block_index].noalias() +=
                    Ba.transpose() *
                    factor.factor.lam().block(off_a, off_b, da, db) *
                    Bb;
            }
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        std::vector<std::vector<Eigen::MatrixXd>> tls_blocks(thread_count);
        std::vector<std::vector<char>> tls_active(thread_count);
        for (int tid = 0; tid < thread_count; ++tid) {
            tls_blocks[tid].resize(num_block_pairs);
            tls_active[tid].assign(num_block_pairs, 0);
        }

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor(*factor, tls_blocks[tid], tls_active[tid]);
        }

        for (int tid = 0; tid < thread_count; ++tid) {
            for (int block_index = 0; block_index < num_block_pairs; ++block_index) {
                if (!tls_active[tid][block_index]) {
                    continue;
                }
                if (!active[block_index]) {
                    blocks[block_index] = std::move(tls_blocks[tid][block_index]);
                    active[block_index] = 1;
                } else {
                    blocks[block_index] += tls_blocks[tid][block_index];
                }
            }
        }
    } else {
        for (const auto& fup : graph.factors) {
            if (!fup) {
                continue;
            }
            accumulate_factor(*fup, blocks, active);
        }
    }

    std::vector<Eigen::Triplet<double>> trips;
    for (int gi = 0; gi < num_groups; ++gi) {
        const int row_offset = basis.coarse_offsets[gi];
        for (int gj = 0; gj < num_groups; ++gj) {
            const int block_index = gi * num_groups + gj;
            if (!active[block_index]) {
                continue;
            }
            const int col_offset = basis.coarse_offsets[gj];
            const Eigen::MatrixXd& block = blocks[block_index];
            for (int r = 0; r < block.rows(); ++r) {
                for (int c = 0; c < block.cols(); ++c) {
                    const double value = block(r, c);
                    if (value != 0.0) {
                        trips.emplace_back(row_offset + r, col_offset + c, value);
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<double> coarse_lam(basis.coarse_dim, basis.coarse_dim);
    coarse_lam.setFromTriplets(trips.begin(), trips.end());
    coarse_lam.makeCompressed();
    return symmetrizeSparse(coarse_lam);
}

Eigen::VectorXd assembleCoarseResidualDirect(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    bool use_parallel,
    int num_threads
) {
    Eigen::VectorXd coarse = Eigen::VectorXd::Zero(basis.coarse_dim);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int group = basis.var_to_group[var.variableID];
        if (group < 0) {
            continue;
        }
        const int local_offset = basis.var_to_local_offset[var.variableID];
        const Eigen::MatrixXd basis_block = basis.local_bases[group].middleRows(local_offset, var.dofs);
        const Eigen::VectorXd x_local = fine_vec.segment(3 * var.variableID, var.dofs);
        coarse.segment(basis.coarse_offsets[group], basis.local_bases[group].cols()).noalias() +=
            basis_block.transpose() * (var.prior.eta() - var.prior.lam() * x_local);
    }

    auto accumulate_factor_residual = [&](const gbp::Factor& factor, Eigen::VectorXd& target) {
        if (!factor.active) {
            return;
        }

        const int arity = static_cast<int>(factor.adj_var_nodes.size());
        if (arity == 1) {
            const gbp::VariableNode* v0 = factor.adj_var_nodes[0];
            const int g0 = basis.var_to_group[v0->variableID];
            const int l0 = basis.var_to_local_offset[v0->variableID];
            const int d0 = v0->dofs;
            const Eigen::MatrixXd B0 = basis.local_bases[g0].middleRows(l0, d0);
            const Eigen::VectorXd x0 = fine_vec.segment(3 * v0->variableID, d0);
            const Eigen::VectorXd r0 =
                factor.factor.eta().segment(0, d0) -
                factor.factor.lam().block(0, 0, d0, d0) * x0;
            target.segment(basis.coarse_offsets[g0], B0.cols()).noalias() += B0.transpose() * r0;
            return;
        }

        if (arity == 2) {
            const gbp::VariableNode* v0 = factor.adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor.adj_var_nodes[1];
            const int d0 = v0->dofs;
            const int d1 = v1->dofs;

            const int g0 = basis.var_to_group[v0->variableID];
            const int g1 = basis.var_to_group[v1->variableID];
            const int l0 = basis.var_to_local_offset[v0->variableID];
            const int l1 = basis.var_to_local_offset[v1->variableID];

            const Eigen::MatrixXd B0 = basis.local_bases[g0].middleRows(l0, d0);
            const Eigen::MatrixXd B1 = basis.local_bases[g1].middleRows(l1, d1);
            const Eigen::VectorXd x0 = fine_vec.segment(3 * v0->variableID, d0);
            const Eigen::VectorXd x1 = fine_vec.segment(3 * v1->variableID, d1);

            const Eigen::VectorXd r0 =
                factor.factor.eta().segment(0, d0) -
                factor.factor.lam().block(0, 0, d0, d0) * x0 -
                factor.factor.lam().block(0, d0, d0, d1) * x1;
            const Eigen::VectorXd r1 =
                factor.factor.eta().segment(d0, d1) -
                factor.factor.lam().block(d0, 0, d1, d0) * x0 -
                factor.factor.lam().block(d0, d0, d1, d1) * x1;

            target.segment(basis.coarse_offsets[g0], B0.cols()).noalias() += B0.transpose() * r0;
            target.segment(basis.coarse_offsets[g1], B1.cols()).noalias() += B1.transpose() * r1;
            return;
        }

        std::vector<int> local_factor_offsets(arity, 0);
        int factor_dim = 0;
        for (int a = 0; a < arity; ++a) {
            local_factor_offsets[a] = factor_dim;
            factor_dim += factor.adj_var_nodes[a]->dofs;
        }

        Eigen::VectorXd factor_x = Eigen::VectorXd::Zero(factor_dim);
        for (int a = 0; a < arity; ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            factor_x.segment(local_factor_offsets[a], va->dofs) = fine_vec.segment(3 * va->variableID, va->dofs);
        }
        const Eigen::VectorXd factor_residual = factor.factor.eta() - factor.factor.lam() * factor_x;

        for (int a = 0; a < arity; ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int group = basis.var_to_group[va->variableID];
            const int local_offset = basis.var_to_local_offset[va->variableID];
            const int dofs = va->dofs;
            const Eigen::MatrixXd basis_block = basis.local_bases[group].middleRows(local_offset, dofs);
            target.segment(basis.coarse_offsets[group], basis.local_bases[group].cols()).noalias() +=
                basis_block.transpose() * factor_residual.segment(local_factor_offsets[a], dofs);
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        std::vector<Eigen::VectorXd> tls(thread_count, Eigen::VectorXd::Zero(basis.coarse_dim));

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor_residual(*factor, tls[tid]);
        }

        for (const Eigen::VectorXd& local : tls) {
            coarse += local;
        }
    } else {
        for (const auto& fup : graph.factors) {
            if (!fup) {
                continue;
            }
            accumulate_factor_residual(*fup, coarse);
        }
    }

    return coarse;
}

std::string jsonNumber(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

}  // namespace

SyntheticSE2Problem loadSyntheticSE2Problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open synthetic SE2 problem file: " + path);
    }

    std::string token;
    in >> token;
    if (token != "SYNTHETIC_SE2_PROBLEM") {
        throw std::runtime_error("Unexpected problem header in " + path);
    }

    int n = 0;
    in >> token >> n;
    if (token != "N") {
        throw std::runtime_error("Expected N section in " + path);
    }

    SyntheticSE2Problem problem;
    problem.gt_poses.resize(n, Eigen::Vector3d::Zero());
    problem.init_poses.resize(n, Eigen::Vector3d::Zero());

    in >> token;
    if (token != "POSES") {
        throw std::runtime_error("Expected POSES section in " + path);
    }
    for (int idx = 0; idx < n; ++idx) {
        int pose_id = -1;
        in >> pose_id
           >> problem.gt_poses[idx](0) >> problem.gt_poses[idx](1) >> problem.gt_poses[idx](2)
           >> problem.init_poses[idx](0) >> problem.init_poses[idx](1) >> problem.init_poses[idx](2);
        if (pose_id != idx) {
            throw std::runtime_error("Pose ids must be contiguous in synthetic problem file");
        }
    }

    in >> token;
    if (token != "ANCHOR") {
        throw std::runtime_error("Expected ANCHOR section in " + path);
    }
    in >> problem.anchor_pose(0) >> problem.anchor_pose(1) >> problem.anchor_pose(2);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            in >> problem.anchor_information(r, c);
        }
    }

    int num_edges = 0;
    in >> token >> num_edges;
    if (token != "EDGES") {
        throw std::runtime_error("Expected EDGES section in " + path);
    }
    problem.edges.reserve(num_edges);
    for (int e = 0; e < num_edges; ++e) {
        SyntheticSE2Edge edge;
        in >> edge.i >> edge.j >> edge.kind
           >> edge.measurement(0) >> edge.measurement(1) >> edge.measurement(2);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                in >> edge.information(r, c);
            }
        }
        problem.edges.push_back(edge);
    }

    return problem;
}

double nonlinearObjective(const SyntheticSE2Problem& problem, const std::vector<Eigen::Vector3d>& poses) {
    double total = 0.0;
    for (const SyntheticSE2Edge& edge : problem.edges) {
        const Eigen::Vector3d pred = se2Between(poses.at(edge.i), poses.at(edge.j));
        const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(edge.measurement), pred));
        total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
    }
    const Eigen::Vector3d anchor_err = se2Log(se2Compose(se2Inverse(problem.anchor_pose), poses.at(0)));
    total += 0.5 * (anchor_err.transpose() * problem.anchor_information * anchor_err)(0, 0);
    return total;
}

std::vector<Eigen::Vector3d> applyPoseDeltas(
    const std::vector<Eigen::Vector3d>& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    std::vector<Eigen::Vector3d> out(base_poses.size(), Eigen::Vector3d::Zero());
    for (int i = 0; i < static_cast<int>(base_poses.size()); ++i) {
        out[i] = se2Plus(base_poses[i], delta_vec.segment<3>(3 * i));
    }
    return out;
}

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    double tiny_prior
) {
    gbp::FactorGraph graph;
    graph.nonlinear_factors = false;
    graph.eta_damping = 0.0;

    const int n = static_cast<int>(problem.gt_poses.size());
    for (int i = 0; i < n; ++i) {
        gbp::VariableNode* var = graph.addVariable(i, 3);
        var->GT = problem.gt_poses[i];
        var->prior.setLam(tiny_prior * Eigen::Matrix3d::Identity());
        var->prior.setEta(Eigen::Vector3d::Zero());
        var->mu = Eigen::Vector3d::Zero();
    }

    int factor_id = 0;
    for (const SyntheticSE2Edge& edge : problem.edges) {
        gbp::VariableNode* vi = graph.var_nodes.at(edge.i).get();
        gbp::VariableNode* vj = graph.var_nodes.at(edge.j).get();
        const Eigen::Vector3d base_i = base_poses.at(edge.i);
        const Eigen::Vector3d base_j = base_poses.at(edge.j);
        const Eigen::Vector3d z = edge.measurement;
        const Eigen::Matrix3d omega = edge.information;

        auto meas_fn = [base_i, base_j, z](const Eigen::VectorXd& x) -> std::vector<Eigen::VectorXd> {
            const Eigen::Vector3d ei = x.segment<3>(0);
            const Eigen::Vector3d ej = x.segment<3>(3);
            const Eigen::Vector3d xi = se2Plus(base_i, ei);
            const Eigen::Vector3d xj = se2Plus(base_j, ej);
            const Eigen::Vector3d pred = se2Between(xi, xj);
            const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(z), pred));
            return {err};
        };
        auto jac_fn = [base_i, base_j, z](const Eigen::VectorXd& x) -> std::vector<Eigen::MatrixXd> {
            const Eigen::Vector3d ei = x.segment<3>(0);
            const Eigen::Vector3d ej = x.segment<3>(3);
            return {analyticEdgeResidualJacobian(base_i, base_j, z, ei, ej)};
        };

        gbp::Factor* factor = graph.addFactor(
            factor_id++,
            std::vector<gbp::VariableNode*>{vi, vj},
            std::vector<Eigen::VectorXd>{Eigen::Vector3d::Zero()},
            std::vector<Eigen::MatrixXd>{omega},
            meas_fn,
            jac_fn
        );
        factor->computeFactor(Eigen::VectorXd::Zero(6), true);
        graph.connect(factor, vi, 0);
        graph.connect(factor, vj, 1);
    }

    {
        gbp::VariableNode* v0 = graph.var_nodes.at(0).get();
        const Eigen::Vector3d base_anchor = base_poses.at(0);
        const Eigen::Vector3d anchor_pose = problem.anchor_pose;
        const Eigen::Matrix3d anchor_info = problem.anchor_information;

        auto meas_fn = [base_anchor, anchor_pose](const Eigen::VectorXd& x) -> std::vector<Eigen::VectorXd> {
            const Eigen::Vector3d ei = x.segment<3>(0);
            const Eigen::Vector3d xi = se2Plus(base_anchor, ei);
            const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(anchor_pose), xi));
            return {err};
        };
        auto jac_fn = [base_anchor, anchor_pose](const Eigen::VectorXd& x) -> std::vector<Eigen::MatrixXd> {
            const Eigen::Vector3d ei = x.segment<3>(0);
            return {analyticAnchorResidualJacobian(base_anchor, anchor_pose, ei)};
        };

        gbp::Factor* anchor = graph.addFactor(
            factor_id++,
            std::vector<gbp::VariableNode*>{v0},
            std::vector<Eigen::VectorXd>{Eigen::Vector3d::Zero()},
            std::vector<Eigen::MatrixXd>{anchor_info},
            meas_fn,
            jac_fn
        );
        anchor->computeFactor(Eigen::VectorXd::Zero(3), true);
        graph.connect(anchor, v0, 0);
    }

    for (auto& vup : graph.var_nodes) {
        if (vup) {
            vup->updateBelief();
        }
    }
    return graph;
}

ExperimentResults runSyntheticSE2Experiment(
    const SyntheticSE2Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    bool strict_compare,
    int sync_num_threads
) {
    ExperimentResults results;
    const bool compute_exact_reference = strict_compare;
    results.num_poses = static_cast<int>(problem.gt_poses.size());
    results.num_edges = static_cast<int>(problem.edges.size());
    results.initial_objective = nonlinearObjective(problem, problem.init_poses);

    PoseVec direct_poses = problem.init_poses;
    results.direct_history.push_back(OuterDirectRow{0, nonlinearObjective(problem, direct_poses), 0.0, 0.0});
    if (compute_exact_reference) {
        for (int outer = 1; outer <= num_outer; ++outer) {
            const auto outer_t0 = SteadyClock::now();

            const auto build_t0 = SteadyClock::now();
            gbp::FactorGraph local_graph = buildLinearizedResidualGraph(problem, direct_poses);
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            gbp::FactorGraph::JointInfResult J = local_graph.jointDistributionInfSparse();
            const Eigen::SparseMatrix<double> lam = symmetrizeSparse(J.lam);
            const auto joint_t1 = SteadyClock::now();

            const auto solve_t0 = SteadyClock::now();
            const Eigen::VectorXd e_star = solveSparseCholesky(lam, J.eta, 0.0);
            const auto solve_t1 = SteadyClock::now();

            const auto apply_t0 = SteadyClock::now();
            direct_poses = applyPoseDeltas(direct_poses, e_star);
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = nonlinearObjective(problem, direct_poses);
            const auto obj_t1 = SteadyClock::now();
            const auto outer_t1 = SteadyClock::now();
            results.direct_history.push_back(
                OuterDirectRow{
                    outer,
                    nonlinear_obj,
                    e_star.norm(),
                    (J.eta - lam * e_star).norm(),
                    elapsedSeconds(outer_t0, outer_t1),
                    elapsedSeconds(build_t0, build_t1),
                    elapsedSeconds(joint_t0, joint_t1),
                    elapsedSeconds(solve_t0, solve_t1),
                    elapsedSeconds(apply_t0, apply_t1),
                    elapsedSeconds(obj_t0, obj_t1),
                }
            );
        }
    }

    PoseVec mg_poses = problem.init_poses;
    results.mg_history.push_back(OuterMGRow{0, nonlinearObjective(problem, mg_poses), 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0});
    for (int outer = 1; outer <= num_outer; ++outer) {
        const auto outer_t0 = SteadyClock::now();

        const auto build_t0 = SteadyClock::now();
        gbp::FactorGraph residual_graph = buildLinearizedResidualGraph(problem, mg_poses);
        residual_graph.setStrictCompareMode(strict_compare);
        residual_graph.setSyncNumThreads(sync_num_threads);
        residual_graph.setSyncUpdateMeans(false);
        residual_graph.setProfileSyncTiming(compute_exact_reference && sync_num_threads == 1);
        residual_graph.resetSyncTiming();
        const auto build_t1 = SteadyClock::now();

        const auto basis_t0 = SteadyClock::now();
        const BasisData basis = buildMessageConditionedBasis(residual_graph, group_size, r_reduced);
        const auto basis_t1 = SteadyClock::now();

        const auto coarse_lam_t0 = SteadyClock::now();
        const Eigen::SparseMatrix<double> coarse_lam = assembleCoarseLambdaDirect(
            residual_graph,
            basis,
            !strict_compare,
            sync_num_threads
        );
        const auto coarse_lam_t1 = SteadyClock::now();

        Eigen::SparseMatrix<double> lam;
        Eigen::VectorXd eta;
        Eigen::VectorXd e_star;
        double exact_joint_assembly_sec = 0.0;
        double exact_solve_sec = 0.0;
        if (compute_exact_reference) {
            const auto joint_t0 = SteadyClock::now();
            gbp::FactorGraph::JointInfResult J = residual_graph.jointDistributionInfSparse();
            lam = symmetrizeSparse(J.lam);
            eta = std::move(J.eta);
            const auto joint_t1 = SteadyClock::now();
            exact_joint_assembly_sec = elapsedSeconds(joint_t0, joint_t1);

            const auto solve_t0 = SteadyClock::now();
            e_star = solveSparseCholesky(lam, eta, 0.0);
            const auto solve_t1 = SteadyClock::now();
            exact_solve_sec = elapsedSeconds(solve_t0, solve_t1);
        }

        double sweeps_sec = 0.0;
        double coarse_eta_sec = 0.0;
        double coarse_solve_sec = 0.0;
        double prolong_inject_sec = 0.0;
        for (int cyc = 0; cyc < inner_cycles; ++cyc) {
            const auto sweeps_t0 = SteadyClock::now();
            for (int sweep = 0; sweep < pre_sweeps; ++sweep) {
                residual_graph.synchronousIteration();
            }
            const auto sweeps_t1 = SteadyClock::now();
            sweeps_sec += elapsedSeconds(sweeps_t0, sweeps_t1);

            const Eigen::VectorXd e_now = stackedMeanVector(residual_graph);
            const auto coarse_eta_t0 = SteadyClock::now();
            const Eigen::VectorXd coarse_eta = compute_exact_reference
                ? restrictToCoarse(basis, eta - lam * e_now)
                : assembleCoarseResidualDirect(residual_graph, basis, e_now, true, sync_num_threads);
            const auto coarse_eta_t1 = SteadyClock::now();
            coarse_eta_sec += elapsedSeconds(coarse_eta_t0, coarse_eta_t1);

            const auto coarse_solve_t0 = SteadyClock::now();
            const Eigen::VectorXd delta_z = solveSparseCholesky(coarse_lam, coarse_eta, 1e-10);
            const auto coarse_solve_t1 = SteadyClock::now();
            coarse_solve_sec += elapsedSeconds(coarse_solve_t0, coarse_solve_t1);

            const auto inject_t0 = SteadyClock::now();
            injectCorrectionKeepMessages(residual_graph, prolongToFine(basis, delta_z));
            const auto inject_t1 = SteadyClock::now();
            prolong_inject_sec += elapsedSeconds(inject_t0, inject_t1);
        }

        const Eigen::VectorXd e_hat = stackedMeanVector(residual_graph);
        const auto apply_t0 = SteadyClock::now();
        mg_poses = applyPoseDeltas(mg_poses, e_hat);
        const auto apply_t1 = SteadyClock::now();
        const auto obj_t0 = SteadyClock::now();
        const double nonlinear_obj = nonlinearObjective(problem, mg_poses);
        const auto obj_t1 = SteadyClock::now();
        const auto outer_t1 = SteadyClock::now();
        results.mg_history.push_back(
            OuterMGRow{
                outer,
                nonlinear_obj,
                e_hat.norm(),
                compute_exact_reference ? e_star.norm() : std::numeric_limits<double>::quiet_NaN(),
                compute_exact_reference ? (e_hat - e_star).norm() / std::max(e_star.norm(), 1e-15) : std::numeric_limits<double>::quiet_NaN(),
                compute_exact_reference ? (eta - lam * e_star).norm() : std::numeric_limits<double>::quiet_NaN(),
                compute_exact_reference ? (eta - lam * e_hat).norm() : std::numeric_limits<double>::quiet_NaN(),
                static_cast<int>(basis.groups.size()),
                basis.coarse_dim,
                elapsedSeconds(outer_t0, outer_t1),
                elapsedSeconds(build_t0, build_t1),
                elapsedSeconds(basis_t0, basis_t1),
                elapsedSeconds(coarse_lam_t0, coarse_lam_t1),
                exact_joint_assembly_sec,
                exact_solve_sec,
                sweeps_sec,
                residual_graph.sync_factor_pass_sec_accum,
                residual_graph.sync_variable_pass_sec_accum,
                coarse_eta_sec,
                coarse_solve_sec,
                prolong_inject_sec,
                elapsedSeconds(apply_t0, apply_t1),
                elapsedSeconds(obj_t0, obj_t1),
            }
        );
    }

    return results;
}

void writeExperimentResultsJson(
    const ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    bool strict_compare,
    int sync_num_threads
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output JSON path: " + path);
    }
    const bool compute_exact_reference = strict_compare;

    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"num_outer\": " << num_outer << ",\n";
    out << "    \"inner_cycles\": " << inner_cycles << ",\n";
    out << "    \"pre_sweeps\": " << pre_sweeps << ",\n";
    out << "    \"group_size\": " << group_size << ",\n";
    out << "    \"r_reduced\": " << r_reduced << ",\n";
    out << "    \"exact_reference_enabled\": " << (compute_exact_reference ? "true" : "false") << ",\n";
    out << "    \"strict_compare\": " << (strict_compare ? "true" : "false") << ",\n";
    out << "    \"sync_num_threads\": " << sync_num_threads << ",\n";
    out << "    \"basis_source\": \"message_conditioned_information\"\n";
    out << "  },\n";
    out << "  \"problem\": {\n";
    out << "    \"num_poses\": " << results.num_poses << ",\n";
    out << "    \"num_edges\": " << results.num_edges << "\n";
    out << "  },\n";
    out << "  \"initial_objective\": " << jsonNumber(results.initial_objective) << ",\n";

    out << "  \"direct_history\": [\n";
    for (size_t i = 0; i < results.direct_history.size(); ++i) {
        const OuterDirectRow& row = results.direct_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"linear_step_norm\": " << jsonNumber(row.linear_step_norm)
            << ", \"linear_residual_norm\": " << jsonNumber(row.linear_residual_norm)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"joint_assembly_sec\": " << jsonNumber(row.joint_assembly_sec)
            << ", \"exact_solve_sec\": " << jsonNumber(row.exact_solve_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << "}";
        out << (i + 1 == results.direct_history.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"mg_history\": [\n";
    for (size_t i = 0; i < results.mg_history.size(); ++i) {
        const OuterMGRow& row = results.mg_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"e_hat_norm\": " << jsonNumber(row.e_hat_norm)
            << ", \"e_star_norm\": " << jsonNumber(row.e_star_norm)
            << ", \"e_rel_to_exact\": " << jsonNumber(row.e_rel_to_exact)
            << ", \"linear_residual_exact\": " << jsonNumber(row.linear_residual_exact)
            << ", \"linear_residual_approx\": " << jsonNumber(row.linear_residual_approx)
            << ", \"num_groups\": " << row.num_groups
            << ", \"coarse_dim\": " << row.coarse_dim
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"basis_build_sec\": " << jsonNumber(row.basis_build_sec)
            << ", \"coarse_lambda_sec\": " << jsonNumber(row.coarse_lambda_sec)
            << ", \"exact_joint_assembly_sec\": " << jsonNumber(row.exact_joint_assembly_sec)
            << ", \"exact_solve_sec\": " << jsonNumber(row.exact_solve_sec)
            << ", \"sweeps_sec\": " << jsonNumber(row.sweeps_sec)
            << ", \"factor_pass_sec\": " << jsonNumber(row.factor_pass_sec)
            << ", \"variable_pass_sec\": " << jsonNumber(row.variable_pass_sec)
            << ", \"coarse_eta_sec\": " << jsonNumber(row.coarse_eta_sec)
            << ", \"coarse_solve_sec\": " << jsonNumber(row.coarse_solve_sec)
            << ", \"prolong_inject_sec\": " << jsonNumber(row.prolong_inject_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << "}";
        out << (i + 1 == results.mg_history.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

}  // namespace slam
