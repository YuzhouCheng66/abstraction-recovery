#pragma once
#include <map>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include <manif/Rn.h>
#include <manif/SE2.h>
#include <manif/SO2.h>

#include <simulation/simulation.h>

#include <gbp/factor_graph.h>
#include <gbp/multi_factor_graph.h>

inline void solve_with_batch(std::map<int32_t, FactorGraph> &factor_graphs) {
  // Assume all of the variables are SE2
  int32_t dim_f = 0;
  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &[factor_id, factor] : factor_graph.factors_) {
      if (factor->type_ == FactorType::BETWEEN2D) {
        dim_f += 3;
      }
      if (factor->type_ == FactorType::PRIOR2D) {
        dim_f += 3;
      }
      if (factor->type_ == FactorType::RANGEBEARING2D) {
        dim_f += 2;
      }
    }
  }
  std::map<Key, int32_t> variable_idx;
  int32_t dim_v = 0;
  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &[variable_id, variable] : factor_graph.variables_) {
      variable_idx[variable_id] = dim_v;
      dim_v += 3;
    }
  }

  // Update factors state vector
  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &[factor_id, factor] : factor_graph.factors_) {
      auto state_vector = make_state_vector(factor_graphs, factor.get());
      factor->update_factor(state_vector, 1);
    }
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim_f, dim_v);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(dim_f);
  // Fill up b
  int32_t i = 0;
  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &[factor_id, factor] : factor_graph.factors_) {
      int32_t n;
      Eigen::MatrixXd R;
      Eigen::VectorXd e;
      if (factor->type_ == FactorType::BETWEEN2D) {
        auto f = dynamic_cast<BetweenSE2 *>(factor.get());
        auto [h, J] = residual_between(f->X0_vec_.at(0), f->X0_vec_.at(1), f->measurement_);
        R = Eigen::MatrixXd(f->measurement_Lam_.llt().matrixL());
        e = h;
        n = 3;
      }
      if (factor->type_ == FactorType::PRIOR2D) {
        auto f = dynamic_cast<PriorSE2 *>(factor.get());
        auto [h, J] = residual_prior(f->X0_vec_.at(0), f->measurement_);
        R = Eigen::MatrixXd(f->measurement_Lam_.llt().matrixL());
        e = h;
        n = 3;
      }
      if (factor->type_ == FactorType::RANGEBEARING2D) {
        auto f = dynamic_cast<RangeBearingSE2 *>(factor.get());
        auto [h, J] = residual_range_bearing(f->X0_vec_.at(0), f->X0_vec_.at(1), f->measurement_);
        R = Eigen::MatrixXd(f->measurement_Lam_.llt().matrixL());
        e = h;
        n = 2;
      }
      b.segment(i, n) = R * e;
      i += n;
    }
  }
  // Fill up A
  i = 0;
  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &[factor_id, factor] : factor_graph.factors_) {
      int32_t n;
      Eigen::MatrixXd R, J;
      Eigen::VectorXd e;
      if (factor->type_ == FactorType::BETWEEN2D) {
        auto f = dynamic_cast<BetweenSE2 *>(factor.get());
        auto [h, Jac] = residual_between(f->X0_vec_.at(0), f->X0_vec_.at(1), f->measurement_);
        R = Eigen::MatrixXd(f->measurement_Lam_.llt().matrixL());
        J = R * Jac;
        n = 3;
      }
      if (factor->type_ == FactorType::PRIOR2D) {
        auto f = dynamic_cast<PriorSE2 *>(factor.get());
        auto [h, Jac] = residual_prior(f->X0_vec_.at(0), f->measurement_);
        R = Eigen::MatrixXd(f->measurement_Lam_.llt().matrixL());
        J = R * Jac;
        n = 3;
      }
      if (factor->type_ == FactorType::RANGEBEARING2D) {
        auto f = dynamic_cast<RangeBearingSE2 *>(factor.get());
        auto [h, Jac] = residual_range_bearing(f->X0_vec_.at(0), f->X0_vec_.at(1), f->measurement_);
        R = Eigen::MatrixXd(f->measurement_Lam_.llt().matrixL());
        J = R * Jac;
        n = 2;
      }
      int32_t ii = 0;
      for (auto &variable_id : factor->neighbors_) {
        int32_t vi = variable_idx.at(variable_id);
        A.block(i, vi, n, 3) = J.block(0, ii, n, 3);
        ii += 3;
      }
      i += n;
    }
  }

  // solve
  // Eigen::VectorXd delta = (A.transpose() * A).inverse() * A.transpose() * b;
  //
  // Eigen::VectorXd delta = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * b);

  Eigen::MatrixXd ATA = A.transpose() * A;
  Eigen::SparseMatrix<double> ATA_S = ATA.sparseView(0);
  // ATA_S.makeCompressed();
  // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solverQR;
  // solverQR.analyzePattern(ATA_S);
  // solverQR.compute(ATA_S);
  // if (solverQR.info() != Eigen::Success) {
  // std::cout << "Least squares solver: decomposition was not successfull." << std::endl;
  //}
  // Eigen::VectorXd delta = solverQR.solve(A.transpose() * b);
  // if (solverQR.info() != Eigen::Success) {
  // std::cout << "Least squares solver: solving was not successfull." << std::endl;
  //}
  //
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
  cg.compute(ATA_S);
  Eigen::VectorXd delta = cg.solve(A.transpose() * b);

  for (auto &[graph_id, factor_graph] : factor_graphs) {
    for (auto &[variable_id, variable] : factor_graph.variables_) {
      int32_t i = variable_idx[variable_id];
      double x = delta.segment(i, 3)[0];
      double y = delta.segment(i, 3)[1];
      double theta = delta.segment(i, 3)[2];
      variable->state_ += -manif::SE2Tangentd(x, y, theta);
    }
  }
}
