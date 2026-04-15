#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "gbp/FactorGraph.h"

namespace slam {

struct SyntheticSE2Edge {
    int i = -1;
    int j = -1;
    Eigen::Vector3d measurement = Eigen::Vector3d::Zero();
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    std::string kind;
};

struct SyntheticSE2Problem {
    std::vector<Eigen::Vector3d> gt_poses;
    std::vector<Eigen::Vector3d> init_poses;
    std::vector<SyntheticSE2Edge> edges;
    Eigen::Vector3d anchor_pose = Eigen::Vector3d::Zero();
    Eigen::Matrix3d anchor_information = Eigen::Matrix3d::Zero();
};

struct OuterDirectRow {
    int outer = 0;
    double nonlinear_objective = 0.0;
    double linear_step_norm = 0.0;
    double linear_residual_norm = 0.0;
    double outer_total_sec = 0.0;
    double build_graph_sec = 0.0;
    double joint_assembly_sec = 0.0;
    double exact_solve_sec = 0.0;
    double apply_step_sec = 0.0;
    double objective_eval_sec = 0.0;
};

struct OuterMGRow {
    int outer = 0;
    double nonlinear_objective = 0.0;
    double e_hat_norm = 0.0;
    double e_star_norm = 0.0;
    double e_rel_to_exact = 0.0;
    double linear_residual_exact = 0.0;
    double linear_residual_approx = 0.0;
    int num_groups = 0;
    int coarse_dim = 0;
    double outer_total_sec = 0.0;
    double build_graph_sec = 0.0;
    double basis_build_sec = 0.0;
    double coarse_lambda_sec = 0.0;
    double exact_joint_assembly_sec = 0.0;
    double exact_solve_sec = 0.0;
    double sweeps_sec = 0.0;
    double factor_pass_sec = 0.0;
    double variable_pass_sec = 0.0;
    double coarse_eta_sec = 0.0;
    double coarse_solve_sec = 0.0;
    double prolong_inject_sec = 0.0;
    double apply_step_sec = 0.0;
    double objective_eval_sec = 0.0;
};

struct ExperimentResults {
    int num_poses = 0;
    int num_edges = 0;
    double initial_objective = 0.0;
    std::vector<OuterDirectRow> direct_history;
    std::vector<OuterMGRow> mg_history;
};

SyntheticSE2Problem loadSyntheticSE2Problem(const std::string& path);

double nonlinearObjective(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& poses
);

std::vector<Eigen::Vector3d> applyPoseDeltas(
    const std::vector<Eigen::Vector3d>& base_poses,
    const Eigen::VectorXd& delta_vec
);

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    double tiny_prior = 1e-12
);

ExperimentResults runSyntheticSE2Experiment(
    const SyntheticSE2Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    bool strict_compare = false,
    int sync_num_threads = 0
);

void writeExperimentResultsJson(
    const ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    bool strict_compare = false,
    int sync_num_threads = 0
);

}  // namespace slam
