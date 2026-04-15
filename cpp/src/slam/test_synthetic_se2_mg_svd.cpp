#include <iostream>
#include <stdexcept>
#include <string>

#include "slam/SyntheticSE2MG.h"

namespace {

bool parseInt(const char* s, int& out) {
    try {
        out = std::stoi(std::string(s));
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string problem_file;
    std::string out_json = "synthetic_se2_mg_svd_results.json";
    int num_outer = 3;
    int inner_cycles = 3;
    int pre_sweeps = 50;
    int group_size = 20;
    int r_reduced = 4;
    bool strict_compare = false;
    int sync_num_threads = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--problem-file" && i + 1 < argc) {
            problem_file = argv[++i];
        } else if (arg == "--out-json" && i + 1 < argc) {
            out_json = argv[++i];
        } else if (arg == "--num-outer" && i + 1 < argc) {
            parseInt(argv[++i], num_outer);
        } else if (arg == "--inner-cycles" && i + 1 < argc) {
            parseInt(argv[++i], inner_cycles);
        } else if (arg == "--pre-sweeps" && i + 1 < argc) {
            parseInt(argv[++i], pre_sweeps);
        } else if (arg == "--group-size" && i + 1 < argc) {
            parseInt(argv[++i], group_size);
        } else if (arg == "--r-reduced" && i + 1 < argc) {
            parseInt(argv[++i], r_reduced);
        } else if (arg == "--sync-threads" && i + 1 < argc) {
            parseInt(argv[++i], sync_num_threads);
        } else if (arg == "--strict-compare") {
            strict_compare = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: test_synthetic_se2_mg_svd --problem-file <path> [--out-json <path>] "
                         "[--num-outer N] [--inner-cycles C] [--pre-sweeps K] "
                         "[--group-size G] [--r-reduced R] [--strict-compare] [--sync-threads T]\n";
            return 0;
        }
    }

    if (problem_file.empty()) {
        throw std::runtime_error("--problem-file is required");
    }

    slam::SyntheticSE2Problem problem = slam::loadSyntheticSE2Problem(problem_file);
    slam::ExperimentResults results = slam::runSyntheticSE2Experiment(
        problem,
        num_outer,
        inner_cycles,
        pre_sweeps,
        group_size,
        r_reduced,
        strict_compare,
        sync_num_threads
    );
    slam::writeExperimentResultsJson(
        results,
        out_json,
        num_outer,
        inner_cycles,
        pre_sweeps,
        group_size,
        r_reduced,
        strict_compare,
        sync_num_threads
    );

    std::cout << "wrote " << out_json << "\n";
    if (results.direct_history.size() > 1) {
        std::cout << "direct_final_objective=" << results.direct_history.back().nonlinear_objective << "\n";
    }
    if (!results.mg_history.empty()) {
        std::cout << "mg_final_objective=" << results.mg_history.back().nonlinear_objective << "\n";
    }
    return 0;
}
