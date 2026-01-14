#include "slam/SlamFactorGraph.h"
#include <random>

namespace slam {

NoiseCache generateNoiseCache(
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config)
{
    NoiseCache cache;

    std::mt19937 rng(config.use_seed ? config.seed : std::random_device{}());
    std::normal_distribution<double> normal01(0.0, 1.0);

    for (const auto& e : edges) {
        // Binary edge: odom
        if (!e.is_prior && !e.is_anchor) {
            const int src = e.source;
            const int dst = e.target;
            if (src >= 0 && dst >= 0) {
                Eigen::Vector2d noise;
                noise(0) = normal01(rng) * config.odom_sigma;
                noise(1) = normal01(rng) * config.odom_sigma;
                cache.odom_noises[{src, dst}] = noise;
            }
        }
        // Unary edge: strong prior (anchor 不加 noise)
        else if (e.is_prior && !e.is_anchor) {
            const int src = e.source;
            if (src >= 0) {
                Eigen::Vector2d noise;
                noise(0) = normal01(rng) * config.prior_sigma;
                noise(1) = normal01(rng) * config.prior_sigma;
                cache.prior_noises[src] = noise;
            }
        }
    }

    return cache;
}

gbp::FactorGraph buildNoisyPoseGraph(
    const std::vector<SlamNode>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfig& config)
{
    gbp::FactorGraph fg;
    fg.nonlinear_factors = false;
    fg.eta_damping = 0.0;

    const int N = (int)nodes.size();
    const Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();

    // ---- Python: Pre-generate noise ----
    NoiseCache noise_cache = generateNoiseCache(edges, config);

    // ---- Python: var nodes ----
    std::vector<gbp::VariableNode*> var_ptrs(N, nullptr);

    for (int i = 0; i < N; ++i) {
        gbp::VariableNode* v = fg.addVariable(i, 2);
        v->GT = nodes[i].GT;

        // tiny prior
        v->prior.setLam(config.tiny_prior * I2);
        v->prior.setEta(Eigen::Vector2d::Zero());

        var_ptrs[i] = v;
    }

    // ---- factors ----
    int fid = 0;

    for (const auto& e : edges) {

        // ========== ODOM (binary) ==========
        if (!e.is_prior && !e.is_anchor) {
            const int i = e.source;
            const int j = e.target;
            if (i < 0 || j < 0 || i >= N || j >= N) continue;

            gbp::VariableNode* vi = var_ptrs[i];
            gbp::VariableNode* vj = var_ptrs[j];

            // meas = (vj.GT - vi.GT) + noise
            Eigen::Vector2d meas = (vj->GT - vi->GT);
            auto itn = noise_cache.odom_noises.find({i, j});
            if (itn != noise_cache.odom_noises.end()) {
                meas += itn->second;
            }

            // meas_lambda = I / odom_sigma^2
            Eigen::Matrix2d meas_precision = I2 / (config.odom_sigma * config.odom_sigma);

            // linpoint = [vi.GT; vj.GT]
            Eigen::VectorXd linpoint(4);
            linpoint.segment<2>(0) = vi->GT;
            linpoint.segment<2>(2) = vj->GT;

            gbp::Factor* f = fg.addFactor(
                fid++,
                std::vector<gbp::VariableNode*>{vi, vj},
                std::vector<Eigen::VectorXd>{meas},
                std::vector<Eigen::MatrixXd>{meas_precision},
                slam::measFnOdom,
                slam::jacFnOdom
            );

            // Python: f.compute_factor(linpoint, update_self=True)
            f->computeFactor(linpoint, true);

            // 关键：补 connect（你当前 addFactor 不做连接）
            fg.connect(f, vi, 0);
            fg.connect(f, vj, 1);
        }

        // ========== PRIOR (unary strong prior) ==========
        else if (e.is_prior && !e.is_anchor) {
            const int i = e.source;
            if (i < 0 || i >= N) continue;

            gbp::VariableNode* vi = var_ptrs[i];

            // z = GT + noise
            Eigen::Vector2d z = vi->GT;
            auto itp = noise_cache.prior_noises.find(i);
            if (itp != noise_cache.prior_noises.end()) {
                z += itp->second;
            }

            Eigen::Matrix2d z_precision = I2 / (config.prior_sigma * config.prior_sigma);

            Eigen::VectorXd linpoint = z;

            gbp::Factor* f = fg.addFactor(
                fid++,
                std::vector<gbp::VariableNode*>{vi},
                std::vector<Eigen::VectorXd>{z},
                std::vector<Eigen::MatrixXd>{z_precision},
                slam::measFnUnary,
                slam::jacFnUnary
            );

            f->computeFactor(linpoint, true);

            fg.connect(f, vi, 0);
        }

        // ========== ANCHOR (unary, very strong) ==========
        else if (e.is_anchor && e.source == 0) {
            gbp::VariableNode* v0 = var_ptrs[0];

            Eigen::Vector2d z = v0->GT;
            Eigen::Matrix2d z_precision = I2 / (1e-4 * 1e-4);

            Eigen::VectorXd linpoint = z;

            gbp::Factor* f = fg.addFactor(
                fid++,
                std::vector<gbp::VariableNode*>{v0},
                std::vector<Eigen::VectorXd>{z},
                std::vector<Eigen::MatrixXd>{z_precision},
                slam::measFnUnary,
                slam::jacFnUnary
            );

            f->computeFactor(linpoint, true);

            fg.connect(f, v0, 0);
        }
    }

    return fg;
}

} // namespace slam
