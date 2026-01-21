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


// ============================================================
// SE(2) nonlinear builder
// ============================================================

static inline Eigen::Vector3d relposeSE2_fromGT(const Eigen::Vector3d& pose_i,
                                                const Eigen::Vector3d& pose_j) {
    const double xi = pose_i(0), yi = pose_i(1), thi = pose_i(2);
    const double xj = pose_j(0), yj = pose_j(1), thj = pose_j(2);
    const double c = std::cos(thi), s = std::sin(thi);
    Eigen::Matrix2d RT;
    RT <<  c, s,
          -s, c;
    Eigen::Vector2d dp(xj - xi, yj - yi);
    Eigen::Vector2d trans_local = RT * dp;
    const double dth = wrapAngleSE2(thj - thi);
    return Eigen::Vector3d(trans_local(0), trans_local(1), dth);
}

static inline Eigen::Vector3d composeSE2(const Eigen::Vector3d& pose_i,
                                         const Eigen::Vector3d& z_ij) {
    const double xi = pose_i(0), yi = pose_i(1), thi = pose_i(2);
    const double dx = z_ij(0), dy = z_ij(1), dth = z_ij(2);
    const double c = std::cos(thi), s = std::sin(thi);
    Eigen::Vector2d t_global(c*dx - s*dy, s*dx + c*dy);
    const double xj = xi + t_global(0);
    const double yj = yi + t_global(1);
    const double thj = wrapAngleSE2(thi + dth);
    return Eigen::Vector3d(xj, yj, thj);
}

NoiseCacheSE2 generateNoiseCacheSE2(
    const std::vector<SlamEdge>& edges,
    const NoiseConfigSE2& config)
{
    NoiseCacheSE2 cache;

    std::mt19937 rng(config.use_seed ? config.seed : std::random_device{}());
    std::normal_distribution<double> normal01(0.0, 1.0);

    const double odom_sigma_xy = config.odom_sigma;
    const double loop_sigma_xy = config.loop_sigma;
    const double prior_sigma_xy = config.prior_sigma;

    const double odom_sigma_th  = config.odom_sigma * config.theta_ratio;
    const double loop_sigma_th  = config.loop_sigma * config.theta_ratio;
    const double prior_sigma_th = config.prior_sigma * config.theta_ratio;

    for (const auto& e : edges) {
        // Between edge
        if (!e.is_prior && !e.is_anchor) {
            const int i = e.source;
            const int j = e.target;
            if (i >= 0 && j >= 0) {
                const bool is_odom = (j == i + 1);
                const double s_xy = is_odom ? odom_sigma_xy : loop_sigma_xy;
                const double s_th = is_odom ? odom_sigma_th : loop_sigma_th;

                Eigen::Vector3d noise;
                noise(0) = normal01(rng) * s_xy;
                noise(1) = normal01(rng) * s_xy;
                noise(2) = normal01(rng) * s_th;
                cache.between_noises[{i, j}] = noise;
            }
        }
        // Strong prior (exclude anchor)
        else if (e.is_prior && !e.is_anchor) {
            const int i = e.source;
            if (i >= 0) {
                Eigen::Vector3d noise;
                noise(0) = normal01(rng) * prior_sigma_xy;
                noise(1) = normal01(rng) * prior_sigma_xy;
                noise(2) = normal01(rng) * prior_sigma_th;
                cache.prior_noises[i] = noise;
            }
        }
    }

    return cache;
}

gbp::FactorGraph buildNoisyPoseGraphSE2(
    const std::vector<SlamNodeSE2>& nodes,
    const std::vector<SlamEdge>& edges,
    const NoiseConfigSE2& config)
{
    gbp::FactorGraph fg;
    fg.nonlinear_factors = true;
    fg.eta_damping = 0.0;

    const int N = (int)nodes.size();
    std::vector<gbp::VariableNode*> var_ptrs(N, nullptr);

    // Pre-generate noises (Python: rng.normal per edge)
    NoiseCacheSE2 noise_cache = generateNoiseCacheSE2(edges, config);

    // Weak prior (Python: tiny_prior * diag([1,1,100]))
    Eigen::Matrix3d Lam_weak = Eigen::Matrix3d::Zero();
    Lam_weak(0,0) = config.tiny_prior;
    Lam_weak(1,1) = config.tiny_prior;
    Lam_weak(2,2) = config.tiny_prior * 100.0;

    // Create variables
    for (int i = 0; i < N; ++i) {
        gbp::VariableNode* v = fg.addVariable(i, 3);
        v->GT = nodes[i].GT;
        v->mu = Eigen::Vector3d::Zero();

        v->prior.setLam(Lam_weak);
        v->prior.setEta(Lam_weak * v->mu);

        var_ptrs[i] = v;
    }

    // Precision blocks
    auto makeLambda = [&](double sigma_xy) -> Eigen::Matrix3d {
        const double sigma_th = sigma_xy * config.theta_ratio;
        Eigen::Matrix3d L = Eigen::Matrix3d::Zero();
        L(0,0) = 1.0 / (sigma_xy * sigma_xy);
        L(1,1) = 1.0 / (sigma_xy * sigma_xy);
        L(2,2) = 1.0 / (sigma_th * sigma_th);
        return L;
    };

    const Eigen::Matrix3d Lambda_odom  = makeLambda(config.odom_sigma);
    const Eigen::Matrix3d Lambda_loop  = makeLambda(config.loop_sigma);
    const Eigen::Matrix3d Lambda_prior = makeLambda(config.prior_sigma);

    // Store sequential odom measurements for initialization
    std::map<std::pair<int,int>, Eigen::Vector3d> odom_meas;
    std::map<int, Eigen::Vector3d> prior_meas;

    int fid = 0;

    // ---------- Anchor (very strong) at node 0 ----------
    {
        gbp::VariableNode* v0 = var_ptrs[0];
        const Eigen::Vector3d z_anchor = v0->GT.head<3>();

        Eigen::Matrix3d Lambda_anchor = Eigen::Matrix3d::Zero();
        Lambda_anchor(0,0) = 1.0 / (1e-3 * 1e-3);
        Lambda_anchor(1,1) = 1.0 / (1e-3 * 1e-3);
        Lambda_anchor(2,2) = 1.0 / (1e-5 * 1e-5);

        gbp::Factor* f = fg.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{v0},
            std::vector<Eigen::VectorXd>{z_anchor},
            std::vector<Eigen::MatrixXd>{Lambda_anchor},
            slam::measFnUnarySE2,
            slam::jacFnUnarySE2
        );

        // linearize at z_anchor (matches Python anchor creation)
        f->computeFactor(z_anchor, true);
        fg.connect(f, v0, 0);
    }

    // ---------- Create all other factors (do not depend on mu yet) ----------
    for (const auto& e : edges) {
        // Between
        if (!e.is_prior && !e.is_anchor) {
            const int i = e.source;
            const int j = e.target;
            if (i < 0 || j < 0 || i >= N || j >= N) continue;

            gbp::VariableNode* vi = var_ptrs[i];
            gbp::VariableNode* vj = var_ptrs[j];

            // GT relative pose
            Eigen::Vector3d z = relposeSE2_fromGT(vi->GT.head<3>(), vj->GT.head<3>());

            // Add noise (wrap theta)
            auto itn = noise_cache.between_noises.find({i, j});
            if (itn != noise_cache.between_noises.end()) {
                z(0) += itn->second(0);
                z(1) += itn->second(1);
                z(2)  = wrapAngleSE2(z(2) + itn->second(2));
            }

            const bool is_odom = (j == i + 1);
            const Eigen::Matrix3d& z_precision = is_odom ? Lambda_odom : Lambda_loop;

            // Store sequential odom for init
            if (is_odom) odom_meas[{i, j}] = z;

            gbp::Factor* f = fg.addFactor(
                fid++,
                std::vector<gbp::VariableNode*>{vi, vj},
                std::vector<Eigen::VectorXd>{z},
                std::vector<Eigen::MatrixXd>{z_precision},
                slam::measFnBetweenSE2, 
                slam::jacFnBetweenSE2
            );

            fg.connect(f, vi, 0);
            fg.connect(f, vj, 1);
        }
        // Strong prior (exclude anchor)
        else if (e.is_prior && !e.is_anchor) {
            const int i = e.source;
            if (i < 0 || i >= N) continue;

            gbp::VariableNode* vi = var_ptrs[i];

            Eigen::Vector3d z = vi->GT.head<3>();
            auto itp = noise_cache.prior_noises.find(i);
            if (itp != noise_cache.prior_noises.end()) {
                z(0) += itp->second(0);
                z(1) += itp->second(1);
                z(2)  = wrapAngleSE2(z(2) + itp->second(2));
            }

            prior_meas[i] = z;

            gbp::Factor* f = fg.addFactor(
                fid++,
                std::vector<gbp::VariableNode*>{vi},
                std::vector<Eigen::VectorXd>{z},
                std::vector<Eigen::MatrixXd>{Lambda_prior},
                slam::measFnUnarySE2,
                slam::jacFnUnarySE2
            );

            fg.connect(f, vi, 0);
        }
        // Anchor edge is handled above; ignore here
    }

    // ---------- Sequentially initialize mu (Python policy) ----------
    if (N > 0) {
        var_ptrs[0]->mu = var_ptrs[0]->GT.head<3>();
    }

    for (int i = 0; i < N - 1; ++i) {
        auto ito = odom_meas.find({i, i + 1});
        if (ito != odom_meas.end()) {
            const Eigen::Vector3d mu_i = var_ptrs[i]->mu.head<3>();
            var_ptrs[i + 1]->mu = composeSE2(mu_i, ito->second);
        } else {
            var_ptrs[i + 1]->mu = var_ptrs[i + 1]->GT.head<3>();
        }

        // override if strong prior exists
        auto itp = prior_meas.find(i + 1);
        if (itp != prior_meas.end()) {
            var_ptrs[i + 1]->mu = itp->second;
        }
    }

    // Update weak prior eta to match initialized mu
    for (int i = 0; i < N; ++i) {
        auto* v = var_ptrs[i];
        v->prior.setEta(Lam_weak * v->mu);
    }

    // ---------- Linearize all factors at current mu ----------
    for (auto& fup : fg.factors) {
        gbp::Factor* f = fup.get();
        if (!f) continue;

        int total = 0;
        for (auto* vn : f->adj_var_nodes) total += vn->dofs;

        Eigen::VectorXd linpoint(total);
        int off = 0;
        for (auto* vn : f->adj_var_nodes) {
            linpoint.segment(off, vn->dofs) = vn->mu;
            off += vn->dofs;
        }

        f->computeFactor(linpoint, true);
    }

    return fg;
}

} // namespace slam
