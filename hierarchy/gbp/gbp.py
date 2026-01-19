"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""

import numpy as np
import time
import scipy.linalg
from utils.gaussian import NdimGaussian
from collections import deque
import heapq

#from amg import classes as amg_cls
#from amg import functions as amg_fnc

class FactorGraph:
    def __init__(self,
                 nonlinear_factors=True,
                 eta_damping=0.0,
                 beta=None,
                 num_undamped_iters=None,
                 min_linear_iters=None,
                 wild_thresh=0):

        self.var_nodes = []
        self.factors = []

        self.n_var_nodes = 0
        self.n_factor_nodes = 0

        self.nonlinear_factors = nonlinear_factors

        self.eta_damping = eta_damping

        #self.Q = deque()
        #self.b_wild = False
        self.wild_thresh = wild_thresh
        #self.multigrid_vars = [[]]
        #self.multigrid_factors = [[]]
        #self.multigrid = False
        #self.conv_width = 1
        #self.conv_stride = 1

        #self.energy_history = []
        #self.error_history = []
        #self.nmsgs_history = []
        self.mus = []

        self.residual_eps = 1e-6
        self.var_residual = {}
        self.var_heap = []


        if nonlinear_factors:
            # For linearising nonlinear measurement factors.
            self.beta = beta  # Threshold change in mean of adjacent beliefs for relinearisation.
            self.num_undamped_iters = num_undamped_iters  # Number of undamped iterations after relinearisation before damping is set to 0.4
            self.min_linear_iters = min_linear_iters  # Minimum number of linear iterations before a factor is allowed to realinearise.

    def compute_all_messages(self, factors=None, level=None, local_relin=True):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        if level is not None:
            factors = self.multigrid_factors[level]
        for count, factor in enumerate(factors):
            if factor.active:
                # If relinearisation is local then damping is also set locally per factor.
                if self.nonlinear_factors and local_relin:
                    if factor.iters_since_relin == self.num_undamped_iters:
                        factor.eta_damping = self.eta_damping
                    factor.compute_messages(factor.eta_damping)
                else:
                    factor.compute_messages(self.eta_damping)

    def update_all_beliefs(self, vars=None, level=None, smoothing=False):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]
        if level is not None:
            vars = self.multigrid_vars[level]

        for var in vars:
            if var.active:
                if smoothing:
                    var.update_smooth_belief()
                else:
                    var.update_belief()


    def compute_all_factors(self, factors=None, level=None):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        if level is not None:
            factors = self.multigrid_factors[level]
        for count, factor in enumerate(factors):
            factor.compute_factor()

    def relinearise_factors(self):
        """
            Compute the factor distribution for all factors for which the local belief mean has deviated a distance
            greater than beta from the current linearisation point.
            Relinearisation is only allowed at a maximum frequency of once every min_linear_iters iterations.
        """
        if self.nonlinear_factors:
            for factor in self.factors:
                adj_belief_means = np.array([])
                for belief in factor.adj_beliefs:
                    adj_belief_means = np.concatenate((adj_belief_means, 1/np.diagonal(belief.lam) * belief.eta))
                if np.linalg.norm(factor.linpoint - adj_belief_means) > self.beta and factor.iters_since_relin >= self.min_linear_iters:
                    factor.compute_factor(linpoint=adj_belief_means)
                    factor.iters_since_relin = 0
                    factor.eta_damping = 0.0
                else:
                    factor.iters_since_relin += 1

    def robustify_all_factors(self):
        for factor in self.factors[:self.n_factor_nodes]:
            factor.robustify_loss()

    def synchronous_iteration(self, factors=None, level=None, local_relin=True, robustify=False):
        if level is not None:
            vars = self.multigrid_vars[level]
            factors = self.multigrid_factors[level]
        else:
            vars = self.var_nodes[:self.n_var_nodes]
            factors = self.factors[:self.n_factor_nodes]

        if robustify:
            self.robustify_all_factors(factors)
        if self.nonlinear_factors and local_relin:
            self.relinearise_factors(factors)

        self.compute_all_messages(factors, local_relin=local_relin)
        self.update_all_beliefs(vars)


    """
    def wildfire_iteration(self, local_relin=True, robustify=False):
        if len(self.Q) == 0:
            self.Q.append(self.factors[0]) 
            self.factors[0].in_queue = True

        for i in range(len(self.factors)):
            self.Q[0].compute_messages(self.eta_damping)

            for var in self.Q[0].adj_var_nodes:
                var.update_belief()
                #if any(self.Q[0].messages_dist[count] >= self.wild_thresh):

                for f in var.adj_factors:
                    if (not f.in_queue):
                        f.in_queue = True
                        self.Q.append(f)

            self.Q[0].in_queue = False
            self.Q.popleft()


    def residual_iteration(self, local_relin=True, robustify=False):

        # ---------- 默认参数 ----------
        if not hasattr(self, "residual_beta"):
            self.residual_beta = 1.0
        if not hasattr(self, "residual_eps"):
            self.residual_eps = 1e-6

        # 每个 factor 的 residual
        if not hasattr(self, "factor_residual"):
            self.factor_residual = {}

        # ---------- 初始化 self.heapq ----------
        if not hasattr(self, "heapq"):
            # ★★★ 就是你需要的初始化行 ★★★
            self.heapq = []     # 这是 priority queue（min-heap，用负号实现 max-heap）

        # 如果第一次运行，没有任何条目，那么加入所有 factor
        if len(self.heapq) == 0:
            for f in self.factors:
                r0 = self.factor_residual.get(f, 1.0)  # 初始 residual = 1.0
                self.factor_residual[f] = r0
                heapq.heappush(self.heapq, (-r0, f.factorID, f))   # push into PQ (max-heap by negation)

        # 每轮最多更新 N 个（和原 wildfire 一样）
        max_updates = len(self.factors)
        n_updates = 0

        while self.heapq and n_updates < max_updates:

            neg_r_f, _, f = heapq.heappop(self.heapq)
            r_f = -neg_r_f

            cur_r = self.factor_residual.get(f, 0.0)
            if r_f < cur_r - 1e-12:
                continue

            if r_f < self.residual_eps:
                break

            # ===== 1. 更新 factor 的 messages =====
            f.compute_messages(self.eta_damping)
            self.factor_residual[f] = 0.0

            # ===== 2. 变量 belief 更新 + 激活邻居因子 =====
            for var in f.adj_var_nodes:
                old_eta = np.array(var.belief.eta, copy=True)

                var.update_belief()

                new_eta = np.array(var.belief.eta)
                delta = float(np.linalg.norm(new_eta - old_eta))  # 默认 2-norm

                if delta <= 1e-12:
                    continue

                # 通知所有邻居 factor
                for g in var.adj_factors:
                    if g is f:
                        continue

                    est_r_g = self.residual_beta * delta
                    old_r_g = self.factor_residual.get(g, 0.0)

                    if est_r_g > old_r_g:
                        self.factor_residual[g] = est_r_g
                        heapq.heappush(self.heapq, (-est_r_g, g.factorID, g))

            # ===== 3. 本 factor 的 residual =====
            

            n_updates += 1
    """


    # ================================================================
    # Online / Incremental construction helpers
    # ================================================================
    # ===================== PATCHED: Remove duplicate global-scope methods =====================
    # (No code here, just a marker for clarity)
    def push_var(self, v, residual=1.0):
        """
        Push (or re-push) a variable into the residual heap.
        We allow stale heap entries and skip them when popped.
        """
        r = float(residual)
        self.var_residual[v] = r
        heapq.heappush(self.var_heap, (-r, v.variableID, v))

    def ensure_variable(self, variable_id, dofs=2, GT=None, tiny_prior=1e-6, init_mu=None, activate=True):
        """
        Get or create a variable with given id.
        Used for online pose-graph growth.
        """
        if variable_id < len(self.var_nodes) and self.var_nodes[variable_id] is not None:
            v = self.var_nodes[variable_id]
            if GT is not None:
                v.GT = np.array(GT, copy=True)
            return v

        while len(self.var_nodes) <= variable_id:
            self.var_nodes.append(None)

        v = VariableNode(variable_id, dofs=dofs)
        v.type = "variable"

        if GT is not None:
            v.GT = np.array(GT, copy=True)

        # tiny prior to avoid singularity
        v.prior.lam = float(tiny_prior) * np.eye(dofs)
        if init_mu is None:
            v.prior.eta = np.zeros(dofs)
        else:
            init_mu = np.asarray(init_mu).reshape(-1)
            v.prior.eta = v.prior.lam @ init_mu

        v.update_belief()

        self.var_nodes[variable_id] = v
        self.n_var_nodes = max(self.n_var_nodes, variable_id + 1)
        self.push_var(v, 1.0)
        return v

    def add_binary_factor(self, vi, vj, measurement, measurement_lambda,
                          meas_fn, jac_fn, linpoint=None,
                          ftype="base", activate=True):
        """
        Add a binary factor (odometry / loop closure) online.
        """
        fid = self.n_factor_nodes
        f = Factor(fid, [vi, vj],
                   measurement=measurement,
                   measurement_lambda=measurement_lambda,
                   meas_fn=meas_fn,
                   jac_fn=jac_fn)
        f.type = ftype

        vi.adj_factors.append(f)
        vj.adj_factors.append(f)

        if linpoint is None:
            linpoint = np.concatenate([vi.mu, vj.mu])
        f.compute_factor(linpoint=linpoint, update_self=True)
        f.compute_messages(eta_damping=0.0)

        self.factors.append(f)
        self.n_factor_nodes += 1
        vi.update_belief()
        vj.update_belief()

        if activate:
            self.push_var(vi, 1.0)
            self.push_var(vj, 1.0)

        return f

    def add_unary_prior_factor(self, v, measurement, measurement_lambda,
                               meas_fn, jac_fn, linpoint=None,
                               ftype="prior", activate=True):
        """
        Add a unary prior / anchor factor online.
        """
        fid = self.n_factor_nodes
        f = Factor(fid, [v],
                   measurement=measurement,
                   measurement_lambda=measurement_lambda,
                   meas_fn=meas_fn,
                   jac_fn=jac_fn)
        f.type = ftype

        v.adj_factors.append(f)

        if linpoint is None:
            linpoint = v.mu
        f.compute_factor(linpoint=linpoint, update_self=True)
        f.compute_messages(eta_damping=0.0)

        self.factors.append(f)
        self.n_factor_nodes += 1

        v.update_belief()

        if activate:
            self.push_var(v, 1.0)

        return f
    

    def residual_iteration_var_heap(self, max_updates=50):
        """
        Residual-based GBP iteration, priority queue on *variables*.

        逻辑：
        - 堆元素 = (-residual_v, varID, var)
        residual_v = ||Δ eta_v||_2
        - 每次选 residual 最大的 variable v：
            对其所有邻居 factor f:
                f.compute_messages(...)
                对 f.adj_var_nodes 里的每个 u:
                    更新 u.belief
                    计算新的 residual_u，并丢回堆
        """

        # ---------- 初始化 var_heap ----------
        if len(self.var_heap) == 0:
            for v in self.var_nodes:
                r0 = 0.0
                self.var_residual[v] = r0
                heapq.heappush(self.var_heap, (-r0, v.variableID, v))

        n_updates = 0

        # 本轮最多扩散 |V| 次；heap 空则提前结束
        while (n_updates < max_updates) and len(self.var_heap) > 0:
            neg_r_v, _, v = heapq.heappop(self.var_heap)
            r_v = -neg_r_v

            cur_r = self.var_residual.get(v, 0.0)

            # stale entry
            if abs(r_v - cur_r) > 1e-12:
                continue

            # ===== 对 v 的邻居 factors 做一次“扩散” =====
            for f in v.adj_factors:
                f.compute_messages(self.eta_damping)

                for u in f.adj_var_nodes:
                    old_eta = np.array(u.belief.eta, copy=True)
                    u.update_belief()
                    new_eta = np.array(u.belief.eta)

                    est_r_u = float(np.linalg.norm(new_eta - old_eta))
                    old_r_u = self.var_residual.get(u, 0.0)

                    if est_r_u > old_r_u + 1e-12:
                        self.var_residual[u] = est_r_u
                        heapq.heappush(self.var_heap, (-est_r_u, u.variableID, u))

            # 本轮用 v 扩散过，视为 residual 清零
            self.var_residual[v] = 0.0
            n_updates += 1


    def joint_distribution_inf(self):
        """
            Get the joint distribution over all variables in the information form
            If nonlinear factors, it is taken at the current linearisation point.
        """

        sizes = [vn.dofs for vn in self.var_nodes]
        total = sum(sizes)

        eta = np.empty(total, dtype=float)                 # Direct one-time allocation
        lam = np.zeros((total, total), dtype=float)        # Pre-allocate large zero array
        var_ix = np.empty(len(self.var_nodes), dtype=int)

        offset = 0
        for vn in self.var_nodes:
            m = vn.dofs
            var_ix[vn.variableID] = offset
            eta[offset:offset+m] = vn.prior.eta           # Write fragments without copying extra memory
            lam[offset:offset+m, offset:offset+m] = vn.prior.lam
            offset += m


        for count, factor in enumerate(self.factors):
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                vID = adj_var_node.variableID
                # Diagonal contribution of factor
                eta[var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                    factor.factor.eta[factor_ix:factor_ix + adj_var_node.dofs]
                lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                    factor.factor.lam[factor_ix:factor_ix + adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if other_adj_var_node.variableID > adj_var_node.variableID:
                        other_vID = other_adj_var_node.variableID
                        # Off diagonal contributions of factor
                        lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs] += \
                            factor.factor.lam[factor_ix:factor_ix + adj_var_node.dofs, other_factor_ix:other_factor_ix + other_adj_var_node.dofs]
                        lam[var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                            factor.factor.lam[other_factor_ix:other_factor_ix + other_adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
                    other_factor_ix += other_adj_var_node.dofs
                factor_ix += adj_var_node.dofs

        return eta, lam


    def joint_distribution_cov(self):
        """
            Get the joint distribution over all variables in the covariance.
            If nonlinear factors, it is taken at the current linearisation point.
        """
        eta, lam = self.joint_distribution_inf()
        sigma = np.linalg.inv(lam)
        mu = sigma @ eta
        return mu, sigma
   

    def get_sigmas(self):
        """
            Get an array containing all current estimates of belief sigmas.
        """
        sigmas = np.array([])
        for var_node in self.var_nodes:
            sigmas = np.concatenate((sigmas, var_node.Sigma[0]))
        return sigmas



class VariableNode:
    def __init__(self,
                 variable_id,
                 dofs):

        self.variableID = variable_id
        self.adj_factors = []
        self.InfoMat = []  # Row vector of prior Information vector in factor order
        self.EtaVec = []  # Vector of prior eta values
        #self.multigrid = amg_cls.mutligrid_var_info(self)
        self.type = "None specified"
        self.active = True

        # Node variables are position of landmark in world frame. Initialize variable nodes at origin
        self.mu = np.zeros(dofs)
        self.Sigma = np.zeros([dofs, dofs])
        self.belief = NdimGaussian(dofs)

        self.prior = NdimGaussian(dofs)
        self.dofs = dofs


    def update_belief(self):
        """
            Update local belief estimate by taking product of all incoming messages along all edges.
            Then send belief to adjacent factor nodes.
        """
        # Update local belief
        eta = self.prior.eta.copy()
        lam = self.prior.lam.copy()
        for factor in self.adj_factors:
            message_ix = factor.adj_var_nodes.index(self)
            eta_inward, lam_inward = factor.messages[message_ix].eta, factor.messages[message_ix].lam
            eta += eta_inward
            lam += lam_inward

        self.belief.eta = eta 
        self.belief.lam = lam

        # try-except 注释，仅保留原 try 内容
        c, lower = scipy.linalg.cho_factor(lam, lower=False, check_finite=False)
        self.Sigma = scipy.linalg.cho_solve((c, lower), np.eye(lam.shape[0]))  # 解 Lam Sigma = I
        self.mu = self.Sigma @ eta
        # except np.linalg.LinAlgError:
        #     # fallback: 用伪逆
        #     self.Sigma = np.linalg.pinv(lam)
        #     self.mu = self.Sigma @ eta

        # Send belief to adjacent factors
        for factor in self.adj_factors:
            belief_ix = factor.adj_var_nodes.index(self)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = self.belief.eta, self.belief.lam


    def __str__(self):
        # print(obj) 时的显示
        return f"{self.variableID}"

    def __repr__(self):
        # 在 list / dict / 交互式终端里显示
        return f"VariableNode({self.variableID})"

class Factor:
    def __init__(self,
                 factor_id,
                 adj_var_nodes,
                 measurement,
                 measurement_lambda,
                 meas_fn,
                 jac_fn,
                 loss=None,
                 mahalanobis_threshold=2,
                 wildfire=False,
                 *args):
        """
            n_stds: number of standard deviations from mean at which loss transitions to robust loss function.
        """

        self.factorID = factor_id

        self.dofs_conditional_vars = 0
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = []
        self.adj_beliefs = []
        self.messages = []
        self.messages_prior = []
        self.messages_dist = []

        self.active = True

        self.level = 0
        self.in_queue = False

        self.type = "factor"

        for adj_var_node in self.adj_var_nodes:
            self.dofs_conditional_vars += adj_var_node.dofs
            self.adj_vIDs.append(adj_var_node.variableID)
            self.adj_beliefs.append(NdimGaussian(adj_var_node.dofs))
            self.messages.append(NdimGaussian(adj_var_node.dofs))#, eta=adj_var_node.prior.eta, lam=adj_var_node.prior.lam))
            self.messages_prior.append(NdimGaussian(adj_var_node.dofs))
            self.messages_dist.append(np.zeros(adj_var_node.dofs))

        self.factor = NdimGaussian(self.dofs_conditional_vars)
        self.linpoint = np.zeros(self.dofs_conditional_vars)  # linearisation point

        self.residual = None
        self.b_calc_mess_dist = wildfire

        # Measurement model
        self.measurement = measurement
        self.measurement_lambda = measurement_lambda
        self.meas_fn = meas_fn
        self.jac_fn = jac_fn
        self.args = args

        # Robust loss function
        self.loss = loss
        self.mahalanobis_threshold = mahalanobis_threshold
        self.robust_flag = False

        # Local relinearisation
        self.eta_damping = 0.
        self.iters_since_relin = 1

    def compute_residual(self):
        """
            Calculate the reprojection error vector.
        """
        adj_belief_means = []
        for belief in self.adj_beliefs:
            #adj_belief_means = np.concatenate((adj_belief_means, np.linalg.inv(belief.lam) @ belief.eta))
            adj_belief_means = np.concatenate((adj_belief_means, 1/np.diagonal(belief.lam) * belief.eta))
        
        # d = (self.meas_fn(adj_belief_means, *self.args) - self.measurement) / self.adaptive_gauss_noise_var
        # d = (np.array([[-1,0,1,0],[0,-1,0,1]]) @ adj_belief_means - self.measurement) / self.adaptive_gauss_noise_var
        d = np.array(self.measurement) @ self.factor.lam[:2,2:] - self.factor.lam[:2,:] @ adj_belief_means
        # d = np.array(self.measurement) * self.gauss_noise_var - (adj_belief_means[2:] - adj_belief_means[:2]) * self.adaptive_gauss_noise_var
        # ^^^ This is equivalent to the equations below which are explicitly r = b - Ax
        # J = self.jac_fn(self.linpoint, *self.args)
        # meas_model_lambda = np.eye(len(self.measurement)) / self.adaptive_gauss_noise_var
        # d = J.T @ meas_model_lambda @ self.measurement - self.factor.lam @ adj_belief_means

        self.residual = d
        
        return d

    def energy(self):
        """
            Computes the squared error using the appropriate loss function.
        """
        return 0.5 * np.linalg.norm(self.residual) ** 2

    def compute_factor(self, linpoint=None, update_self=True):
        """
            Compute the factor given the linearisation point.
            If not given then linearisation point is mean of belief of adjacent nodes.
            If measurement model is linear then factor will always be the same regardless of linearisation point.
        """

        if linpoint is None:
            self.linpoint = np.concatenate([
                np.asarray(v.mu) for v in self.adj_var_nodes
            ])

        else:
            self.linpoint = linpoint

        # They are all lists
        J = self.jac_fn(self.linpoint, *self.args)
        pred_measurement = self.meas_fn(self.linpoint, *self.args)
        lambda_factor = np.zeros_like(self.factor.lam)
        eta_factor = np.zeros_like(self.factor.eta)
        
        for i in range(len(J)):
            lambda_factor += J[i].T @ self.measurement_lambda[i] @ J[i]
            eta_factor += J[i].T @ (self.measurement_lambda[i] @ (J[i] @ self.linpoint + self.measurement[i] - pred_measurement[i]))

        if update_self:
            self.factor.eta, self.factor.lam = eta_factor, lambda_factor

        return eta_factor, lambda_factor

    def robustify_loss(self):
        """
            Rescale the variance of the noise in the Gaussian measurement model if necessary and update the factor
            correspondingly.
        """
        old_adaptive_gauss_noise_var = self.adaptive_gauss_noise_var
        if self.loss is None:
            self.adaptive_gauss_noise_var = self.gauss_noise_var

        else:
            adj_belief_means = np.array([])
            for belief in self.adj_beliefs:
                adj_belief_means = np.concatenate((adj_belief_means, 1/np.diagonal(belief.lam) * belief.eta))
            pred_measurement = self.meas_fn(self.linpoint, *self.args)

            if self.loss == 'huber':  # Loss is linear after Nstds from mean of measurement model
                mahalanobis_dist = np.linalg.norm(self.measurement - pred_measurement) / np.sqrt(self.gauss_noise_var)
                if mahalanobis_dist > self.mahalanobis_threshold:
                    self.adaptive_gauss_noise_var = self.gauss_noise_var * mahalanobis_dist**2 / \
                            (2*(self.mahalanobis_threshold * mahalanobis_dist - 0.5 * self.mahalanobis_threshold**2))
                    self.robust_flag = True
                else:
                    self.robust_flag = False
                    self.adaptive_gauss_noise_var = self.gauss_noise_var

            elif self.loss == 'constant':  # Loss is constant after Nstds from mean of measurement model
                mahalanobis_dist = np.linalg.norm(self.measurement - pred_measurement) / np.sqrt(self.gauss_noise_var)
                if mahalanobis_dist > self.mahalanobis_threshold:
                    self.adaptive_gauss_noise_var = mahalanobis_dist**2
                    self.robust_flag = True
                else:
                    self.robust_flag = False
                    self.adaptive_gauss_noise_var = self.gauss_noise_var

        # Update factor using existing linearisation point (we are not relinearising).
        self.factor.eta *= old_adaptive_gauss_noise_var / self.adaptive_gauss_noise_var
        self.factor.lam *= old_adaptive_gauss_noise_var / self.adaptive_gauss_noise_var

    def relinearise(self, min_linear_iters, beta):
        adj_belief_means = np.array([])
        for belief in self.adj_beliefs:
            adj_belief_means = np.concatenate((adj_belief_means, 1/np.diagonal(belief.lam) * belief.eta))
        if np.linalg.norm(self.linpoint - adj_belief_means) > beta and self.iters_since_relin >= min_linear_iters:
            self.compute_factor(linpoint=adj_belief_means)
            self.iters_since_relin = 0
            self.eta_damping = 0.0
        else:
            self.iters_since_relin += 1

    #@profile
    def compute_messages(self, eta_damping):
        """
            Compute all outgoing messages from the factor.
            This is specialised for one and two variable factors.
        """

        if len(self.adj_vIDs) == 1:
            self.messages[0].eta = self.factor.eta.copy()
            self.messages[0].lam = self.factor.lam.copy()
            return
        
    
        messages_eta, messages_lam = [], []

        for v in range(len(self.adj_vIDs)):
            eta_factor, lam_factor = self.factor.eta.copy(), self.factor.lam.copy()

            # Take product of factor with incoming messages
            mess_start_dim = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_factor[mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[var].eta - self.messages[var].eta
                    lam_factor[mess_start_dim:mess_start_dim + var_dofs, mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[var].lam - self.messages[var].lam
                mess_start_dim += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            divide =  self.adj_var_nodes[0].dofs
            if v == 0:
                eo = eta_factor[:divide]
                eno = eta_factor[divide:]

                loo = lam_factor[:divide, :divide]
                lono = lam_factor[:divide, divide:]
                lnoo = lam_factor[divide:, :divide]
                lnono = lam_factor[divide:, divide:]
            elif v == 1:
                eo = eta_factor[divide:]
                eno = eta_factor[:divide]

                loo = lam_factor[divide:, divide:]
                lono = lam_factor[divide:, :divide]
                lnoo = lam_factor[:divide, divide:]
                lnono = lam_factor[:divide, :divide]

            lnono += 1e-12 * np.eye(lnono.shape[0])
            # concat RHS
            rhs_j = np.concatenate([lnoo, eno.reshape(-1, 1)], axis=1)   # (n, n+1)
            # try-except 注释，仅保留原 try 内容
            L = np.linalg.cholesky(lnono)
            X = scipy.linalg.cho_solve((L, True), rhs_j)    # True: L is lower diagonal
            # except np.linalg.LinAlgError:
            #     # fallback: 用伪逆
            #     X = np.linalg.pinv(lnono) @ rhs_j
            X_lam = X[:, :lnoo.shape[1]]
            X_eta = X[:, -1]

            new_message_lam = loo - lono @ X_lam
            new_message_eta = eo  - lono @ X_eta
            messages_lam.append((1 - eta_damping) * new_message_lam + eta_damping * self.messages[v].lam)
            messages_eta.append((1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta)


        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]


        #time.sleep(0.00000001)


    def compute_messages_from_v(self, v, eta_damping):
        """
            Compute all outgoing messages from the factor.
            This is specialised for one and two variable factors.
        """

        if len(self.adj_vIDs) == 1:
            v = 0
            self.messages[v].eta = self.factor.eta.copy()
            self.messages[v].lam = self.factor.lam.copy()
            return
        
        eta_factor, lam_factor = self.factor.eta.copy(), self.factor.lam.copy()
        # Take product of factor with incoming messages
        mess_start_dim = 0
        # Divide up parameters of distribution
        divide = self.adj_var_nodes[0].dofs
        
        if v == 1:
            u = 0
            mess_start_dim = self.adj_var_nodes[0].dofs
            var_dofs = self.adj_var_nodes[1].dofs
            eta_factor[mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[1].eta - self.messages[1].eta
            lam_factor[mess_start_dim:mess_start_dim + var_dofs, mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[1].lam - self.messages[1].lam

            eo = eta_factor[:divide]
            eno = eta_factor[divide:]

            loo = lam_factor[:divide, :divide]
            lono = lam_factor[:divide, divide:]
            lnoo = lam_factor[divide:, :divide]
            lnono = lam_factor[divide:, divide:]

        elif v == 0:
            u = 1
            mess_start_dim = 0
            var_dofs = self.adj_var_nodes[0].dofs
            eta_factor[mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[0].eta - self.messages[0].eta
            lam_factor[mess_start_dim:mess_start_dim + var_dofs, mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[0].lam - self.messages[0].lam

            eo = eta_factor[divide:]
            eno = eta_factor[:divide]

            loo = lam_factor[divide:, divide:]
            lono = lam_factor[divide:, :divide]
            lnoo = lam_factor[:divide, divide:]
            lnono = lam_factor[:divide, :divide]

        lnono += 1e-12 * np.eye(lnono.shape[0])
        L = np.linalg.cholesky(lnono)

        # concat RHS
        rhs_j = np.concatenate([lnoo, eno.reshape(-1, 1)], axis=1)   # (n, n+1)
        # cho_solve:  lnono_j * X = rhs_j
        X = scipy.linalg.cho_solve((L, True), rhs_j)    # True: L is lower diagonal
        X_lam = X[:, :lnoo.shape[1]]
        X_eta = X[:, -1]

        new_message_lam = loo - lono @ X_lam
        new_message_eta = eo  - lono @ X_eta
        self.messages[u].lam = (1 - eta_damping) * new_message_lam + eta_damping * self.messages[v].lam
        self.messages[u].eta = (1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta
        #time.sleep(0.00000001)    

    def __repr__(self):
        # 在 list / dict / debugger 里显示
        return f"Factor({self.factorID})"

