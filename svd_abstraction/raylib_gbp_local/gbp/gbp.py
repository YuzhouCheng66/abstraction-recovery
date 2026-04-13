"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""

import numpy as np
import time
import scipy.linalg

from utils.gaussian import NdimGaussian
from utils.distances import bhattacharyya, mahalanobis

from amg import classes as amg_cls
from amg import functions as amg_fnc

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
        self.n_edges = 0
        self.n_msgs = 0

        self.nonlinear_factors = nonlinear_factors

        self.eta_damping = eta_damping

        self.Q = []
        self.b_wild = False
        self.wild_thresh = wild_thresh
        self.multigrid_vars = [[]]
        self.multigrid_factors = [[]]
        self.multigrid = False
        self.conv_width = 1
        self.conv_stride = 1
        self.prolongation_eta_mode = "none"
        self.prolongation_eta_eps = 1e-12
        self.prolongation_eta_levels = None
        self.multigrid_coarse_edge_mode = "relative"
        self._sequential_reverse_by_level = {}
        self._hybrid_counter_by_level = {}

        self.energy_history = []
        self.error_history = []
        self.nmsgs_history = []
        self.mus = []

        if nonlinear_factors:
            # For linearising nonlinear measurement factors.
            self.beta = beta  # Threshold change in mean of adjacent beliefs for relinearisation.
            self.num_undamped_iters = num_undamped_iters  # Number of undamped iterations after relinearisation before damping is set to 0.4
            self.min_linear_iters = min_linear_iters  # Minimum number of linear iterations before a factor is allowed to realinearise.

    def set_prolongation_eta_mode(self, mode="none", eps=1e-12, levels=None):
        valid_modes = {
            "none",
            "uniform",
            "message_lam_trace",
            "factor_diag_trace",
            "local_direct",
            "local_iter5",
        }
        if mode not in valid_modes:
            raise ValueError(f"Unknown prolongation eta mode: {mode}")

        self.prolongation_eta_mode = mode
        self.prolongation_eta_eps = eps
        self.prolongation_eta_levels = None if levels is None else tuple(sorted(set(levels)))

        for var in self.var_nodes:
            var.multigrid.prolongation_eta_mode = mode
            var.multigrid.prolongation_eta_eps = eps
            var.multigrid.prolongation_eta_levels = self.prolongation_eta_levels

    def _effective_prolongation_eta_mode(self, level):
        if self.prolongation_eta_levels is None:
            return self.prolongation_eta_mode
        if level in self.prolongation_eta_levels:
            return self.prolongation_eta_mode
        return "none"

    def energy(self, vars=None):
        """
            Computes the sum of all of the squared errors in the graph using the appropriate local loss function.
        """
        # if slice_e is None:
        #     slice_e = slice(len(self.factors))
        # energy = 0
        # for factor in self.factors[slice_e]:
        #     # Variance of Gaussian noise at each factor is weighting of each term in squared loss.
        #     energy += 0.5 * np.linalg.norm(factor.compute_residual()) ** 2
        # return energy
        if vars is None:
            vars = self.var_nodes
        energy = 0
        for var in vars:
            if var.type != "multigrid":
                # Variance of Gaussian noise at each factor is weighting of each term in squared loss.
                energy += 0.5 * np.linalg.norm(var.residual) ** 2
        return energy

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
                    self.n_msgs += 2

    def compute_all_smoothing_messages(self, factors=None, level=None, local_relin=True):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        if level is not None:
            factors = self.multigrid_factors[level]
        for count, factor in enumerate(factors):
            factor.smoothing_compute_messages(self.eta_damping)

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

    def update_all_beliefs_eta_only_fixed_lam(self, vars=None, level=None):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]
        if level is not None:
            vars = self.multigrid_vars[level]

        for var in vars:
            if var.active:
                var.update_belief_eta_only_fixed_lam()

    def update_all_residuals(self, vars=None, level=None, smoothing=False):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]
        if level is not None:
            vars = self.multigrid_vars[level]

        for var in vars:
            if var.active:
                res = var.compute_residual()

    def restrict_all_residuals(self, vars=None, level=None, smoothing=False):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]
        if level is not None:
            vars = self.multigrid_vars[level]

        for var in vars:
            if var.active:
                var.multigrid.send_restricted_residual()

    def update_all_residual_etas(self, vars=None, level=None, smoothing=False):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]
        if level is not None:
            vars = self.multigrid_vars[level]

        for var in vars:
            if var.active:
                if var.type[0:5] == "multi":
                    for i_var in var.multigrid.interpolation_vars:
                        i_var.compute_residual()
                        i_var.multigrid.send_restricted_residual()
                    var.multigrid.update_eta()
                else:
                    print("You just tried to update the eta on a base variable... you should probably \
                        check something because this ain't it!")
                
    def prolongate_corrections(self, vars=None, level=None, smoothing=False):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]
        if level is not None:
            vars = self.multigrid_vars[level]

        affected_vars = []
        for var in vars:
            if var.active:
                if var.type[0:5] == "multi":
                    affected_vars.extend(var.multigrid.send_corrections())
                else:
                    print("You just tried to prolongate using a base variable... you should probably \
                        check something because this ain't it!")

        if self._effective_prolongation_eta_mode(level) == "local_direct":
            self.post_prolongation_local_eta_refresh(affected_vars)
        elif self._effective_prolongation_eta_mode(level) == "local_iter5":
            self.post_prolongation_local_eta_refresh(affected_vars, n_eta_iters=5)
        elif self._effective_prolongation_eta_mode(level) == "cached_iter1":
            self.post_prolongation_cached_eta_refresh(affected_vars)

        return affected_vars

    def post_prolongation_local_eta_refresh(self, vars, n_eta_iters=1):
        seen_factor_ids = set()
        factors = []
        for var in vars:
            if not var.active:
                continue
            for factor in var.adj_factors:
                if not factor.active or factor.factorID in seen_factor_ids:
                    continue
                if len(factor.adj_vIDs) != 2:
                    continue
                seen_factor_ids.add(factor.factorID)
                factors.append(factor)

        if n_eta_iters <= 1:
            for factor in factors:
                factor.local_direct_eta_refresh()
            return

        for _ in range(n_eta_iters):
            for factor in factors:
                factor.compute_messages_eta_only_fixed_lam(self.eta_damping)

    def cache_eta_only_refresh_response(self, factors=None, level=None):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        if level is not None:
            factors = self.multigrid_factors[level]

        for factor in factors:
            if factor.active and len(factor.adj_vIDs) == 2:
                factor.cache_eta_only_fixed_lam_response()

    def post_prolongation_cached_eta_refresh(self, vars, n_eta_iters=1):
        seen_factor_ids = set()
        factors = []
        touched_vars = []
        touched_ids = set()
        for var in vars:
            if not var.active:
                continue
            for factor in var.adj_factors:
                if not factor.active or factor.factorID in seen_factor_ids:
                    continue
                if len(factor.adj_vIDs) != 2:
                    continue
                seen_factor_ids.add(factor.factorID)
                factors.append(factor)
                for adj_var in factor.adj_var_nodes:
                    if adj_var.active and adj_var.variableID not in touched_ids:
                        touched_ids.add(adj_var.variableID)
                        touched_vars.append(adj_var)

        for factor in factors:
            factor.cache_eta_only_fixed_lam_response()

        for _ in range(n_eta_iters):
            for factor in factors:
                factor.compute_messages_eta_only_cached(self.eta_damping)

        self.update_all_beliefs_eta_only_fixed_lam(vars=touched_vars)

    def compute_all_factors(self, factors=None, level=None):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        if level is not None:
            factors = self.multigrid_factors[level]
        for count, factor in enumerate(factors):
            factor.compute_factor()

    def relinearise_factors(self, factors=None):
        """
            Compute the factor distribution for all factors for which the local belief mean has deviated a distance
            greater than beta from the current linearisation point.
            Relinearisation is only allowed at a maximum frequency of once every min_linear_iters iterations.
        """
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        if self.nonlinear_factors:
            for factor in factors:
                adj_belief_means = np.array([])
                for belief in factor.adj_beliefs:
                    adj_belief_means = np.concatenate((adj_belief_means, 1/np.diagonal(belief.lam) * belief.eta))
                if np.linalg.norm(factor.linpoint - adj_belief_means) > self.beta and factor.iters_since_relin >= self.min_linear_iters:
                    factor.compute_factor(linpoint=adj_belief_means)
                    factor.iters_since_relin = 0
                    factor.eta_damping = 0.0
                else:
                    factor.iters_since_relin += 1

    def robustify_all_factors(self, factors=None):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]
        for factor in factors:
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
        time.sleep(1e-9)
        self.update_all_beliefs(vars)

    def sequential_iteration(self, factors=None, level=None, local_relin=True, robustify=False):
        if level is not None:
            vars = self.multigrid_vars[level]
            factors = self.multigrid_factors[level]
        else:
            vars = self.var_nodes[:self.n_var_nodes]
            factors = self.factors[:self.n_factor_nodes]

        scope_key = "base" if level is None else level
        reverse = self._sequential_reverse_by_level.get(scope_key, False)
        factors = sorted(
            factors,
            key=lambda factor: (
                min(factor.adj_vIDs) if factor.adj_vIDs else -1,
                max(factor.adj_vIDs) if factor.adj_vIDs else -1,
                factor.factorID,
            ),
            reverse=reverse,
        )
        self._sequential_reverse_by_level[scope_key] = not reverse

        if robustify:
            self.robustify_all_factors(factors)
        if self.nonlinear_factors and local_relin:
            self.relinearise_factors(factors)

        touched_vars = set()
        for factor in factors:
            if not factor.active:
                continue
            if self.nonlinear_factors and local_relin:
                if factor.iters_since_relin == self.num_undamped_iters:
                    factor.eta_damping = self.eta_damping
                factor.compute_messages(factor.eta_damping)
            else:
                factor.compute_messages(self.eta_damping)
                self.n_msgs += 2

            for var in factor.adj_var_nodes:
                if not var.active:
                    continue
                var.update_belief()
                touched_vars.add(var.variableID)

        return [var for var in vars if var.variableID in touched_vars]

    def _run_iteration(self, level=None, mode="synchronous", local_relin=True, robustify=False):
        scope_key = "base" if level is None else level
        if mode == "seq4_sync1":
            hybrid_count = self._hybrid_counter_by_level.get(scope_key, 0)
            resolved_mode = "synchronous" if hybrid_count % 5 == 4 else "sequential"
            self._hybrid_counter_by_level[scope_key] = hybrid_count + 1
            mode = resolved_mode
        if mode == "synchronous":
            self.synchronous_iteration(level=level, local_relin=local_relin, robustify=robustify)
            return
        if mode == "sequential":
            self.sequential_iteration(level=level, local_relin=local_relin, robustify=robustify)
            return
        raise ValueError(f"Unknown smoother mode: {mode}")

    def synchronous_smooth(self, level=None, local_relin=True, robustify=False):
        if level is not None:
            vars = self.multigrid_vars[level]
            factors = self.multigrid_factors[level]
        else:
            vars = self.var_nodes
            factors = self.factors
        if robustify:
            self.robustify_all_factors()
        if self.nonlinear_factors and local_relin:
            self.relinearise_factors()
        self.compute_all_smoothing_messages(local_relin=local_relin)
        self.update_all_beliefs(smoothing=True)
            

    def synchronous_loop(self, vis):
        i=0
        # self.get_means()
        while not vis.reset_event.isSet(): #i<1000 and 
            while vis.pause_event.isSet() and not vis.reset_event.isSet():
                time.sleep(0.5)

            self.visualisation_sync(vis)

            self.synchronous_iteration()
            self.update_all_residuals()

            i+=1
            av_dist = np.mean(np.linalg.norm(np.array([var.mu - var.GT for var in self.var_nodes if var.type != "multigrid"]),axis=1))
            self.energy_history.append(self.energy())
            self.error_history.append(av_dist)
            self.nmsgs_history.append(self.n_msgs)
            print(f'Iteration {i}  // Energy {self.energy_history[-1]:.6f} // ' 
                  f'Average error {av_dist:.4f} // msgs sent {self.n_msgs/1e6:.3f}x10^6')
            
            self.get_multigrid_stats()
            for level in range(len(self.n_active)):
                if self.n_active[level] > 0:
                    print(f'Multigrid stats // level {level} // {(self.n_coarse[level]/(len(self.multigrid_vars[level])))*100:.2f}% coarse ' \
                        f'// {(self.n_active[level]/(len(self.multigrid_vars[level])))*100:.2f}% active ' \
                        f'// {len(self.multigrid_vars[level])} total ')
            
            print('')        
            if vis.skip_event.isSet():
                vis.pause_event.set()
                vis.skip_event.clear()

    def joint_distribution_inf_level(self, level):
        vars = [var for var in self.multigrid_vars[level] if var.type != "dead"]
        factors = [factor for factor in self.multigrid_factors[level] if factor.type != "dead"]

        total_dofs = sum(var.dofs for var in vars)
        eta = np.zeros(total_dofs)
        lam = np.zeros((total_dofs, total_dofs))
        var_ix = {}

        offset = 0
        for var in vars:
            var_ix[var.variableID] = offset
            eta[offset:offset + var.dofs] = var.prior.eta
            lam[offset:offset + var.dofs, offset:offset + var.dofs] = var.prior.lam
            offset += var.dofs

        for factor in factors:
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                if adj_var_node.variableID not in var_ix:
                    factor_ix += adj_var_node.dofs
                    continue

                v_id = adj_var_node.variableID
                start = var_ix[v_id]
                stop = start + adj_var_node.dofs

                eta[start:stop] += factor.factor.eta[factor_ix:factor_ix + adj_var_node.dofs]
                lam[start:stop, start:stop] += factor.factor.lam[
                    factor_ix:factor_ix + adj_var_node.dofs,
                    factor_ix:factor_ix + adj_var_node.dofs,
                ]

                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if (
                        other_adj_var_node.variableID in var_ix
                        and other_adj_var_node.variableID > adj_var_node.variableID
                    ):
                        other_start = var_ix[other_adj_var_node.variableID]
                        other_stop = other_start + other_adj_var_node.dofs
                        lam[start:stop, other_start:other_stop] += factor.factor.lam[
                            factor_ix:factor_ix + adj_var_node.dofs,
                            other_factor_ix:other_factor_ix + other_adj_var_node.dofs,
                        ]
                        lam[other_start:other_stop, start:stop] += factor.factor.lam[
                            other_factor_ix:other_factor_ix + other_adj_var_node.dofs,
                            factor_ix:factor_ix + adj_var_node.dofs,
                        ]
                    other_factor_ix += other_adj_var_node.dofs
                factor_ix += adj_var_node.dofs

        return eta, lam

    def set_level_mean_vector(self, level, mean_vector):
        vars = [var for var in self.multigrid_vars[level] if var.type != "dead"]

        offset = 0
        for var in vars:
            local_mu = np.asarray(mean_vector[offset:offset + var.dofs]).reshape(-1)
            offset += var.dofs
            var.mu = local_mu.copy()
            var.Sigma = 1 / np.diagonal(var.belief.lam)
            var.belief.eta = var.belief.lam @ var.mu

            for factor in var.adj_factors:
                belief_ix = factor.adj_vIDs.index(var.variableID)
                factor.adj_beliefs[belief_ix].eta = var.belief.eta
                factor.adj_beliefs[belief_ix].lam = var.belief.lam

    def direct_solve_level(self, level, ridge=1e-10):
        eta, lam = self.joint_distribution_inf_level(level)
        if lam.size == 0:
            mu = np.zeros(0)
        else:
            stabilized = 0.5 * (lam + lam.T) + ridge * np.eye(lam.shape[0])
            try:
                chol, lower = scipy.linalg.cho_factor(stabilized, lower=False, check_finite=False)
                mu = scipy.linalg.cho_solve((chol, lower), eta)
            except np.linalg.LinAlgError:
                mu = np.linalg.solve(stabilized, eta)

        self.set_level_mean_vector(level, mu)
        return mu

    def vcycle_step(
        self,
        top_level_solver="iterative",
        top_level_ridge=1e-10,
        base_smoother="synchronous",
        coarse_smoother="synchronous",
    ):
        if top_level_solver not in {"iterative", "direct"}:
            raise ValueError(f"Unknown top_level_solver: {top_level_solver}")
        if base_smoother not in {"synchronous", "sequential", "seq4_sync1"}:
            raise ValueError(f"Unknown base_smoother: {base_smoother}")
        if coarse_smoother not in {"synchronous", "sequential", "seq4_sync1"}:
            raise ValueError(f"Unknown coarse_smoother: {coarse_smoother}")

        top_level = len(self.multigrid_vars) - 1

        for _ in range(1):
            self._run_iteration(level=0, mode=base_smoother)
        if top_level >= 1 and self._effective_prolongation_eta_mode(1) == "cached_iter1":
            self.cache_eta_only_refresh_response(level=0)

        for level in range(1, len(self.multigrid_vars)):
            self.update_all_residual_etas(level=level)
            self.update_all_beliefs(level=level)

            if level == top_level and top_level_solver == "direct":
                self.direct_solve_level(level, ridge=top_level_ridge)
            else:
                for _ in range(1):
                    self._run_iteration(level=level, mode=coarse_smoother)
            self.update_all_residuals(level=level)

        for level in range(len(self.multigrid_vars) - 1, 0, -1):
            if not (level == top_level and top_level_solver == "direct"):
                for _ in range(1):
                    self._run_iteration(level=level, mode=coarse_smoother)
            self.prolongate_corrections(level=level)


    def vcycle_loop(self, vis, top_level_solver="iterative", top_level_ridge=1e-10):
        i=0
        # self.get_means()

        while  not vis.reset_event.isSet():
            while vis.pause_event.isSet() and not vis.reset_event.isSet():
                time.sleep(0.5)

            self.visualisation_sync(vis)

            # if i == 10:  # Number of damped iterations before applying undamping
            #     self.eta_damping = 0.0

            self.vcycle_step(top_level_solver=top_level_solver, top_level_ridge=top_level_ridge)

            i+=1
            av_dist = np.mean(np.linalg.norm(np.array([var.mu - var.GT for var in self.var_nodes if var.type != "multigrid"]),axis=1))
            self.energy_history.append(self.energy())
            self.error_history.append(av_dist)
            self.nmsgs_history.append(self.n_msgs)
            print(f'Iteration {i}  // Energy {self.energy_history[-1]:.6f} // ' 
                  f'Average error {av_dist:.4f} // msgs sent {self.n_msgs/1e6:.3f}x10^6')
            
            self.get_multigrid_stats()
            for level in range(len(self.n_active)):
                if self.n_active[level] > 0:
                    print(f'Multigrid stats // level {level} // {(self.n_coarse[level]/(len(self.multigrid_vars[level])))*100:.2f}% coarse ' \
                        f'// {(self.n_active[level]/(len(self.multigrid_vars[level])))*100:.2f}% active ' \
                        f'// {len(self.multigrid_vars[level])} total ')
            
            print('')
            if vis.skip_event.isSet():
                vis.pause_event.set()
                vis.skip_event.clear()

    def wildfire_iteration(self, vis, local_relin=True, robustify=False):
        breakout_count = 0
        i = 0
        while not vis.reset_event.isSet():
            if vis.pause_event.isSet() and not vis.reset_event.isSet() or not self.Q:
                time.sleep(0.1)
                _ , new_factors = self.visualisation_sync(vis)
                if new_factors:
                    self.Q = new_factors
            else:
                _ , new_factors = self.visualisation_sync(vis)
                if new_factors:
                    self.Q[0:0] = new_factors

                self.Q[0].compute_messages(self.eta_damping)
                self.n_msgs += 2

                for count, var in enumerate(self.Q[0].adj_var_nodes):
                    var.update_belief()
                    var.compute_residual()
                    breakout_count += 1
                    if any(self.Q[0].messages_dist[count] > self.wild_thresh):
                        for f in var.adj_factors:
                            if f not in self.Q:
                                self.Q.append(f)

                self.Q.pop(0)

                if (self.n_msgs / 2) % len(self.factors) == 0:
                    i += 1
                    vis.read_event.clear()
                    av_dist = np.mean(np.linalg.norm(np.array([var.mu - var.GT for var in self.var_nodes if var.type != "multigrid"]),axis=1))
                    self.energy_history.append(self.energy())
                    self.error_history.append(av_dist)
                    self.nmsgs_history.append(self.n_msgs)
                    print(f'Iteration {i}  // Energy {self.energy_history[-1]:.6f} // ' 
                        f'Average error {av_dist:.4f} // msgs sent {self.n_msgs/1e6:.3f}x10^6')
                    
                    self.get_multigrid_stats()
                    for level in range(len(self.n_active)):
                        if self.n_active[level] > 0:
                            print(f'Multigrid stats // level {level} // {(self.n_coarse[level]/(len(self.multigrid_vars[level])))*100:.2f}% coarse ' \
                                f'// {(self.n_active[level]/(len(self.multigrid_vars[level])))*100:.2f}% active ' \
                                f'// {len(self.multigrid_vars[level])} total ')
                    
                    print('')

                    breakout_count = 0

                    if vis.skip_event.isSet():
                        vis.pause_event.set()
                        vis.skip_event.clear()

    def visualisation_sync(self, vis):
        if vis.n_factors > self.n_factor_nodes:
            while vis.write_event.is_set():
                time.sleep(0.001)
            vis.read_event.set()
            new_n_factors = vis.n_factors - self.n_factor_nodes
            new_n_vars = vis.n_vars - self.n_var_nodes
            
            new_vars = self.var_nodes[slice(int(len(self.var_nodes) - new_n_vars), int(len(self.var_nodes)))]
            self.multigrid_vars[0].extend(new_vars)
            new_factors = self.factors[slice(int(len(self.factors) - new_n_factors), int(len(self.factors)))]
            self.multigrid_factors[0].extend(new_factors)

            vars_to_update = []

            for factor in new_factors:
                if vis.b_wild:
                    factor.b_calc_mess_dist = True
                for adj_var in factor.adj_var_nodes:
                    adj_var.adj_factors.append(factor)
                    if adj_var not in vars_to_update:
                        vars_to_update.append(adj_var)
                
            for var in vars_to_update:
                var.update_belief()
            
            for factor in new_factors:
                factor.compute_factor()

            self.n_var_nodes = int(vis.n_vars)
            self.n_factor_nodes = int(vis.n_factors)

            if new_vars and vis.b_multi: # i.e. if there are new vars
                amg_fnc.coarsen_graph(self, vars_to_update)


            vis.n_vars = int(self.n_var_nodes)
            vis.n_factors = int(self.n_factor_nodes)
            
            self.n_vars_active = int(len(self.var_nodes))
            nodes_removed = 0

            for var_id in range(self.n_vars_active):
                if vis.var_nodes[var_id - nodes_removed].type == 'dead':
                    self.multigrid_vars[vis.var_nodes[var_id - nodes_removed].multigrid.level].remove(vis.var_nodes[var_id - nodes_removed])
                    vis.var_nodes.pop(var_id - nodes_removed)
                    nodes_removed += 1

            self.n_vars_active = int(len(self.var_nodes))
            self.n_factors_active = int(len(self.factors))

            factors_removed = 0
            for factor_id in range(self.n_factors_active):
                if vis.factors[factor_id - factors_removed].type == 'dead':
                    self.multigrid_factors[int(vis.factors[factor_id - factors_removed].level)].remove(vis.factors[factor_id - factors_removed])
                    vis.factors.pop(factor_id - factors_removed)
                    factors_removed += 1

            self.n_factors_active = int(len(self.factors))

            # print("{:} node(s) removed : {:} factor(s) removed".format(nodes_removed ,factors_removed))
            # self.n_var_nodes = int(vis.n_vars)
            # self.n_factor_nodes = int(vis.n_factors)

            vis.read_event.clear()

            return new_vars, new_factors
                
        else:

            return None, None
    
    def joint_distribution_inf(self):
        """
            Get the joint distribution over all variables in the information form
            If nonlinear factors, it is taken at the current linearisation point.
        """

        eta = np.array([])
        lam = np.array([])
        var_ix = np.zeros(len(self.var_nodes)).astype(int)
        tot_n_vars = 0
        for var_node in self.var_nodes:
            var_ix[var_node.variableID] = int(tot_n_vars)
            tot_n_vars += var_node.dofs
            eta = np.concatenate((eta, var_node.prior.eta))
            if var_node.variableID == 0:
                lam = var_node.prior.lam
            else:
                lam = scipy.linalg.block_diag(lam, var_node.prior.lam)

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
    
    def get_multigrid_stats(self):
        self.n_coarse = [0 for _ in self.multigrid_vars]
        self.n_active = [0 for _ in self.multigrid_vars]
        self.n_fine = [0 for _ in self.multigrid_vars]
        for var in self.var_nodes:
            if var.active:
                self.n_active[var.multigrid.level] += 1
            if var.multigrid.classification == "coarse":
                self.n_coarse[var.multigrid.level] += 1
            elif var.multigrid.classification == "fine":
                self.n_fine[var.multigrid.level] += 1



    # def get_means(self, slice_m=None):
    #     """
    #         Get an array containing all current estimates of belief means.
    #     """
    #     if slice_m is None:
    #         slice_m = slice(0,len(self.var_nodes),1)
    #     if len(self.mus) != len(self.var_nodes):
    #         self.mus = [None]*(len(self.var_nodes))

    #     for index, var_node in enumerate(self.var_nodes[slice_m]):
    #         self.mus[index] =  [var_node.mu[0],var_node.mu[1]]


    #     return self.mus
    
    def get_sigmas(self):
        """
            Get an array containing all current estimates of belief sigmas.
        """
        sigmas = np.array([])
        for var_node in self.var_nodes:
            sigmas = np.concatenate((sigmas, var_node.Sigma[0]))
        return sigmas
    
    def get_residuals(self, level=None):
        """
            Get an array containing all current estimates of belief means.
        """
        if level is not None:
            slice_v = slice(int(self.multigrid_vars[level].var_ids[0]), int(self.multigrid_vars[level].var_ids[-1]+1))
        else:
            slice_v = slice(int(len(self.var_nodes)))

        for var in self.var_nodes[slice_v]:
            var.residual = np.zeros(var.dofs)
            for factor in var.adj_factors:
                residual = factor.compute_residual() * (1 - 2 * int(factor.adj_vIDs[1] == var.variableID))
                var.residual += residual[:2]

    def get_var_residuals(self, level=None):
        if level is not None:
            slice_v = slice(int(self.multigrid_vars[level].var_ids[0]), int(self.multigrid_vars[level].var_ids[-1]+1))
        else:
            slice_v = slice(int(len(self.var_nodes)))

        for var in self.var_nodes[slice_v]:
            var.residual = var.compute_residual()

class VariableNode:
    def __init__(self,
                 variable_id,
                 dofs, level=0):

        self.variableID = variable_id
        self.adj_factors = []
        self.InfoMat = []  # Row vector of prior Information vector in factor order
        self.EtaVec = []  # Vector of prior eta values
        self.multigrid = amg_cls.mutligrid_var_info(self)
        self.type = "None specified"
        self.active = True

        # Node variables are position of landmark in world frame. Initialize variable nodes at origin
        self.mu = np.zeros(dofs)
        self.Sigma = np.zeros([dofs, dofs])
        self.residual = np.zeros(dofs)

        self.belief = NdimGaussian(dofs)

        self.prior = NdimGaussian(dofs)
        self.prior_lambda_end = -1  # -1 flag if the sigma of self.prior is prior_sigma_end
        self.prior_lambda_logdiff = -1

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
        self.Sigma = 1/np.diagonal(self.belief.lam)
        self.mu = self.Sigma * self.belief.eta
        
        # Send belief to adjacent factors
        for factor in self.adj_factors:
            belief_ix = factor.adj_var_nodes.index(self)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = self.belief.eta, self.belief.lam

    def update_belief_eta_only_fixed_lam(self):
        """
            Eta-only belief refresh with lam-side held fixed.

            This rebuilds the variable eta from the current incoming message eta
            values while keeping lam unchanged except for a numerically identical
            recomputation from the frozen lam-side messages.
        """
        eta = self.prior.eta.copy()
        lam = self.prior.lam.copy()
        for factor in self.adj_factors:
            message_ix = factor.adj_var_nodes.index(self)
            eta_inward, lam_inward = factor.messages[message_ix].eta, factor.messages[message_ix].lam
            eta += eta_inward
            lam += lam_inward

        self.belief.eta = eta
        self.belief.lam = lam
        self.Sigma = 1/np.diagonal(self.belief.lam)
        self.mu = self.Sigma * self.belief.eta

        for factor in self.adj_factors:
            belief_ix = factor.adj_var_nodes.index(self)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = self.belief.eta, self.belief.lam

    def update_smooth_belief(self):
        """
            Update local belief estimate by taking product of all incoming messages along all edges.
            Then send belief to adjacent factor nodes.
        """
        # Update local belief
        eta = self.prior.eta.copy()
        lam = self.prior.lam.copy()
        for factor in self.adj_factors:
            message_ix = factor.adj_vIDs.index(self.variableID)
            mu_inward, lam_inward = factor.messages[message_ix].eta, factor.messages[message_ix].lam
            factor.messages_prior[message_ix].eta = mu_inward  # Update messages for belief calculation now sync itr is complete
            factor.messages_prior[message_ix].lam = lam_inward  # If don't have a prior messages then its not truely parallel
            eta += np.diag(lam_inward) * mu_inward
            lam += lam_inward

        self.belief.eta = eta 
        self.belief.lam = lam
        self.Sigma = 1/np.diagonal(self.belief.lam)
        self.mu = self.Sigma * self.belief.eta
        
        # Send belief to adjacent factors
        for factor in self.adj_factors:
            belief_ix = factor.adj_vIDs.index(self.variableID)
            factor.adj_beliefs[belief_ix].eta, factor.adj_beliefs[belief_ix].lam = self.belief.eta, self.belief.lam

    def compute_residual(self):
        
        # Ax = np.zeros(2)

        # Ax += self.prior.lam @ self.mu

        # for factor in self.adj_factors:
        #     for var in factor.adj_var_nodes:
        #         if var.variableID != self.variableID:
        #             Ax += var.mu @ factor.factor.lam[0:2, 2:4]

        # d = self.prior.eta - Ax

        # res = self.prior.eta - self.prior.lam @ self.mu

        # for factor in self.adj_factors:
        #     # get factor residual. Second part flips the sign if the var is second var in the factor
        #     res += factor.compute_residual() * (1 - 2 * int(factor.adj_vIDs[1] == self.variableID))

        # self.residual = res

        res = self.prior.eta - self.prior.lam @ self.mu
        for factor in self.adj_factors:
            if factor.adj_vIDs.index(self.variableID) == 0:
                 res += factor.factor.eta[:self.dofs] - (factor.factor.lam[:self.dofs, :self.dofs] @ self.mu \
                        + factor.factor.lam[:self.dofs, self.dofs:] @ factor.adj_var_nodes[1].mu)
            else:
                res += factor.factor.eta[-self.dofs:] - (factor.factor.lam[-self.dofs:, -self.dofs:] @ self.mu  \
                        + factor.factor.lam[-self.dofs:, :-self.dofs] @ factor.adj_var_nodes[0].mu)
        
        self.residual = res

        return res


class Factor:
    def __init__(self,
                 factor_id,
                 adj_var_nodes,
                 measurement,
                 gauss_noise_std,
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

        for adj_var_node in self.adj_var_nodes:
            self.dofs_conditional_vars += adj_var_node.dofs
            self.adj_vIDs.append(adj_var_node.variableID)
            self.adj_beliefs.append(NdimGaussian(adj_var_node.dofs))
            self.messages.append(NdimGaussian(adj_var_node.dofs))#, eta=adj_var_node.prior.eta, lam=adj_var_node.prior.lam))
            self.messages_prior.append(NdimGaussian(adj_var_node.dofs))
            self.messages_dist.append(np.zeros(adj_var_node.dofs))

        self.factor = NdimGaussian(self.dofs_conditional_vars)
        self.linpoint = np.zeros(self.dofs_conditional_vars)  # linearisation point

        self.measurement = measurement

        self.residual = None

        self.b_calc_mess_dist = wildfire

        # Measurement model
        self.gauss_noise_var = gauss_noise_std**2
        self.meas_fn = meas_fn
        self.jac_fn = jac_fn
        self.args = args

        # Robust loss function
        self.adaptive_gauss_noise_var = gauss_noise_std**2
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
            self.linpoint = []
            for belief in self.adj_beliefs:
                self.linpoint += list(1/np.diagonal(belief.lam) * belief.eta)
        else:
            self.linpoint = linpoint

        if isinstance(self.jac_fn, list):
            J = np.array(self.jac_fn)
            pred_measurement = J @ self.linpoint
        else:
            J = self.jac_fn(self.linpoint, *self.args)
            pred_measurement = self.meas_fn(self.linpoint, *self.args)
        if isinstance(self.measurement, float):
            meas_model_lambda = 1 / self.adaptive_gauss_noise_var
            lambda_factor = meas_model_lambda * np.outer(J, J)
            eta_factor = meas_model_lambda * J.T * (J @ self.linpoint + self.measurement - pred_measurement)
        else:
            meas_model_lambda = np.eye(len(self.measurement)) / self.adaptive_gauss_noise_var
            lambda_factor = J.T @ meas_model_lambda @ J
            eta_factor = (J.T @ meas_model_lambda) @ (J @ self.linpoint + self.measurement - pred_measurement)

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
        """
        if self.type[0:5] == "multi":
            eta_damping = eta_damping
        messages_eta, messages_lam = [], []
        start_dim = 0
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
            var_side = 1+v*-2
            mess_dofs = self.adj_var_nodes[v].dofs
            adj_var_start_dim = var_side * start_dim + mess_dofs

            eo = eta_factor[start_dim:start_dim + mess_dofs]
            eno = eta_factor[adj_var_start_dim: adj_var_start_dim + mess_dofs]

            loo = lam_factor[start_dim:start_dim + mess_dofs, start_dim:start_dim + mess_dofs]
            lono = lam_factor[start_dim:start_dim + mess_dofs, adj_var_start_dim : adj_var_start_dim + mess_dofs]
            lnoo = lam_factor[adj_var_start_dim:adj_var_start_dim + mess_dofs, start_dim:start_dim + mess_dofs]
            lnono = lam_factor[adj_var_start_dim:adj_var_start_dim + mess_dofs, adj_var_start_dim:adj_var_start_dim + mess_dofs]
            # lono = np.hstack((lam_factor[start_dim:start_dim + mess_dofs, :start_dim],
            #                   lam_factor[start_dim:start_dim + mess_dofs, start_dim + mess_dofs:]))
            # lnoo = np.vstack((lam_factor[:start_dim, start_dim:start_dim + mess_dofs],
            #                   lam_factor[start_dim + mess_dofs:, start_dim:start_dim + mess_dofs]))
            # lnono = np.block([[lam_factor[:start_dim, :start_dim], lam_factor[:start_dim, start_dim + mess_dofs:]],
            #                   [lam_factor[start_dim + mess_dofs:, :start_dim], lam_factor[start_dim + mess_dofs:, start_dim + mess_dofs:]]])

            # Compute outgoing messages
            new_message_lam = loo - lono @ np.linalg.inv(lnono) @ lnoo
            messages_lam.append((1 - eta_damping) * new_message_lam + eta_damping * self.messages[v].lam)
            new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno
            messages_eta.append((1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta)
            start_dim += self.adj_var_nodes[v].dofs

        
        for v in range(len(self.adj_vIDs)):
            #self.messages_dist[v] = bhattacharyya(self.messages[v], NdimGaussian(len(messages_eta[v]), eta=messages_eta[v], lam=messages_lam[v]))
            if self.b_calc_mess_dist:
                self.messages_dist[v] = mahalanobis(self.messages[v], NdimGaussian(len(messages_eta[v]), eta=messages_eta[v], lam=messages_lam[v]))
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]

        #time.sleep(0.00000001)

    def local_direct_eta_refresh(self):
        """
            Refresh eta messages while keeping factor/belief/message lam fixed.

            This treats the adjacent beliefs as clamped and solves the two-way
            eta-message fixed point for a binary factor in one local 4x4 solve
            (or more generally (d_i + d_j)-dimensional solve).
        """
        if len(self.adj_vIDs) != 2:
            raise ValueError("local_direct_eta_refresh only supports binary factors")

        dofs0 = self.adj_var_nodes[0].dofs
        dofs1 = self.adj_var_nodes[1].dofs
        sl0 = slice(0, dofs0)
        sl1 = slice(dofs0, dofs0 + dofs1)

        lam_01 = self.factor.lam[sl0, sl1]
        lam_10 = self.factor.lam[sl1, sl0]
        lam_11 = self.factor.lam[sl1, sl1] + self.adj_beliefs[1].lam - self.messages[1].lam
        lam_00 = self.factor.lam[sl0, sl0] + self.adj_beliefs[0].lam - self.messages[0].lam

        coupling_01 = lam_01 @ np.linalg.inv(lam_11)
        coupling_10 = lam_10 @ np.linalg.inv(lam_00)

        rhs_0 = self.factor.eta[sl0] - coupling_01 @ (self.factor.eta[sl1] + self.adj_beliefs[1].eta)
        rhs_1 = self.factor.eta[sl1] - coupling_10 @ (self.factor.eta[sl0] + self.adj_beliefs[0].eta)

        system = np.block(
            [
                [np.eye(dofs0), -coupling_01],
                [-coupling_10, np.eye(dofs1)],
            ]
        )
        rhs = np.concatenate((rhs_0, rhs_1))
        solved = np.linalg.solve(system, rhs)

        self.messages[0].eta = solved[:dofs0]
        self.messages[1].eta = solved[dofs0:dofs0 + dofs1]

    def cache_eta_only_fixed_lam_response(self):
        """
            Cache the fixed-lam eta-only response matrices for a binary factor.

            The cache is valid as long as factor/message/belief lam blocks remain
            unchanged. In the intended use, this is populated immediately after a
            full pre-sweep and then reused for a light eta-only post refresh.
        """
        if len(self.adj_vIDs) != 2:
            raise ValueError("cache_eta_only_fixed_lam_response only supports binary factors")

        dofs0 = self.adj_var_nodes[0].dofs
        dofs1 = self.adj_var_nodes[1].dofs
        sl0 = slice(0, dofs0)
        sl1 = slice(dofs0, dofs0 + dofs1)

        lam_01 = self.factor.lam[sl0, sl1]
        lam_10 = self.factor.lam[sl1, sl0]
        lam_11 = self.factor.lam[sl1, sl1] + self.adj_beliefs[1].lam - self.messages[1].lam
        lam_00 = self.factor.lam[sl0, sl0] + self.adj_beliefs[0].lam - self.messages[0].lam

        self._eta_only_cached_couplings = [
            lam_01 @ np.linalg.inv(lam_11),
            lam_10 @ np.linalg.inv(lam_00),
        ]

    def compute_messages_eta_only_cached(self, eta_damping):
        """
            Eta-only message update reusing cached fixed-lam response matrices.
        """
        if len(self.adj_vIDs) != 2:
            raise ValueError("compute_messages_eta_only_cached only supports binary factors")

        couplings = getattr(self, "_eta_only_cached_couplings", None)
        if couplings is None:
            self.cache_eta_only_fixed_lam_response()
            couplings = self._eta_only_cached_couplings

        dofs0 = self.adj_var_nodes[0].dofs
        dofs1 = self.adj_var_nodes[1].dofs
        sl0 = slice(0, dofs0)
        sl1 = slice(dofs0, dofs0 + dofs1)

        new_eta_0 = self.factor.eta[sl0] - couplings[0] @ (
            self.factor.eta[sl1] + self.adj_beliefs[1].eta - self.messages[1].eta
        )
        new_eta_1 = self.factor.eta[sl1] - couplings[1] @ (
            self.factor.eta[sl0] + self.adj_beliefs[0].eta - self.messages[0].eta
        )

        self.messages[0].eta = (1 - eta_damping) * new_eta_0 + eta_damping * self.messages[0].eta
        self.messages[1].eta = (1 - eta_damping) * new_eta_1 + eta_damping * self.messages[1].eta

    def compute_messages_eta_only_fixed_lam(self, eta_damping):
        """
            Eta-only message update with all lam-side terms frozen.

            This matches one synchronous eta-message iteration under clamped
            adjacent beliefs: only message eta values are updated, while all
            factor/message/belief lam blocks are kept fixed.
        """
        messages_eta = []
        start_dim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor = self.factor.eta.copy()
            lam_factor = self.factor.lam.copy()

            mess_start_dim = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_factor[mess_start_dim:mess_start_dim + var_dofs] += (
                        self.adj_beliefs[var].eta - self.messages[var].eta
                    )
                    lam_factor[
                        mess_start_dim:mess_start_dim + var_dofs,
                        mess_start_dim:mess_start_dim + var_dofs,
                    ] += self.adj_beliefs[var].lam - self.messages[var].lam
                mess_start_dim += self.adj_var_nodes[var].dofs

            var_side = 1 + v * -2
            mess_dofs = self.adj_var_nodes[v].dofs
            adj_var_start_dim = var_side * start_dim + mess_dofs

            eo = eta_factor[start_dim:start_dim + mess_dofs]
            eno = eta_factor[adj_var_start_dim: adj_var_start_dim + mess_dofs]
            lono = lam_factor[start_dim:start_dim + mess_dofs, adj_var_start_dim: adj_var_start_dim + mess_dofs]
            lnono = lam_factor[
                adj_var_start_dim:adj_var_start_dim + mess_dofs,
                adj_var_start_dim:adj_var_start_dim + mess_dofs,
            ]

            new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno
            messages_eta.append((1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta)
            start_dim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].eta = messages_eta[v]

    def smoothing_compute_messages(self, eta_damping):


        for v in range(len(self.adj_vIDs)):
            # Pii = np.array([self.adj_var_nodes[v].prior.lam[0,0], self.adj_var_nodes[v].prior.lam[1,1]])
            # uii = self.adj_var_nodes[v].prior.eta / Pii

            # var_dofs = self.adj_var_nodes[v].dofs
             
            # Pki_sum = 0
            # uki_sum = 0

            # for factor in self.adj_var_nodes[v].adj_factors:
            #     if factor.factorID is not self.factorID:
            #         if factor.adj_vIDs[0] is self.adj_vIDs[v]:
            #             v_ix = 0
            #         else:
            #             v_ix = 1
                    
            #         Pki = np.array([factor.messages[v_ix].lam[0,0], factor.messages[v_ix].lam[1,1]])
            #         uki = factor.messages[v_ix].eta
                 
            #         Pki_sum += Pki
            #         uki_sum += uki * Pki
                     
            # Aij = np.array([self.factor.lam[0,2],self.factor.lam[1,3]])
            # Pij = (-Aij**2) / (Pii + Pki_sum)
            # uij = ((Pii * uii) + uki_sum) / Aij

            # self.messages[1-v].lam[0,0] = Pij[0]
            # self.messages[1-v].lam[1,1] = Pij[1]
            # self.messages[1-v].eta = uij 

            Aij = np.array([self.factor.lam[0,2],self.factor.lam[1,3]])
            Pij = (-Aij**2) / (np.diagonal(self.adj_beliefs[v].lam - self.messages_prior[v].lam))
            uij = (self.adj_beliefs[v].eta - self.messages_prior[v].eta * np.diag(self.messages_prior[v].lam)) / Aij

            self.messages[1-v].lam[0,0] = (1 - eta_damping) * Pij[0] + eta_damping * self.messages[1-v].lam[0,0]
            self.messages[1-v].lam[1,1] = (1 - eta_damping) * Pij[1] + eta_damping * self.messages[1-v].lam[1,1]
            self.messages[1-v].eta = (1 - eta_damping) * uij + eta_damping * self.messages[1-v].eta

            #self.adj_var_nodes[1-v].update_smooth_belief()

        #time.sleep(0.00000001)
