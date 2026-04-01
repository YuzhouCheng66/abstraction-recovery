import numpy as np
# from sksparse.cholmod import cholesky
from utils.gaussian import NdimGaussian

class mutligrid_var_info:
    def __init__(self, var_node):
        self.var_node = var_node
        self.level = 0
        self.theta = 0.25
        self.interp_mode = "extended_if_needed"
        self.prolongation_eta_mode = "none"
        self.prolongation_eta_eps = 1e-12
        self.prolongation_eta_levels = None
        self.neighbour_vars = []
        self.neighbour_class = []
        self.lam_incoming = []
        self.res_incoming = []
        self.corrections_outgoing = []

        self.classification = "unassigned"  # "coarse", "fine"
        self.parent = None
        self.child = None

        self.interpolation_vars = [] # vars on the level below which will get the correction
        self.interpolation_coefficients = []

        self.restriction_vars = [] # contribution of self to the var on the level above
        self.restriction_coefficients = []

        self.deactivate_threshold = -1e-3 # pixel

    def _build_interpolation_stencil(self, direct_coarse_vars, strong_vars):
        coarse_vars = direct_coarse_vars.copy()
        if self.interp_mode == "direct":
            return coarse_vars

        coarse_var_set = set(coarse_vars)
        for strong_var in strong_vars:
            strong_coarse_vars = [
                var for var in strong_var.multigrid.neighbour_vars
                if var.multigrid.classification == "coarse"
            ]
            if self.interp_mode == "extended_if_needed":
                if coarse_var_set.intersection(strong_coarse_vars):
                    continue

            for coarse_var in strong_coarse_vars:
                if coarse_var not in coarse_var_set:
                    coarse_vars.append(coarse_var)
                    coarse_var_set.add(coarse_var)

        return coarse_vars

    def update_neighbour_vars(self):
        self.neighbour_vars = []
        self.neighbour_class = []
        for factor in self.var_node.adj_factors:
            for var in factor.adj_var_nodes:
                if var != self.var_node:
                    self.neighbour_vars.append(var)
                    self.neighbour_class.append("unknown")

    def categorise_neighbours(self):
        self.lam_incoming = []
        self.neighbour_class = []
        self.neighbour_vars = []

        self.update_neighbour_vars()

        if self.neighbour_vars:
            for factor in self.var_node.adj_factors:
                dofs = self.var_node.dofs
                self.lam_incoming.append(factor.factor.lam[0:dofs,dofs:dofs+dofs])  # off diagonal lam

            max_lam = np.zeros_like(self.lam_incoming[0])

            for lam in self.lam_incoming:
                if (-lam >= max_lam).all():
                    max_lam = -lam

            for idx, lam in enumerate(self.lam_incoming):
                if self.neighbour_vars[idx].multigrid.classification != "coarse":
                    if (-lam >= max_lam * self.theta).all():
                        self.neighbour_class[idx] = "strong"
                    else:
                        self.neighbour_class[idx] = "weak"

                else:
                    self.neighbour_class[idx] = "coarse"

    def unassign(self):
        # remove restriction/intepolation connection to high level var
        if self.classification == "fine":
            for r_var in self.restriction_vars:
                for i_idx ,i_var in enumerate(r_var.multigrid.interpolation_vars):
                    if i_var == self.var_node:
                        r_var.multigrid.interpolation_vars.pop(i_idx)
                        r_var.multigrid.interpolation_coefficients.pop(i_idx)
                        r_var.multigrid.res_incoming.pop(i_idx)
                        r_var.multigrid.corrections_outgoing.pop(i_idx)
        # If coarse we need to remove all connections in the higher level var
        # The actual high level var needs to be removed in the graph list outside of here
        elif self.classification == "coarse":
            for i_var in self.parent.multigrid.interpolation_vars:
                for r_idx, r_var in enumerate(i_var.multigrid.restriction_vars):
                    if r_var == self.parent:
                        i_var.multigrid.restriction_vars.pop(r_idx)
                        i_var.multigrid.restriction_coefficients.pop(r_idx)

            if self.parent:
                self.parent.type = "dead"
                self.parent.multigrid.unassign()
                
            self.parent.multigrid.child = None
            self.parent.multigrid.interpolation_vars = [] # vars on the level below which will get the correction
            self.parent.multigrid.interpolation_coefficients = []
            self.parent.multigrid.res_incoming = []
            self.parent.multigrid.corrections_outgoing = []
            self.parent = None
            
        if self.var_node.type == "dead":
            for factor in self.var_node.adj_factors:
                for var in factor.adj_var_nodes:
                    if var != self.var_node:
                        var.adj_factors.remove(factor)
                factor.adj_var_nodes = []
                factor.type = "dead"
            self.var_node.adj_factors = []
            self.res_incoming = []
            self.corrections_outgoing = []

        self.neighbour_vars = []
        self.neighbour_class = []
        self.lam_incoming = []
        self.classification = "unassigned"  # "coarse", "fine"
        self.restriction_vars = [] # contribution of self to the var on the level above
        self.restriction_coefficients = []


    def update_restriction_interpolation(self):
        if self.classification == "fine":
            self.restriction_vars = [] # contribution of self to the var on the level above
            self.restriction_coefficients = []
            direct_coarse_vars = []
            strong_vars = []
            weak_vars = []
            lam_ii = self.var_node.prior.lam.copy()
            neighbour_idx = {var: idx for idx, var in enumerate(self.neighbour_vars)}
            for idx in range(len(self.neighbour_vars)):
                if self.neighbour_class[idx] == "coarse":
                    direct_coarse_vars.append(self.neighbour_vars[idx])
                elif self.neighbour_class[idx] == "strong":
                    strong_vars.append(self.neighbour_vars[idx])
                elif self.neighbour_class[idx] == "weak":
                    weak_vars.append(self.neighbour_vars[idx])

                lam_ii -=  self.lam_incoming[idx]

            coarse_vars = self._build_interpolation_stencil(direct_coarse_vars, strong_vars)

            weak_sum = np.zeros_like(lam_ii)
            for n in weak_vars:
                weak_sum += self.lam_incoming[neighbour_idx[n]]

            for j in coarse_vars:
                if j in neighbour_idx:
                    lam_ij = self.lam_incoming[neighbour_idx[j]]
                else:
                    lam_ij = np.zeros_like(lam_ii)

                strong_sum = np.zeros_like(lam_ij)

                for m in strong_vars:
                    if j in m.multigrid.neighbour_vars:
                        lam_im = self.lam_incoming[neighbour_idx[m]]

                        mj_idx = m.multigrid.neighbour_vars.index(j)
                        lam_mj = m.multigrid.lam_incoming[mj_idx]

                        mk_sum = np.zeros_like(lam_mj)

                        for k in coarse_vars:
                            if k in m.multigrid.neighbour_vars:
                                mk_idx = m.multigrid.neighbour_vars.index(k)
                                mk_sum += m.multigrid.lam_incoming[mk_idx]

                        strong_sum += (lam_im * lam_mj) @ np.linalg.inv(mk_sum)

                wij = (lam_ij + strong_sum) @ np.linalg.inv(lam_ii + weak_sum)

                self.restriction_vars.append(j.multigrid.parent)
                self.restriction_coefficients.append(-wij)

                try:
                    i_idx = j.multigrid.parent.multigrid.interpolation_vars.index(self.var_node)
                    j.multigrid.parent.multigrid.interpolation_coefficients[i_idx] = -wij
                    j.multigrid.parent.multigrid.res_incoming[i_idx] = 0
                    j.multigrid.parent.multigrid.corrections_outgoing[i_idx] = 0
                except:
                    j.multigrid.parent.multigrid.interpolation_vars.append(self.var_node)
                    j.multigrid.parent.multigrid.interpolation_coefficients.append(-wij)
                    j.multigrid.parent.multigrid.res_incoming.append(0)
                    j.multigrid.parent.multigrid.corrections_outgoing.append(0)

            if np.sum(self.restriction_coefficients) > self.var_node.dofs:
                print('err')


    def send_restricted_residual(self):
        max_res = 0
        for r_idx, r_var in enumerate(self.restriction_vars):
            i_idx = r_var.multigrid.interpolation_vars.index(self.var_node)
            r_var.multigrid.res_incoming[i_idx] = self.restriction_coefficients[r_idx] @ self.var_node.residual
            max_res = np.max([max_res,np.linalg.norm(r_var.multigrid.res_incoming[i_idx])])

        # if max_res < self.deactivate_threshold:
        #     self.var_node.active = False
        #     for factor in self.var_node.adj_factors:
        #         factor.active = False

    def update_eta(self):
        incoming_residuals = np.sum(self.res_incoming, axis=0)
        self.var_node.prior.eta = incoming_residuals
        # ---------------
        # If the incoming res from all vars is small enough, we can
        # deactive that group of vars and their factors. If the incoming res is large
        # then it should be smoothed next time around
        #----------------
        
        if all(np.linalg.norm(self.res_incoming,axis=1) < self.deactivate_threshold):
            for i_var in self.interpolation_vars:
                i_var.active = False
                for factor in i_var.adj_factors:
                    factor.active = False
        elif any(np.linalg.norm(self.res_incoming,axis=1) > self.deactivate_threshold*100):
            for i_var in self.interpolation_vars:
                i_var.active = True
                for factor in i_var.adj_factors:
                    factor.active = True

        # ---------------
        # If the incoming res from a single var is small enough, we can
        # deactive the var and its factors. If the incoming res is large
        # then it should be smoothed next time around
        #----------------
        # for i_idx, i_var in enumerate(self.interpolation_vars):
        #     if i_var.active:
        #         if np.linalg.norm(self.res_incoming[i_idx]) < self.deactivate_threshold:
        #             i_var.active = False
        #             for factor in i_var.adj_factors:
        #                 factor.active = False
        #     else:
        #         if np.linalg.norm(self.res_incoming[i_idx]) > self.deactivate_threshold * 1000:
        #             for factor in i_var.adj_factors:
        #                 factor.active = True
        #                 for var in factor.adj_var_nodes:
        #                     var.active = True
                    


    def send_corrections(self):
        affected_vars = []
        for i_idx, i_var in enumerate(self.interpolation_vars):
            self.corrections_outgoing[i_idx] = self.interpolation_coefficients[i_idx] @ self.var_node.mu
            i_var.mu +=  self.corrections_outgoing[i_idx]
            target_eta = i_var.belief.lam @ i_var.mu
            self._update_eta_side_after_correction(i_var, target_eta)
            affected_vars.append(i_var)

        # outgoing_corrections = np.sum(self.corrections_outgoing, axis=0)
        # if np.linalg.norm(outgoing_corrections) > self.deactivate_threshold * 100:
        #     for i_idx, i_var in enumerate(self.interpolation_vars):
        #         for factor in i_var.adj_factors:
        #             factor.active = True
        #             for var in factor.adj_var_nodes:
        #                 var.active = True

            # Update residual of inactive vars after correction to make sure this is accounted for in activation thresholding
            # if np.linalg.norm(self.corrections_outgoing[i_idx]) > self.deactivate_threshold*100:
            #     for factor in i_var.adj_factors:
            #         factor.active = True
            #         for var in factor.adj_var_nodes:
            #             var.active = True
        return affected_vars

    def _update_eta_side_after_correction(self, i_var, target_eta):
        mode = self._effective_prolongation_eta_mode()
        if mode in {"none", "local_direct", "local_iter5", "cached_iter1"}:
            i_var.belief.eta = target_eta
            for factor in i_var.adj_factors:
                belief_ix = factor.adj_vIDs.index(i_var.variableID)
                factor.adj_beliefs[belief_ix].eta = i_var.belief.eta
            return

        incoming_target = target_eta - i_var.prior.eta
        current_incoming = np.zeros_like(target_eta)
        message_refs = []
        weights = []
        eps = getattr(self, "prolongation_eta_eps", 1e-12)

        for factor in i_var.adj_factors:
            belief_ix = factor.adj_vIDs.index(i_var.variableID)
            msg = factor.messages[belief_ix]
            current_incoming += msg.eta
            message_refs.append((factor, belief_ix, msg))
            weights.append(self._eta_redistribution_weight(i_var, factor, belief_ix, mode, eps))

        if message_refs:
            weights = np.asarray(weights, dtype=float)
            weights_sum = float(np.sum(weights))
            if weights_sum <= 0:
                weights = np.full(len(message_refs), 1.0 / len(message_refs))
            else:
                weights = weights / weights_sum

            delta_incoming = incoming_target - current_incoming
            for weight, (_, _, msg) in zip(weights, message_refs):
                msg.eta = msg.eta + weight * delta_incoming

        rebuilt_eta = i_var.prior.eta.copy()
        for factor in i_var.adj_factors:
            belief_ix = factor.adj_vIDs.index(i_var.variableID)
            rebuilt_eta += factor.messages[belief_ix].eta

        i_var.belief.eta = rebuilt_eta
        for factor in i_var.adj_factors:
            belief_ix = factor.adj_vIDs.index(i_var.variableID)
            factor.adj_beliefs[belief_ix].eta = i_var.belief.eta

    def _eta_redistribution_weight(self, i_var, factor, belief_ix, mode, eps):
        if mode == "uniform":
            return 1.0

        if mode == "message_lam_trace":
            lam = factor.messages[belief_ix].lam
            return max(float(np.sum(np.abs(np.diagonal(lam)))), eps)

        if mode == "factor_diag_trace":
            dofs = i_var.dofs
            var_idx = factor.adj_vIDs.index(i_var.variableID)
            start = var_idx * dofs
            stop = start + dofs
            lam_block = factor.factor.lam[start:stop, start:stop]
            return max(float(np.sum(np.abs(np.diagonal(lam_block)))), eps)

        raise ValueError(f"Unknown prolongation eta mode: {mode}")

    def _effective_prolongation_eta_mode(self):
        mode = getattr(self, "prolongation_eta_mode", "none")
        levels = getattr(self, "prolongation_eta_levels", None)
        if levels is None:
            return mode
        if self.var_node.multigrid.level in levels:
            return mode
        return "none"
    



class layer:
    def __init__(self, level):
        self.var_ids = []
        self.factor_ids = []
        self.level = level
        self.A = []
        self.b = []
        self.coarseIDs = []
        self.fineIDs = []
        self.interp_neighbours = []
        self.interp_coeff = []
        self.restrict_neighbours = []
        self.restrict_coeff = []
        self.corrections = []
        self.n_vars = 0
        self.n_vars_layer_below = 0

        #  TODO: Add the variable and factor of the graph to the class var so we don't have to keep finding where it is store in the graph

    def restrict(self, graph):
        self.residuals = [0] * self.n_vars
        for vID in range(self.n_vars):
            restricted_residual = np.zeros(graph.var_nodes[self.var_ids[vID]].dofs)
            for jID, neighbour in enumerate(self.restrict_neighbours[vID]):
                restricted_residual += graph.var_nodes[neighbour].residual * self.restrict_coeff[vID][jID]

            self.residuals[vID] = restricted_residual
            graph.var_nodes[self.var_ids[vID]].prior.eta = restricted_residual
            graph.var_nodes[self.var_ids[vID]].Sigma = 1/np.diagonal(graph.var_nodes[self.var_ids[vID]].belief.lam)
            graph.var_nodes[self.var_ids[vID]].mu = graph.var_nodes[self.var_ids[vID]].Sigma * graph.var_nodes[self.var_ids[vID]].belief.eta

        self.b = np.ravel(self.residuals)
        # factor = cholesky(self.A.tocsc())
        # self.correction = factor(self.residuals)

        for fid in self.factor_ids:
            for v in range(0,2):
                graph.factors[fid].messages[v] = NdimGaussian(2)
               

    def prolongate(self, graph):
        for vID in range(self.n_vars_layer_below):
            prolongated_error = 0
            for jID, fineID in enumerate(self.interp_neighbours[vID]):
                coarseID = self.coarseIDs.index(fineID)
                graphID = self.var_ids[0] + coarseID
                mu = graph.var_nodes[graphID].mu
                #mu = self.correction[jID]
                #print(mu_exact - mu)
                prolongated_error += mu * self.interp_coeff[vID][jID]

            self.corrections[vID] = prolongated_error
            graph.var_nodes[vID].mu += prolongated_error
            graph.var_nodes[vID].belief.eta = graph.var_nodes[vID].mu * np.diagonal(graph.var_nodes[vID].belief.lam)
            # Send belief to adjacent factors
            for factor in graph.var_nodes[vID].adj_factors:
                belief_ix = factor.adj_vIDs.index(vID)
                factor.adj_beliefs[belief_ix].eta = graph.var_nodes[vID].belief.eta
