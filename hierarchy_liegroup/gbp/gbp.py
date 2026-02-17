"""
    Defines classes for variable nodes, factor nodes and edges and factor graph.
"""

import manifpy as m
import numpy as np
import time
import scipy.linalg

class Message:
    def __init__(self, status=None, Lam=None):
        if status: 
            self.status = status 
        else: 
            self.status = m.SE2(0,0,0)
        
        if (Lam is not None): 
            self.Lam = Lam
        else: 
            self.Lam = np.eye(3)*1e-10



class VariableNode:
    def __init__(self,
                 variable_id,
                 dofs=3):

        self.dofs = dofs
        self.variableID = variable_id
        self.adj_factors = []
        
        self.to_factor_messages = []

        self.status = m.SE2(0,0,0)
        self.GT = None
        self.Lam = np.eye(dofs)*1e-10
        
    def update_belief(self):
        # Calculate the product of all incoming messages to get the belief. 
        eta_all = np.zeros(self.dofs)
        Lam_all = np.eye(self.dofs)*1e-10

        for factor in self.adj_factors:
            message_ix = factor.adj_var_nodes.index(self)
            x_msg, Lam_msg = factor.to_variable_messages[message_ix].status, factor.to_variable_messages[message_ix].Lam

            tau_msg = x_msg - self.status
            L_msg = tau_msg.rjac().T @ Lam_msg @ tau_msg.rjac()

            eta_all += L_msg @ tau_msg.coeffs()
            Lam_all += L_msg

        # Calculate the outcoming messages to each adjacent factor.
        for factor in self.adj_factors:
            if len(factor.adj_var_nodes) <= 1:
                continue
            message_ix = factor.adj_var_nodes.index(self)
            x_msg, Lam_msg = factor.to_variable_messages[message_ix].status, factor.to_variable_messages[message_ix].Lam

            tau_msg = x_msg - self.status
            L_msg = tau_msg.rjac().T @ Lam_msg @ tau_msg.rjac()
            eta_msg = L_msg @ tau_msg.coeffs()

            eta_msg_a = eta_all - eta_msg
            Lam_msg_a = Lam_all - L_msg

            tau_msg_a = m.SE2Tangent(np.linalg.solve(Lam_msg_a, eta_msg_a))
            rjacinv = np.linalg.inv(tau_msg_a.rjac())
            self.to_factor_messages[self.adj_factors.index(factor)].status = self.status + tau_msg_a
            self.to_factor_messages[self.adj_factors.index(factor)].Lam = rjacinv.T @ Lam_msg_a @ rjacinv

        tau_all = m.SE2Tangent(np.linalg.solve(Lam_all, eta_all))
        self.status = self.status + tau_all 
        rjacinv = np.linalg.inv(tau_all.rjac())
        self.Lam = rjacinv.T @ Lam_all @ rjacinv


class Factor:
    def __init__(self,
                factor_id,
                adj_var_nodes,
                measurement,
                measurement_lambda,
                robustify=False):
            self.factorID = factor_id
            self.adj_var_nodes = adj_var_nodes
            self.iters_since_relin = 0
            self.threshold = 1e8
    
            self.linpoints = []
            self.measurement = measurement
            self.measurement_lambda = measurement_lambda
            self.factor_eta = None
            self.factor_Lam = None
            self.robustify = robustify

            self.last_messages_eta = [np.zeros(3) for _ in self.adj_var_nodes]
            self.last_messages_Lam = [np.zeros([3, 3]) for _ in self.adj_var_nodes]
            self.to_variable_messages = [Message(status=adj_var_node.status) for adj_var_node in self.adj_var_nodes]


    def update_factor(self):
        return


    def robust_kernel(self, error):
        if error > self.threshold:
            return self.threshold / error
        return 1

    def set_factor_eta_Lam(self, h, J):
        if self.robustify:
            error = h.T @ self.measurement_lambda @ h
            scale = self.robust_kernel(error)
            self.factor_eta = scale * J.T @ self.measurement_lambda @ (-h)
            self.factor_Lam = scale * J.T @ self.measurement_lambda @ J + np.eye(J.shape[1])*1e-12  # Add small value to ensure positive definiteness
        else:
            self.factor_eta = J.T @ self.measurement_lambda @ (-h)
            self.factor_Lam = J.T @ self.measurement_lambda @ J + np.eye(J.shape[1])*1e-12  # Add small value to ensure positive definiteness

    def update_all(self):
        eta_c = self.factor_eta.copy()
        Lam_c = self.factor_Lam.copy()
        dof = 3
        for i, adj_var_node in enumerate(self.adj_var_nodes):
            message_ix = adj_var_node.adj_factors.index(self)
            message = adj_var_node.to_factor_messages[message_ix]

            tau = message.status - self.linpoints[i]
            Lam = tau.rjac().T @ message.Lam @ tau.rjac()
            eta = Lam @ tau.coeffs()

            eta_c[i*dof:(i+1)*dof] += eta
            Lam_c[i*dof:(i+1)*dof, i*dof:(i+1)*dof] += Lam
            
        return eta_c, Lam_c

    def downdate_one(self, eta_c, Lam_c, adj_var_node):
        dof = 3
        i = self.adj_var_nodes.index(adj_var_node)
        message_ix = adj_var_node.adj_factors.index(self)
        message = adj_var_node.to_factor_messages[message_ix]

        tau = message.status - self.linpoints[i]
        Lam = tau.rjac().T @ message.Lam @ tau.rjac()
        eta = Lam @ tau.coeffs()

        eta_c[i*dof:(i+1)*dof] -= eta
        Lam_c[i*dof:(i+1)*dof, i*dof:(i+1)*dof] -= Lam

        return eta_c, Lam_c

    def compute_messages(self, eta_damping=0.0):
        if len(self.adj_var_nodes) == 1:
            new_message_eta = self.factor_eta.copy()
            new_message_lam = self.factor_Lam.copy()
            tau = m.SE2Tangent(np.linalg.solve(new_message_lam, new_message_eta))
            tau_rjacinv = np.linalg.inv(tau.rjac())
            self.to_variable_messages[self.adj_var_nodes.index(self.adj_var_nodes[0])].status = self.linpoints[0] + tau
            self.to_variable_messages[self.adj_var_nodes.index(self.adj_var_nodes[0])].Lam = tau_rjacinv.T@new_message_lam@tau_rjacinv
            return
        
        eta_updated, Lam_updated = self.update_all()
        divide =  self.adj_var_nodes[0].dofs
        for v, adj_var_node in enumerate(self.adj_var_nodes):
            if v == 0:
                eta_d, Lam_d = self.downdate_one(eta_updated.copy(), Lam_updated.copy(), adj_var_node)
                eo = eta_d[:divide]
                eno = eta_d[divide:]

                loo = Lam_d[:divide, :divide]
                lono = Lam_d[:divide, divide:]
                lnoo = Lam_d[divide:, :divide]
                lnono = Lam_d[divide:, divide:]
            elif v == 1:
                eta_d, Lam_d = self.downdate_one(eta_updated.copy(), Lam_updated.copy(), adj_var_node)
                eo = eta_d[divide:]
                eno = eta_d[:divide]

                loo = Lam_d[divide:, divide:]
                lono = Lam_d[divide:, :divide]
                lnoo = Lam_d[:divide, divide:]
                lnono = Lam_d[:divide, :divide]

            # concat RHS
            rhs_j = np.concatenate([lnoo, eno.reshape(-1, 1)], axis=1)   # (n, n+1)
            X = np.linalg.solve(lnono, rhs_j)
            X_lam = X[:, :lnoo.shape[1]]
            X_eta = X[:, -1]

            new_message_lam = (1-eta_damping)*(loo - lono @ X_lam) + eta_damping * self.last_messages_Lam[v]
            new_message_eta = (1-eta_damping)*(eo  - lono @ X_eta) + eta_damping * self.last_messages_eta[v]
            self.last_messages_Lam[v] = new_message_lam
            self.last_messages_eta[v] = new_message_eta

            try:
                tau = m.SE2Tangent(np.linalg.solve(new_message_lam, new_message_eta))
            except np.linalg.LinAlgError:
                print("Warning: Singular matrix encountered in message computation. Using zero tangent vector.")
                tau = m.SE2Tangent(np.array([0,0,0]))
            self.to_variable_messages[self.adj_var_nodes.index(adj_var_node)].status = self.linpoints[v] + tau
            tau_rjacinv = np.linalg.inv(tau.rjac())
            self.to_variable_messages[self.adj_var_nodes.index(adj_var_node)].Lam = tau_rjacinv.T@new_message_lam@tau_rjacinv



class priorSE2(Factor):
    def __init__(self, factor_id, adj_var_nodes, measurement, measurement_lambda, robustify):
        super().__init__(factor_id, adj_var_nodes, measurement, measurement_lambda, robustify)


    def residual_prior(self, X, p):
        J = np.zeros((3, 3))
        ret = p.minus(X, None, J).coeffs()

        return ret, J

    def update_factor(self):
        h, J = self.residual_prior(self.linpoints[0], self.measurement)
        self.set_factor_eta_Lam(h, J)
        return True
    
    def loss(self):
        h, J = self.residual_prior(self.adj_var_nodes[0].status, self.measurement)
        return h.transpose() @ self.measurement_lambda @ h


class odometrySE2(Factor):
    def __init__(self, factor_id, adj_var_nodes, measurement, measurement_lambda, robustify):
        super().__init__(factor_id, adj_var_nodes, measurement, measurement_lambda, robustify)

    def between(self, X1, X2):
        J1_X2_X1 = np.zeros((3, 3))  # d( X1.between(X2) ) / dX1
        J2_X2_X1 = np.zeros((3, 3))  # d( X1.between(X2) ) / dX2
        ret = X1.between(X2, J1_X2_X1, J2_X2_X1)

        J = np.zeros((3, 6))
        J[:, 0:3] = J1_X2_X1
        J[:, 3:6] = J2_X2_X1

        return ret, J

    def residual_between(self, X1, X2, meas):
        bt, J = self.between(X1, X2)
        J_r = np.zeros((3, 3))     # d( meas.rminus(bt) ) / d(bt)
        ret = meas.rminus(bt, None, J_r).coeffs()
        J = J_r @ J

        return ret, J

    def update_factor(self):
        h, J = self.residual_between(self.linpoints[0], self.linpoints[1], self.measurement)
        self.set_factor_eta_Lam(h, J)
        return True
    
    def loss(self):
        h, J = self.residual_between(self.adj_var_nodes[0].status, self.adj_var_nodes[1].status, self.measurement)
        return h.transpose() @ self.measurement_lambda @ h


class FactorGraph:
    def __init__(self,
                 nonlinear_factors=True,
                 eta_damping=0,
                 beta=0.0,
                 num_undamped_iters=0,
                 min_linear_iters=10):

        self.var_nodes = []
        self.factors = []

        self.n_var_nodes = 0
        self.n_factor_nodes = 0
        self.nonlinear_factors = nonlinear_factors
        self.eta_damping = eta_damping

        if nonlinear_factors:
            # For linearising nonlinear measurement factors.
            self.beta = beta  # Threshold change in mean of adjacent beliefs for relinearisation.
            self.num_undamped_iters = num_undamped_iters  # Number of undamped iterations after relinearisation before damping is set to 0.4
            self.min_linear_iters = min_linear_iters  # Minimum number of linear iterations before a factor is allowed to realinearise.


    def compute_all_messages(self, factors=None):
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]

        for factor in factors:
            if factor.iters_since_relin >= self.num_undamped_iters:
                factor.compute_messages(self.eta_damping)
            else:
                factor.compute_messages(eta_damping=0.0)


    def update_all_beliefs(self, vars=None):
        if vars is None:
            vars = self.var_nodes[:self.n_var_nodes]

        for var in vars:
            var.update_belief()


    def relinearise_factors(self, factors=None):
        """
        On a nonlinear graph, determine whether to relinearize based on the 
        deviation between the mean of adjacent beliefs and the current linearized point.

        If factors are passed, relinearize only for that subset; 
        otherwise, relinearize for all self.factors.
        """
        if not self.nonlinear_factors:
            return

        # 允许从 synchronous_iteration 传入子集
        if factors is None:
            factors = self.factors[:self.n_factor_nodes]

        # 给 beta / min_linear_iters 容错默认
        beta = self.beta if self.beta is not None else 0.0              # 0 表示每次都允许重线性化
        min_linear_iters = self.min_linear_iters if self.min_linear_iters is not None else 10

        for factor in factors:
            # 是否满足重线性化条件
            if factor.iters_since_relin >= min_linear_iters:
                # 用新的线性化点重建因子
                factor.linpoints = [adj_var_node.status for adj_var_node in factor.adj_var_nodes]
                factor.update_factor()
                # 重线性化后的计数
                factor.iters_since_relin = 1

            else:
                factor.iters_since_relin += 1


    def robustify_all_factors(self):
        for factor in self.factors[:self.n_factor_nodes]:
            factor.robustify_loss()


    def synchronous_iteration(self, factors=None, robustify=False):
        vars = self.var_nodes[:self.n_var_nodes]
        factors = self.factors[:self.n_factor_nodes]

        if robustify:
            self.robustify_all_factors(factors)
        if self.nonlinear_factors:
            self.relinearise_factors(factors)

        self.compute_all_messages(factors)
        self.update_all_beliefs(vars)
    

    def joint_distribution_inf(self):
        """
            Get the joint distribution over all variables in the information form
            If nonlinear factors, it is taken at the current linearisation point.
        """

        var_ix = np.zeros(len(self.var_nodes)).astype(int)
        tot_n_vars = 0
        for var_node in self.var_nodes:
            var_ix[var_node.variableID] = int(tot_n_vars)
            tot_n_vars += var_node.dofs

        eta = np.zeros(tot_n_vars)
        lam = np.zeros([tot_n_vars, tot_n_vars])

        for count, factor in enumerate(self.factors):
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                vID = adj_var_node.variableID
                # Diagonal contribution of factor
                eta[var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                    factor.factor_eta[factor_ix:factor_ix + adj_var_node.dofs]
                lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                    factor.factor_Lam[factor_ix:factor_ix + adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if other_adj_var_node.variableID > adj_var_node.variableID:
                        other_vID = other_adj_var_node.variableID
                        # Off diagonal contributions of factor
                        lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs] += \
                            factor.factor_Lam[factor_ix:factor_ix + adj_var_node.dofs, other_factor_ix:other_factor_ix + other_adj_var_node.dofs]
                        lam[var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                            factor.factor_Lam[other_factor_ix:other_factor_ix + other_adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
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


    def loss(self):
        l = 0
        for factor in self.factors[:self.n_factor_nodes]:
            l += factor.loss()
        return 0.5*l
    
    def SE(self):
        sqaured_error = 0
        for var_node in self.var_nodes[:self.n_var_nodes]:
            sqaured_error += (var_node.status-var_node.GT).squaredWeightedNorm()
        return sqaured_error