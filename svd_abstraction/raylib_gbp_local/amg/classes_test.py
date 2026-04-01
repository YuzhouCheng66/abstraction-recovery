import numpy as np
from sksparse.cholmod import cholesky
from scipy import sparse
from utils.gaussian import NdimGaussian

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
                restricted_residual[0] += graph.var_nodes[neighbour].residual[0] * self.restrict_coeff[vID][jID]
                restricted_residual[1] += graph.var_nodes[neighbour].residual[1] * self.restrict_coeff[vID][jID]

            self.residuals[vID] = restricted_residual
            graph.var_nodes[self.var_ids[vID]].prior.eta = restricted_residual
            graph.var_nodes[self.var_ids[vID]].Sigma = 1/np.diagonal(graph.var_nodes[self.var_ids[vID]].belief.lam)
            graph.var_nodes[self.var_ids[vID]].mu = graph.var_nodes[self.var_ids[vID]].Sigma * graph.var_nodes[self.var_ids[vID]].belief.eta
        
        self.b = np.ravel(self.residuals)
        # factor = cholesky(sparse.csc_matrix(self.A))
        # self.correction = factor(self.b)

        for fid in self.factor_ids:
            for v in range(0,2):
                graph.factors[fid].messages[v] = NdimGaussian(2)
        #     res0_id = self.var_ids.index(graph.factors[fid].adj_vIDs[0])
        #     res1_id = self.var_ids.index(graph.factors[fid].adj_vIDs[1])
            
        #     lam_ii_ij = graph.factors[fid].factor.lam[0:2,0:2]
        #     etai_ij = self.residuals[res0_id]
        #     lamij_ij = graph.factors[fid].factor.lam[0:2,2:4]
        #     lamji_ij = graph.factors[fid].factor.lam[2:4,0:2]
        #     lamjj_ij = graph.factors[fid].factor.lam[2:4,2:4]
        #     lambj = graph.factors[fid].adj_beliefs[0].lam
        #     etaj_ij = self.residuals[res1_id]
        #     eta_bj = self.residuals[res1_id]

        #     eta_ij = etai_ij - lamij_ij@np.linalg.inv(lamjj_ij+lambj)@(etaj_ij+eta_bj)
        #    lam_ij = lam_ii_ij - lamij_ij@np.linalg.inv(lamjj_ij+lambj)@lamji_ij


    def prolongate(self, graph):
        for vID in range(self.n_vars_layer_below):
            prolongated_error = np.array([0.,0.])
            for jID, fineID in enumerate(self.interp_neighbours[vID]):
                coarseID = self.coarseIDs.index(fineID)
                graphID = self.var_ids[0] + coarseID
                mu = graph.var_nodes[graphID].mu
                #mu_exact = self.correction[coarseID*2:coarseID*2+2]
                #print(mu_exact - mu)
                prolongated_error[0] += mu[0] * self.interp_coeff[vID][jID]
                prolongated_error[1] += mu[1] * self.interp_coeff[vID][jID]

            self.corrections[vID] = prolongated_error
            graph.var_nodes[vID].mu += prolongated_error
            graph.var_nodes[vID].belief.eta = graph.var_nodes[vID].mu * np.diagonal(graph.var_nodes[vID].belief.lam)
            # Send belief to adjacent factors
            for factor in graph.var_nodes[vID].adj_factors:
                belief_ix = factor.adj_vIDs.index(vID)
                factor.adj_beliefs[belief_ix].eta = graph.var_nodes[vID].belief.eta
