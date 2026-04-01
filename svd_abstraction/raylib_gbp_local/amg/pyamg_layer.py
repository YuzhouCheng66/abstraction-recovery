import numpy as np
from gbp import gbp
from gbp.factors import linear_displacement
import pyamg
import scipy.sparse as sp
from . import classes as amg_cls
from amg.solver_diagnostics import solver_diagnostics
import time

def base_A_mat(graph, A_type="full"):
        neighbours = [[] for _ in range(graph.n_var_nodes)]
        b = [[] for _ in range(graph.n_var_nodes)]
        data = []
        i = []
        j = []
        dof = 2

        b = [[] for _ in range(graph.n_var_nodes)]

        for vID in range(graph.n_var_nodes):
            b[vID] = graph.var_nodes[vID].prior.eta.copy()

            for factor in graph.var_nodes[vID].adj_factors:
                factor_idx = factor.adj_vIDs.index(vID)
                neighbours[vID].extend([x for x in factor.adj_vIDs if x != vID])
                data.append(-np.sum(np.diag(factor.factor.lam[:dof])))
                i.append(vID)
                j.append(neighbours[vID][-1])

                b[vID] += factor.factor.eta[factor_idx:factor_idx+dof]

            data.append(len(neighbours[vID]) * -data[-1] + np.sum(np.diag(graph.var_nodes[vID].prior.lam)))
            i.append(vID)
            j.append(vID)
        
        A = sp.coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(graph.n_var_nodes, graph.n_var_nodes))

        data = []
        neighbours = [[] for _ in range(graph.n_var_nodes)]
        i = []
        j = []

        b_full = np.zeros(graph.n_var_nodes * dof)

        for vID in range(graph.n_var_nodes):
            b_full[vID * dof] = graph.var_nodes[vID].prior.eta[0].copy()
            b_full[vID * dof + 1] = graph.var_nodes[vID].prior.eta[1].copy()

            for factor in graph.var_nodes[vID].adj_factors:
                factor_idx = factor.adj_vIDs.index(vID)
                neighbours[vID].extend([x for x in factor.adj_vIDs if x != vID])
                data.append(-factor.factor.lam[0,0])
                i.append(vID * dof)
                j.append(neighbours[vID][-1] * dof)
                data.append(-factor.factor.lam[0,0])
                i.append(vID * dof + 1)
                j.append(neighbours[vID][-1] * dof + 1)

                b_full[vID * dof] += factor.factor.eta[factor_idx*dof]
                b_full[vID * dof + 1] += factor.factor.eta[factor_idx*dof + 1]

            data.append(len(neighbours[vID]) * factor.factor.lam[0,0] + graph.var_nodes[vID].prior.lam[0,0])
            i.append(vID * dof)
            j.append(vID * dof)

            data.append(len(neighbours[vID]) * factor.factor.lam[0,0] + graph.var_nodes[vID].prior.lam[1,1])
            i.append(vID * dof + 1)
            j.append(vID * dof + 1)
        
        A_full = sp.coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(graph.n_var_nodes * dof, graph.n_var_nodes * dof))

        A = A.tocsr()
        A_full = A_full.tocsr()

        return A, A_full, b, b_full

def create_layers(A, b, graph, A_full=None, b_full=None):

    # Auto get best solver for problem matrix ------------------
    # solver_diagnostics(A_full, fname='posegraph_diagnostic', cycle_list=['V'])
    # ----------------------------------------------------------
    
    arg_sym = 'symmetric'  
    arg_strength = 'energy_based' # 'symmetric', 'classical', ''distance', 'evolution', 'energy_based', 'algebraic_distance', 'affinity'
    arg_agg = 'standard'  # 'standard', 'lloyd', 'naive'
    arg_smooth = None  # ('jacobi', {'omega': 4.0 / 3.0, 'degree': 2})
    arg_presmooth = ('block_gauss_seidel', {'sweep': 'symmetric'})
    arg_postsmooth = ('block_gauss_seidel', {'sweep': 'symmetric'})
    arg_improveCandidates = [('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None]
    arg_maxLevels = 5
    arg_accel = 'gmres'  # 'gmres', 'cg'

    ml = pyamg.smoothed_aggregation_solver(A ,symmetry=arg_sym, strength=arg_strength, aggregate=arg_agg, \
            max_levels=arg_maxLevels, smooth=arg_smooth, presmoother=arg_presmooth, postsmoother=arg_postsmooth, \
            improve_candidates=arg_improveCandidates)
    #ml = pyamg.ruge_stuben_solver(A)

    # PyAMG solver -------------------------------
    if b_full is not None:
        # Custom config --------------------------
        # ml_full = pyamg.smoothed_aggregation_solver(A_full, symmetry=arg_sym, strength=arg_strength, aggregate=arg_agg, \
        #     max_levels=arg_maxLevels, smooth=arg_smooth, presmoother=arg_presmooth, postsmoother=arg_postsmooth, \
        #     improve_candidates=arg_improveCandidates)

        # Solver diagnostics reccomendation- -----
        B = np.ones((A_full.shape[0],1), dtype=A_full.dtype); BH = B.copy()
        ml_full = pyamg.smoothed_aggregation_solver(A_full, B=B, BH=BH,
        strength=('evolution', {'k': 2, 'proj_type': 'l2', 'epsilon': 4.0}),
        smooth=('energy', {'krylov': 'cg', 'maxiter': 3, 'degree': 2, 'weighting': 'local'}),
        improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        aggregate="standard",
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 1}),
        max_levels=15,
        max_coarse=300,
        coarse_solver="pinv")
        
        # Solver phase ---------------------------
        res1 = []
        x = np.zeros_like(b_full)
        start = time.time_ns()
        for itr in range(20):
            x = ml_full.solve(b_full, x0=x, accel=arg_accel, tol=1e-20, maxiter=1, residuals=res1)
            end = time.time_ns()
            err = np.mean(np.linalg.norm(graph.mus_GT - np.reshape(x,(graph.n_var_nodes,2)), axis=1))
            print("PyAMG  //  {} iteration(s) in {:.3f}ms  //  avg. pixel error: {:.3f} pixels".format(itr+1,(end-start)/1e6, err))


    graph.layers[0].A = A
    graph.layers[0].var_ids = [*range(A.shape[0])]
    graph.layers[0].coarseIDs = [*range(A.shape[0])]
    graph.layers[0].coarse_baseIDs = [*range(A.shape[0])]
    graph.layers[0].factor_ids = [*range(len(graph.factors))]
    graph.layers[0].n_vars = A.shape[0]
    graph.multigrid = True

    for lvl_id in range(1, len(ml.levels)):
        if ml.levels[lvl_id].A.shape[0] > 1:
            layer = amg_cls.layer(lvl_id)

            P = ml.levels[lvl_id - 1].P.tocsr()
            R = ml.levels[lvl_id - 1].R.tocsr()

            layer.A = ml.levels[lvl_id].A.tocsr()
            layer.b = np.zeros(layer.A.shape[0])

            layer.n_vars = layer.A.shape[0]
            layer.coarseIDs = [*range(layer.n_vars)]
            layer.n_vars_layer_below = P.shape[0] 

            layer.interp_neighbours = [[] for _ in range(P.shape[0])]
            layer.interp_coeff = [[] for _ in range(P.shape[0])]
            
            for fID in range(P.shape[0]):
                _ , js = P[fID,:].nonzero()
                for j in js:
                    layer.interp_neighbours[fID].append(j)
                    layer.interp_coeff[fID].append(P[fID,j]) 

            layer.restrict_neighbours = [[] for _ in range(layer.n_vars)]
            layer.restrict_coeff = [[] for _ in range(layer.n_vars)]
            layer.coarse_baseIDs = [None] * layer.n_vars
                
            for cID in range(R.shape[0]):
                _ , js = R[cID,:].nonzero()
                for j in js:
                    layer.restrict_neighbours[cID].append(graph.layers[lvl_id-1].var_ids[j])
                    layer.restrict_coeff[cID].append(R[cID,j]) 
                
                closest_from_layer_below = layer.restrict_neighbours[cID][np.argmax(layer.restrict_coeff[cID])]
                layer_below_index = graph.layers[-1].var_ids.index(closest_from_layer_below)
                layer.coarse_baseIDs[cID] = graph.layers[-1].coarse_baseIDs[layer_below_index]

            layer.corrections = np.zeros((layer.n_vars_layer_below,2))
        
            n_vars = len(graph.var_nodes)
            n_c_vars = layer.n_vars

            layer.var_ids =[*range(n_vars, n_vars + n_c_vars)]

            for cID in range(n_c_vars):
                new_var_node = gbp.VariableNode(n_vars + cID, 2)
                new_var_node.prior.eta = np.array([0,0]).astype(float)
                new_var_node.prior.lam = np.array([[layer.A[cID, cID ]/2,0],[0, layer.A[cID, cID]/2]]).astype(float)
                #new_var_node.prior.lam = np.array([[np.sum(A[cID * 2, :]),0],[0, np.sum(A[cID * 2 + 1, :])]]).astype(float)
                graph.var_nodes.append(new_var_node)
            
            factors_start = len(graph.factors)
            
            for cID in range(n_c_vars):
                _ , js = layer.A[cID,:].nonzero()
                for j_id in js:
                    if j_id > cID:
                        new_factor = gbp.Factor(len(graph.factors),
                                        [graph.var_nodes[n_vars + cID], graph.var_nodes[n_vars + j_id]],
                                        [0., 0.],
                                        np.sqrt(1/(np.abs(layer.A[cID, j_id] / 2))),
                                        linear_displacement.meas_fn,
                                        linear_displacement.jac_fn,
                                        loss=None,
                                        mahalanobis_threshold=2)
                        
                        graph.var_nodes[n_vars + cID].adj_factors.append(new_factor)
                        graph.var_nodes[n_vars + j_id].adj_factors.append(new_factor)
                        graph.factors.append(new_factor)
            
            layer.factor_ids= [*range(factors_start, len(graph.factors))]

            graph.layers.append(layer)

            graph.update_all_beliefs(layer=graph.layers[-1].level)
            graph.compute_all_factors(layer=graph.layers[-1].level)
