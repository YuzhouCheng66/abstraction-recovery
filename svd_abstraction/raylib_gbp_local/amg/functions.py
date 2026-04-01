from amg import classes as amg_cls
from gbp import gbp
import numpy as np
from scipy.sparse import coo_matrix
# from gbp.factors import linear_displacement
# from gbp.gbp import VariableNode
# from gbp.gbp import Factor

def _promote_to_coarse(graph, var, coarse_vars):
    if var.multigrid.classification == "coarse" and var.multigrid.parent is not None:
        return var.multigrid.parent

    var.multigrid.classification = "coarse"

    coarse_var_node = gbp.VariableNode(graph.n_var_nodes, 2)
    coarse_var_node.type = "multigrid"
    coarse_var_node.multigrid.level = var.multigrid.level + 1
    coarse_var_node.multigrid.theta = var.multigrid.theta
    coarse_var_node.multigrid.interp_mode = var.multigrid.interp_mode
    coarse_var_node.multigrid.interpolation_vars.append(var)
    coarse_var_node.multigrid.interpolation_coefficients.append(np.diag([1., 1.]))
    coarse_var_node.multigrid.res_incoming.append(0)
    coarse_var_node.multigrid.corrections_outgoing.append(0)

    var.multigrid.parent = coarse_var_node
    coarse_var_node.multigrid.child = var

    graph.var_nodes.append(coarse_var_node)
    try:
        graph.multigrid_vars[coarse_var_node.multigrid.level].append(coarse_var_node)
    except:  # Catch for if the level hasn't been created yet
        graph.multigrid_vars.append([])
        graph.multigrid_factors.append([])
        graph.multigrid_vars[coarse_var_node.multigrid.level].append(coarse_var_node)
    coarse_vars.append(coarse_var_node)
    graph.n_var_nodes += 1

    var.multigrid.restriction_vars = [graph.var_nodes[-1]]
    var.multigrid.restriction_coefficients = [np.diag([1., 1.])]

    return coarse_var_node


def _split_coarse_fine_rs(graph, vars_update, coarse_vars, fine_vars, num_strong_connections):
    while max(num_strong_connections) > 0:
        max_id = np.argmax(num_strong_connections)
        num_strong_connections[max_id] = -1
        i_var = vars_update[max_id]

        _promote_to_coarse(graph, i_var, coarse_vars)

        for j_var in i_var.multigrid.neighbour_vars:
            j_var.multigrid.categorise_neighbours()
            if j_var.multigrid.classification == "unassigned":
                j_var.multigrid.classification = "fine"
                fine_vars.append(j_var)
                num_strong_connections[vars_update.index(j_var)] = -1
                for k_var in j_var.multigrid.neighbour_vars:
                    if k_var.multigrid.classification == "unassigned":
                        num_strong_connections[vars_update.index(k_var)] += 1


def _split_coarse_fine_pmis(graph, vars_update, coarse_vars, fine_vars, num_strong_connections):
    priorities = {}
    undecided = []
    for idx, var in enumerate(vars_update):
        if num_strong_connections[idx] >= 0:
            priorities[var] = (int(num_strong_connections[idx]), -var.variableID)
            undecided.append(var)

    undecided = set(undecided)

    while undecided:
        i_var = max(undecided, key=lambda var: priorities[var])
        undecided.remove(i_var)
        _promote_to_coarse(graph, i_var, coarse_vars)

        for j_var in i_var.multigrid.neighbour_vars:
            if j_var in undecided and j_var.multigrid.classification == "unassigned":
                j_var.multigrid.classification = "fine"
                fine_vars.append(j_var)
                undecided.remove(j_var)

    for var in vars_update:
        if var.multigrid.classification == "unassigned":
            var.multigrid.classification = "fine"
            fine_vars.append(var)


def _split_coarse_fine_pmis2(graph, vars_update, coarse_vars, fine_vars, num_strong_connections):
    priorities = {}
    strong_neighbours = {}
    undecided = []
    for idx, var in enumerate(vars_update):
        if num_strong_connections[idx] >= 0:
            priorities[var] = (int(num_strong_connections[idx]), -var.variableID)
            strong_neighbours[var] = [
                neigh for neigh, neigh_class in zip(var.multigrid.neighbour_vars, var.multigrid.neighbour_class)
                if neigh_class in {"strong", "coarse"}
            ]
            undecided.append(var)

    undecided = set(undecided)

    while undecided:
        i_var = max(undecided, key=lambda var: priorities[var])
        undecided.remove(i_var)
        _promote_to_coarse(graph, i_var, coarse_vars)

        radius_two = set()
        for j_var in strong_neighbours.get(i_var, []):
            radius_two.add(j_var)
            for k_var in strong_neighbours.get(j_var, []):
                radius_two.add(k_var)

        for j_var in radius_two:
            if j_var in undecided and j_var.multigrid.classification == "unassigned":
                j_var.multigrid.classification = "fine"
                fine_vars.append(j_var)
                undecided.remove(j_var)

    for var in vars_update:
        if var.multigrid.classification == "unassigned":
            var.multigrid.classification = "fine"
            fine_vars.append(var)


def coarsen_graph(graph, vars):
    # If performing iterative coarsening, then var_ids should be specified as the new variables 
    # Else we'll assume we're trying to coarsen the whole graph
    
    vars_update = vars.copy()

    # All neighbours of the new nodes and neighbours of those neighbours need to be reassigned as well
    for var in vars:
        var.multigrid.unassign()
        for factor in var.adj_factors:
            for adj_var in factor.adj_var_nodes:
                if adj_var not in vars_update:
                    adj_var.multigrid.unassign()
                    vars_update.append(adj_var)
                    # for factor2 in adj_var.adj_factors:
                    #     for adj_var2 in factor2.adj_var_nodes:
                    #         if adj_var2 not in vars_update:
                    #             adj_var2.multigrid.unassign()
                    #             vars_update.append(adj_var2)

    # vars_update = []
    # for var in graph.var_nodes:
    #     if var.type[0:5] != "multi":
    #         var.multigrid.unassign()
    #         vars_update.append(var)

    # Init lists for storing which nodes are set to coarse and fine and how many strong connections 
    num_strong_connections = np.ones(len(vars_update)).astype(int) * -1
    coarse_vars = []
    fine_vars = []

    for idx, var in enumerate(vars_update):
        var.multigrid.categorise_neighbours()
        # if "coarse" in var.multigrid.neighbour_class:
        #     var.multigrid.classification = "fine"
        #     fine_idx.append(idx)
        #     skip = True
        # else:
        num_strong_connections[idx] = int(var.multigrid.neighbour_class.count("strong"))

    # ----- Begin splitting coarse and fine nodes ---------------------------------

    split_mode = getattr(graph, "multigrid_split_mode", "rs")
    if split_mode == "pmis":
        _split_coarse_fine_pmis(graph, vars_update, coarse_vars, fine_vars, num_strong_connections)
    elif split_mode == "pmis2":
        _split_coarse_fine_pmis2(graph, vars_update, coarse_vars, fine_vars, num_strong_connections)
    else:
        _split_coarse_fine_rs(graph, vars_update, coarse_vars, fine_vars, num_strong_connections)


    # ----- Begin checking for coarse matches between fine nodes -----------------

    if getattr(graph, "enable_second_pass_coarse_match", True):
        for f1_node in fine_vars:
            f1_coarse_vars = [var for var in f1_node.multigrid.neighbour_vars if var.multigrid.classification == "coarse"]
            for f2_node in f1_node.multigrid.neighbour_vars:
                if f2_node.multigrid.classification == "fine":
                    f2_coarse_vars  = [var for var in f2_node.multigrid.neighbour_vars if var.multigrid.classification == "coarse"]
                    if not any([cVar in f2_coarse_vars for cVar in f1_coarse_vars]):
                        fine_vars.remove(f1_node)
                        _promote_to_coarse(graph, f1_node, coarse_vars)

                        for neighbour in f1_node.multigrid.neighbour_vars:
                            neighbour.multigrid.categorise_neighbours()

                        break

    # ---- All fine nodes have at least one matching coarse node between them -------
    # ---- Now we need to calculate the interpolation weightings for each nodes,
    # ---- including the neighbours of any newly added nodes ------------------------

    vars_update_neighbours = []
    for var in vars_update:
        for var_neighbour in var.multigrid.neighbour_vars:
            if var_neighbour not in vars_update_neighbours and var_neighbour not in vars_update:
                vars_update_neighbours.append(var_neighbour)
    
    vars_update.extend(vars_update_neighbours)

    # vars_update = []
    # for var in graph.var_nodes:
    #     if var.type[0:5] != "multi":
    #         vars_update.append(var)

    for var in vars_update:
        #var.active = True
        # for factor in var.adj_factors:
        #     if not factor.active:
        #         factor.active = True
        var.multigrid.categorise_neighbours()
        var.multigrid.update_restriction_interpolation()

    # for var in graph.multigrid_vars[0]:
    #     var.active = True
    #     var.multigrid.categorise_neighbours()
    #     var.multigrid.update_restriction_interpolation()

    # Restriction and interpolation operators created locally ----------------------
    # Each restriction operator is store in the lower level and each interpolation
    # operator is store in the upper level aggregate.

    # Now calculate linear operators for each coarse node that needs to be update
    # This is defined as any coarse nodes connected to any node in the "nodes_to_update" set

    high_level_vars_to_update = []
    for var in vars_update:
        for coarse_var in var.multigrid.restriction_vars:
            if coarse_var not in high_level_vars_to_update:
                 high_level_vars_to_update.append(coarse_var)

    # high_level_vars_to_update_all = []
    # for var in graph.multigrid_vars[1]:
    #     if var.type != "dead":
    #         high_level_vars_to_update_all.append(var)

    # Neighbours of any nodes where the weightings have changed need to also be included in the
    # coarse linear operator calculation and they effect coarse linkages

    for var in vars_update:
        for var_neighbour in var.multigrid.neighbour_vars:
            if var_neighbour not in vars_update:
                vars_update.append(var_neighbour)

    # Sub matrix Galerkin condition
    data = []
    i = []
    j = []
    dof = 2

    for i_idx, var in enumerate(vars_update):
        lam = var.prior.lam.copy()
        for factor in var.adj_factors:
            for factor_var in factor.adj_var_nodes:
                if factor_var != var:
                    try:
                        j_idx = vars_update.index(factor_var)
                        factor_lam = factor.factor.lam[0:dof, dof:dof+dof]
        
                        data.extend(np.diagonal(factor_lam))
                        i.append(i_idx*dof)
                        i.append(i_idx*dof+1)
                        j.append(j_idx*dof)
                        j.append(j_idx*dof+1)
                    except:
                        pass
                    lam += factor.factor.lam[0:dof,0:dof]
        data.extend(np.diagonal(lam))
        i.append(i_idx*dof)
        i.append(i_idx*dof+1)
        j.append(i_idx*dof)
        j.append(i_idx*dof+1)

    A_sub = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(len(vars_update) * dof, len(vars_update) * dof))
    A_sub = A_sub.tocsr()

    data = []
    i = []
    j = []
    
    for c_idx, coarse_var in enumerate(high_level_vars_to_update):
        for idx, var in enumerate(coarse_var.multigrid.interpolation_vars):
            try:
                j_idx = vars_update.index(var)  
                        
                data.extend(np.diagonal(coarse_var.multigrid.interpolation_coefficients[idx]))
                i.append(c_idx * dof)
                i.append(c_idx * dof + 1)
                j.append(j_idx * dof)
                j.append(j_idx * dof + 1)
            except:
                pass

    R_sub = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(len(high_level_vars_to_update) * dof, len(vars_update) * dof))
    R_sub = R_sub.tocsr()
    I_sub = R_sub.T

    A_coarse = R_sub @ A_sub @ I_sub

    # FULL Galerkin condition
    # data = []
    # i = []
    # j = []
    # dof = 2

    # for i_idx, var in enumerate(graph.multigrid_vars[0]):
    #     lam = var.prior.lam.copy()
    #     for factor in var.adj_factors:
    #         for factor_var in factor.adj_var_nodes:
    #             if factor_var != var:
    #                 try:
    #                     j_idx = graph.multigrid_vars[0].index(factor_var)
    #                     factor_lam = factor.factor.lam[0:dof, dof:dof+dof]
        
    #                     data.extend(np.diagonal(factor_lam))
    #                     i.append(i_idx*dof)
    #                     i.append(i_idx*dof+1)
    #                     j.append(j_idx*dof)
    #                     j.append(j_idx*dof+1)
    #                 except:
    #                     pass
    #                 lam += factor.factor.lam[0:dof,0:dof]
    #     data.extend(np.diagonal(lam))
    #     i.append(i_idx*dof)
    #     i.append(i_idx*dof+1)
    #     j.append(i_idx*dof)
    #     j.append(i_idx*dof+1)

    # A_full = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(len(graph.multigrid_vars[0]) * dof, len(graph.multigrid_vars[0]) * dof))
    # A_full = A_full.tocsr()

    # data = []
    # i = []
    # j = []
    
    # for c_idx, coarse_var in enumerate(high_level_vars_to_update_all):
    #     for idx, var in enumerate(coarse_var.multigrid.interpolation_vars):
    #         try:
    #             j_idx = graph.multigrid_vars[0].index(var)  
                        
    #             data.extend(np.diagonal(coarse_var.multigrid.interpolation_coefficients[idx]))
    #             i.append(c_idx * dof)
    #             i.append(c_idx * dof + 1)
    #             j.append(j_idx * dof)
    #             j.append(j_idx * dof + 1)
    #         except:
    #             pass

    # R_full = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(len(high_level_vars_to_update_all) * dof, len(graph.multigrid_vars[0]) * dof))
    # R_full = R_full.tocsr()
    # I_full = R_full.T

    # A_coarse_full = R_full @ A_full @ I_full

    for ci_idx, ci_var in enumerate(high_level_vars_to_update):
        ci_var.prior.lam = A_coarse[ci_idx*dof:ci_idx*dof+dof,ci_idx*dof:ci_idx*dof+dof].toarray()
        ci_var.prior.eta = np.zeros((dof,))
        non_zeros = A_coarse[ci_idx*dof,:].nonzero()
        for cj_idx in non_zeros[1]:
            cj_idx = int(cj_idx/2)
            if cj_idx > ci_idx:
                cj_var = high_level_vars_to_update[cj_idx]
                factor_exists = False
                lam_ij = A_coarse[ci_idx*dof:ci_idx*dof+dof,cj_idx*dof:cj_idx*dof+dof].toarray()
                lam_ii = -lam_ij
                ci_var.prior.lam -= lam_ii
                for factor in ci_var.adj_factors:
                    if cj_var in factor.adj_var_nodes:
                        factor.lam = np.concatenate( \
                                            (np.concatenate((lam_ii,lam_ij),axis=1), \
                                            np.concatenate((lam_ij,lam_ii),axis=1)) \
                                        ,axis=0)
                        factor.eta = np.zeros((dof*2,))
                        factor.gauss_noise_var = -new_factor.factor.lam[:dof,dof:dof+dof]
                        factor_exists = True
                        break

                if not factor_exists:
                    new_factor = gbp.Factor(graph.n_factor_nodes,
                                            [ci_var, cj_var],
                                            np.zeros(dof),
                                            np.zeros(dof),
                                            None,
                                            None,
                                            wildfire=graph.b_wild)
                    
                    new_factor.factor.lam = np.concatenate( \
                                            (np.concatenate((lam_ii,lam_ij),axis=1), \
                                            np.concatenate((lam_ij,lam_ii),axis=1)) \
                                        ,axis=0)
                    new_factor.gauss_noise_var = -new_factor.factor.lam[:dof,dof:dof+dof]
                    new_factor.factor.eta = np.zeros((dof*2,))

                    new_factor.type = "multigrid - multigrid lvl " + str(ci_var.multigrid.level)
                    new_factor.level = ci_var.multigrid.level

                    graph.factors.append(new_factor)
                    try:
                        graph.multigrid_factors[new_factor.level].append(new_factor)
                    except:
                        graph.multigrid_factors.append([])
                        graph.multigrid_factors[new_factor.level].append(new_factor)
                    graph.n_factor_nodes += 1

                    ci_var.adj_factors.append(new_factor)
                    cj_var.adj_factors.append(new_factor)
                
    for coarse_var in high_level_vars_to_update:
        coarse_var.update_belief()
        # Check neighbours for any unassigned vars
        # these need to be assigned in next coarsening level process if they exist
        for factor in coarse_var.adj_factors:
            for var in factor.adj_var_nodes:
                if var not in high_level_vars_to_update:
                    if var.multigrid.classification == "unassigned":
                        high_level_vars_to_update.append(var)

    level = coarse_var.multigrid.level
    if len(graph.multigrid_vars[level]) > 10:
        coarsen_graph(graph, high_level_vars_to_update)

    # Calculate the first R@A matrix multiplication for all the affected coarse nodes
    # RA tracker tracks whcih lower var index is being summed 
    # RA = []
    # RA_idx_tracker = []
    # for coarse_id in coarse_nodes_to_update:
    #     RA.append([])
    #     RA_idx_tracker.append([])
    #     for i_idx, i_var in enumerate(graph.var_nodes.multigrid.interpolation_vars):
    #         r_idx = i_var.variableID
    #         for factor in graph.var_nodes[r_idx].adj_factors:
    #             a_val





















#--------------- OLD VERSION --------------------

def create_coarse_level(graph):
    A, coarse_set, coarse_base_set, fine_set, interp_coeff, interp_neighbours, restrict_coeff, restrict_neighbours = partition_layer(graph)
    
    if len(coarse_set) > 1:
        graph.layers.append(amg_cls.layer(len(graph.layers)))
        
        n_c_vars = len(coarse_set)
        n_vars = len(graph.var_nodes)
        
        graph.layers[-1].n_vars = n_c_vars
        graph.layers[-1].n_vars_layer_below = graph.layers[-2].n_vars
        
        graph.layers[-1].A = A
        graph.layers[-1].b = np.zeros(len(coarse_set))
        graph.layers[-1].coarseIDs = coarse_set
        graph.layers[-1].coarse_baseIDs = coarse_base_set
        graph.layers[-1].fineIDs = fine_set
        graph.layers[-1].interp_neighbours = interp_neighbours
        graph.layers[-1].interp_coeff = interp_coeff
        graph.layers[-1].restrict_neighbours = restrict_neighbours
        graph.layers[-1].restrict_coeff = restrict_coeff
        graph.layers[-1].corrections = [None] * graph.layers[-1].n_vars_layer_below

        graph.layers[-1].var_ids =[*range(n_vars, n_vars + n_c_vars)]

        for cID in range(n_c_vars):
            new_var_node = gbp.VariableNode(n_vars + cID, 2)
            new_var_node.prior.eta = np.array([0.,0.]).astype(float)
            prior_lam = A[cID,cID]/2 
            #prior_lam = 1e-5
            new_var_node.prior.lam = np.array([[prior_lam,0],[0,prior_lam]]).astype(float)
            graph.var_nodes.append(new_var_node)

        for cID in range(n_c_vars):
            nonzeros = A[cID,:].nonzero()
            for j_id in nonzeros[1]:
                if j_id > cID:
                    new_factor = gbp.Factor(len(graph.factors),
                                    [graph.var_nodes[n_vars + cID], graph.var_nodes[n_vars + j_id]],
                                    [0., 0.],
                                    np.sqrt(1/(np.abs(A[cID, j_id])/2)),
                                    linear_displacement.meas_fn,
                                    linear_displacement.jac_fn,
                                    loss=None,
                                    mahalanobis_threshold=2)
                    
                    graph.var_nodes[n_vars + cID].adj_factors.append(new_factor)
                    graph.var_nodes[n_vars + j_id].adj_factors.append(new_factor)
                    graph.factors.append(new_factor)
        
        graph.layers[-1].factor_ids= [*range(graph.layers[-2].factor_ids[-1]+1, (len(graph.factors)))]

        return True
    
    else:

        return False

def partition_layer(graph):
    layer_below = graph.layers[-1]

    C = []
    C_base = []
    F = []
    unassigned_set = layer_below.var_ids.copy()

    theta_threshold = 0.9
    lambda_vals = np.zeros(layer_below.n_vars)
    set_i_depend_j = [[] for _ in range(layer_below.n_vars)]

    for idx, i in enumerate(layer_below.var_ids):
        incoming_lam_sum = []
        
        for factor in graph.var_nodes[i].adj_factors:
            off_diagonal_lams = -np.diag(factor.factor.lam)
            incoming_lam_sum.append(np.sum(off_diagonal_lams[:int(factor.dofs_conditional_vars/2)]))
            
        for factor_id, factor in enumerate(graph.var_nodes[i].adj_factors):
            if -incoming_lam_sum[factor_id] > theta_threshold * -max(incoming_lam_sum):
                set_i_depend_j[idx].extend([vID for vID in factor.adj_vIDs if vID != i])
        
        lambda_vals[idx] = len(set_i_depend_j[idx])

    while unassigned_set:
        candidate_index = np.argmax(lambda_vals)
        candidate = layer_below.var_ids[candidate_index]

        C.append(candidate)
        unassigned_set.remove(candidate)
        lambda_vals[candidate_index] = -1

        for factor in graph.var_nodes[candidate].adj_factors:
            j = [vID for vID in factor.adj_vIDs if vID != candidate]
            j = j[0]
            j_lambda_index = layer_below.var_ids.index(j)
            if lambda_vals[j_lambda_index] != -1:
                F.append(j)
                unassigned_set.remove(j)
                lambda_vals[j_lambda_index] = -1
                for k in set_i_depend_j[j_lambda_index]:
                    k_lambda_index = layer_below.var_ids.index(k)
                    if lambda_vals[k_lambda_index] != -1:
                        lambda_vals[k_lambda_index] += 1

    c_maybe = []
    
    for f in F:
        neighbours = []
        for factor in graph.var_nodes[f].adj_factors:
            neighbours.extend([x for x in factor.adj_vIDs if x != f])
        
        c_in_f = [x for x in neighbours if x in C]
        f_in_f = [x for x in neighbours if x not in C]
        for f2 in f_in_f:
            coarse_match = False
            for factor in graph.var_nodes[f2].adj_factors:
                for c in c_in_f:
                    if c in factor.adj_vIDs:
                        coarse_match = True
                        break
                if coarse_match:
                    break

            if not coarse_match:
                if f not in c_maybe:
                    c_maybe.append(f)


    C.extend(c_maybe)
    F = [f for f in F if f not in c_maybe]

    for c in C:
        base_index = layer_below.var_ids.index(c)
        base_id = layer_below.coarse_baseIDs[base_index]
        C_base.append(base_id)

    interp_neighbours = [[] for _ in range(layer_below.n_vars)]
    interp_coefficients = [[] for _ in range(layer_below.n_vars)]
    
    for vidx, vID in enumerate(layer_below.var_ids):
        if vID in C:
            interp_neighbours[vidx] = [vID]
            interp_coefficients[vidx] = [1.]
        else:
            neighbours = []
            lam_inward = []
            aij = []
            Ds = []
            Dw = []
            Ci = []
            for factor in graph.var_nodes[vID].adj_factors:
                neighbours.extend([x for x in factor.adj_vIDs if x != vID])
                dof = int(factor.dofs_conditional_vars/2)
                lam_inward.append(np.sum(-np.diag(factor.factor.lam[:dof,:dof])))

                if neighbours[-1] in C:
                    Ci.append(neighbours[-1])
                    aij.append(np.sum(-np.diag(factor.factor.lam[:dof,:dof])))
                elif neighbours[-1] in set_i_depend_j[vidx]:
                    Ds.append(neighbours[-1])
                else:
                    Dw.append(neighbours[-1])

            aii = -np.sum(lam_inward) + np.sum(np.diag(graph.var_nodes[vID].prior.lam))

                
            for j_id, j in enumerate(Ci):
                interp_neighbours[vidx].append(j)

                Ds_sum = 0.
                for m in Ds:
                    m_f_id = neighbours.index(m)
                    im_factor = graph.var_nodes[vID].adj_factors[m_f_id]
                    dof = int(im_factor.dofs_conditional_vars/2)
                    aim = np.sum(-np.diag(im_factor.factor.lam[:dof,:dof]))

                    mj_factor = None
                    for factor in graph.var_nodes[m].adj_factors:
                        if j in factor.adj_vIDs:
                            mj_factor = factor
                    
                    if mj_factor:
                        amj = np.sum(-np.diag(mj_factor.factor.lam[:dof,:dof]))
                    else:
                        amj = 0.
                    
                    amk_sum = 0.
                    for k in Ci:
                        mk_factor = None
                        for factor in graph.var_nodes[m].adj_factors:
                            if k in factor.adj_vIDs:
                                mk_factor = factor

                        if mk_factor:
                            amk_sum += np.sum(-np.diag(mk_factor.factor.lam[:dof,:dof]))

                    Ds_sum += (aim * amj) / amk_sum
                
                ain_sum = 0.
                for n in Dw:
                    n_id = neighbours.index(n)
                    in_factor = graph.var_nodes[vID].adj_factors[n_id]
                    dof = int(in_factor.dofs_conditional_vars/2)
                    ain_sum += np.sum(-np.diag(in_factor.factor.lam[:dof,:dof]))
                
                wij = - (aij[j_id] + Ds_sum)/(aii + ain_sum)


                interp_coefficients[vidx].append(wij)

    restrict_neighbours = [[] for _ in range(len(C))]
    restrict_coefficients = [[] for _ in range(len(C))]

    for cID in range(len(C)):
        for vidx, vID in enumerate(layer_below.var_ids):
            for coeff_id, interpID in enumerate(interp_neighbours[vidx]):
                if C[cID] == interpID:
                    restrict_neighbours[cID].append(vID)
                    restrict_coefficients[cID].append(interp_coefficients[vidx][coeff_id])

    data = []
    i = []
    j = []

    for vidx in range(layer_below.n_vars):
        for idx, id in enumerate(interp_neighbours[vidx]):
            data.append(interp_coefficients[vidx][idx])
            i.append(vidx)
            j.append(C.index(id))

    I = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(layer_below.n_vars, len(C)))
    #I = I.toarray()

    neighbours = [[] for _ in range(layer_below.n_vars)]
    data = []
    i = []
    j = []
    
    if layer_below.level == 0:
        for vidx, vID in enumerate(layer_below.var_ids):
            for factor in graph.var_nodes[vID].adj_factors:
                neighbours[vidx].extend([x for x in factor.adj_vIDs if x != vID])
                data.append(-np.sum(np.diag(factor.factor.lam[:dof])))
                i.append(vidx)
                j.append(layer_below.var_ids.index(neighbours[vidx][-1]))

            data.append(len(neighbours[vidx]) * -data[-1] + np.sum(np.diag(graph.var_nodes[vID].prior.lam)))
            i.append(vidx)
            j.append(vidx)
        
        A = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(layer_below.n_vars, layer_below.n_vars))
        A = A.tocsr()
        A_layer = I.T @ A @ I  # TODO: Make this vectorised and not matrix multiplication

    else:
        A_layer = I.T @ layer_below.A @ I


    return A_layer, C, C_base, F, interp_coefficients, interp_neighbours, restrict_coefficients, restrict_neighbours
