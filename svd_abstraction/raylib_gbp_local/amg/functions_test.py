from amg import classes_test as amg_cls
from gbp import gbp
import numpy as np
from scipy.sparse import coo_matrix
from gbp.factors import linear_displacement

def create_coarse_level(graph):
    A, coarse_set, coarse_base_set, fine_set, interp_coeff, interp_neighbours, restrict_coeff, restrict_neighbours = partition_layer(graph)
    
    if len(coarse_set) > 1:
        graph.layers.append(amg_cls.layer(len(graph.layers)))
        
        n_c_vars = len(coarse_set)
        n_vars = len(graph.var_nodes)
        
        graph.layers[-1].n_vars = n_c_vars
        graph.layers[-1].n_vars_layer_below = graph.layers[-2].n_vars
        
        graph.layers[-1].A = A
        graph.layers[-1].b = np.zeros(len(coarse_set)*2)
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
            new_var_node.prior.lam = np.array([[A[cID * 2, cID * 2],0],[0, A[cID * 2 + 1, cID * 2 + 1]]]).astype(float)
            #new_var_node.prior.lam = np.array([[np.sum(A[cID * 2, :]),0],[0, np.sum(A[cID * 2 + 1, :])]]).astype(float)
            graph.var_nodes.append(new_var_node)

        for cID in range(n_c_vars):
            nonzeros = A[cID * 2,:].nonzero()
            for j_id in nonzeros[1]:
                jg_id = int(j_id/2)
                if jg_id > cID:
                    new_factor = gbp.Factor(len(graph.factors),
                                    [graph.var_nodes[n_vars + cID], graph.var_nodes[n_vars + jg_id]],
                                    [0., 0.],
                                    np.sqrt(1/(np.abs(A[cID * 2, j_id]))),
                                    linear_displacement.meas_fn,
                                    linear_displacement.jac_fn,
                                    loss=None,
                                    mahalanobis_threshold=2)
                    
                    graph.var_nodes[n_vars + cID].adj_factors.append(new_factor)
                    graph.var_nodes[n_vars + jg_id].adj_factors.append(new_factor)
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

    theta_threshold = 0.2
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
            i.append(vidx * 2)
            j.append(C.index(id) * 2)
            data.append(interp_coefficients[vidx][idx])
            i.append(vidx * 2 + 1)
            j.append(C.index(id) * 2 + 1)

    I = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(layer_below.n_vars * 2, len(C) * 2))
    #I = I.toarray()

    neighbours = [[] for _ in range(layer_below.n_vars)]
    data = []
    i = []
    j = []
    
    if layer_below.level == 0:
        for vidx, vID in enumerate(layer_below.var_ids):
            for factor in graph.var_nodes[vID].adj_factors:
                neighbours[vidx].extend([x for x in factor.adj_vIDs if x != vID])
                data.append(-np.sum(np.diag(factor.factor.lam[:dof]))/2)
                i.append(2*vidx)
                j.append(2*layer_below.var_ids.index(neighbours[vidx][-1]))
                data.append(-np.sum(np.diag(factor.factor.lam[:dof]))/2)
                i.append(2*vidx+1)
                j.append(1+2*layer_below.var_ids.index(neighbours[vidx][-1]))

            data.append(len(neighbours[vidx]) * factor.factor.lam[0][0] + graph.var_nodes[vID].prior.lam[0][0])
            i.append(2*vidx)
            j.append(2*vidx)
            data.append(len(neighbours[vidx]) * factor.factor.lam[0][0] + graph.var_nodes[vID].prior.lam[1][1])
            i.append(2*vidx+1)
            j.append(2*vidx+1)
        
        
        A = coo_matrix((np.array(data), (np.array(i),np.array(j))), shape=(layer_below.n_vars*2, layer_below.n_vars*2))
        A = A.tocsr()
        A_layer = I.T @ A @ I  # TODO: Make this vectorised and not matrix multiplication

    else:
        A_layer = I.T @ layer_below.A @ I


    return A_layer, C, C_base, F, interp_coefficients, interp_neighbours, restrict_coefficients, restrict_neighbours
