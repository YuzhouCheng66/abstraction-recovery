"""
raylib [core] example - 2d camera mouse zoom
"""

import pyray


"""
2D grid estimation.

Linear problem where we are estimating the position in N-dim space of a number of nodes.
Linear factors connect each node to the M closest nodes in the space.
The linear factors measure the distance between the nodes in each of the N dimensions.
"""

import numpy as np
from numpy.random import RandomState
import time
from threading import Thread
from gbp import gbp
import visualiser.vis_slam as vis
from amg import functions as amg_fnc
from amg import pyamg_layer as amg_pyamg



#@profile
def solver(graph, visualiser):
    
    prng = RandomState(69)

    graph.var_nodes = visualiser.var_nodes
    graph.factors = visualiser.factors
    
    # eta, lam = graph.joint_distribution_inf()
    
    use_pyamg = True

    if visualiser.b_multi:
        if visualiser.b_pyamg:
            A, A_full, b, b_full = amg_pyamg.base_A_mat(graph)
            amg_pyamg.create_layers(A, b, graph, A_full, b_full)
            for layer in graph.layers:
                if layer.level != 0:
                    visualiser.C_var_ids.append(layer.var_ids)
                    visualiser.C.append(layer.coarseIDs)
                    visualiser.C_base_ids.append(layer.coarse_baseIDs)
                    visualiser.layer_factor_ids.append(layer.factor_ids)
                    graph.update_all_beliefs(layer=layer.level)
                    graph.compute_all_factors(layer=layer.level)

        # else:
        #     amg_fnc.coarsen_graph(graph, graph.var_nodes)
        #     visualiser.n_factors = int(graph.n_factor_nodes)
        #     visualiser.n_vars = int(graph.n_var_nodes)

            # graph.layers[0].var_ids = [*range(graph.n_var_nodes)]
            # graph.layers[0].coarseIDs = [*range(graph.n_var_nodes)]
            # graph.layers[0].coarse_baseIDs = [*range(graph.n_var_nodes)]
            # graph.layers[0].factor_ids = [*range(len(graph.factors))]
            # graph.layers[0].n_vars = graph.n_var_nodes
            # graph.multigrid = True
            
            # for n_coarse_layer in range(4):
            #     bool_layer_created = amg_fnc.create_coarse_level(graph)
            #     if bool_layer_created:
            #         visualiser.C_var_ids.append(graph.layers[-1].var_ids)
            #         visualiser.C.append(graph.layers[-1].coarseIDs)
            #         visualiser.C_base_ids.append(graph.layers[-1].coarse_baseIDs)
            #         visualiser.layer_factor_ids.append(graph.layers[-1].factor_ids)
            #         graph.update_all_beliefs(layer=graph.layers[-1].level)
            #         graph.compute_all_factors(layer=graph.layers[-1].level)

    if not visualiser.b_wild:

        # graph.mu_GT, sig_GT = graph.joint_distribution_cov()  # Get batch solution
        # graph.sig_GT = np.diag(sig_GT)

        if visualiser.b_multi:
            graph.vcycle_loop(visualiser)
        else:
            graph.synchronous_loop(visualiser)

        print('SYNC ITERS FINISHED')

    else:
        graph.b_wild= True
        graph.Q = [graph.factors[0]]
        graph.wildfire_iteration(visualiser)
        
        print(f'Msgs {graph.n_msgs}  //')
        print('WILDFIRE FINISHED')


    # de-Initialization
    while not visualiser.reset_event.is_set():
        time.sleep(0.5)
      
graph = gbp.FactorGraph(nonlinear_factors=False, wild_thresh=1e-9, eta_damping=0)
visualiser = vis.game(graph=graph, n_rand=100)
vis_thread = Thread(target=visualiser.run)
vis_thread.start()

while True:

    while not visualiser.b_run:
        time.sleep(0.1)

    solver(graph, visualiser)
    visualiser.reset_event.clear()
    graph = gbp.FactorGraph(nonlinear_factors=False, wild_thresh=1e-9, eta_damping=0)
    visualiser.graph = graph
    print("GRAPH RESET!")
