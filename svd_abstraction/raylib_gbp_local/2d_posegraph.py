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
import matplotlib.pyplot as plt
from gbp import gbp
import visualiser.vis_2d_posegraph as vis
from amg import functions as amg_fnc
from amg import pyamg_layer as amg_pyamg



#@profile
def solver(graph, visualiser):
    
    prng = RandomState(69)

    graph.var_nodes = visualiser.var_nodes
    graph.factors = visualiser.factors

    graph.anchors = np.floor(prng.random_sample(5) * graph.n_var_nodes).astype(int)
    for anchor in graph.anchors:
        graph.var_nodes[anchor].prior.eta = graph.var_nodes[anchor].GT * 1e5
        graph.var_nodes[anchor].prior.lam = np.array([[1e5,0],[0,1e5]]).astype(float)
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
        #     amg_fnc.coarsen_graph(graph)
            
            # graph.layers[0].var_ids = [*range(n_vars)]
            # graph.layers[0].coarseIDs = [*range(n_vars)]
            # graph.layers[0].coarse_baseIDs = [*range(n_vars)]
            # graph.layers[0].factor_ids = [*range(len(graph.factors))]
            # graph.layers[0].n_vars = n_vars
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
    

    print(f'Number of variable nodes {graph.n_var_nodes}')

    if not visualiser.b_wild:

        # graph.mu_GT, sig_GT = graph.joint_distribution_cov()  # Get batch solution
        # graph.sig_GT = np.diag(sig_GT)

        # for i in range(int(len(graph.mu_GT)/2)):
        #     graph.var_nodes[i].mu = [graph.mu_GT[2*i], graph.mu_GT[2*i+1]]

        if visualiser.b_multi:
            graph.vcycle_loop(visualiser)
        else:
            graph.synchronous_loop(visualiser)

        print('SYNC ITERS FINISHED')

    else:
        graph.wildfire_iteration(visualiser)
        
        print(f'Msgs {graph.n_msgs}  //')
        print('WILDFIRE FINISHED')



    # de-Initialization
    while not visualiser.reset_event.is_set() and not visualiser.stop_event.is_set():
        time.sleep(0.5)
      

error_histories = []
energy_histories = []
nmsgs_histories = []

graph = gbp.FactorGraph(nonlinear_factors=False, wild_thresh=1e-3, eta_damping=0.0)
visualiser = vis.game(graph=graph, n_rand=100)


def controller_loop():
    global graph

    while not visualiser.stop_event.is_set():
        while not visualiser.b_run and not visualiser.stop_event.is_set():
            time.sleep(0.1)

        if visualiser.stop_event.is_set():
            break

        solver(graph, visualiser)
        if visualiser.stop_event.is_set():
            break

        error_histories.append(graph.error_history)
        energy_histories.append(graph.energy_history)
        nmsgs_histories.append(graph.nmsgs_history)
        if visualiser.b_show_plots:
            fig, axs = plt.subplots(2,2,  figsize=(15, 9))
            for run in range(len(error_histories)):
                axs[0,0].plot(error_histories[run])
                axs[1,0].plot(nmsgs_histories[run], error_histories[run])
                axs[0,1].plot(energy_histories[run])
                axs[1,1].plot(nmsgs_histories[run], energy_histories[run])
            for ax in axs.flat:
                ax.grid()
            axs[1,0].set(xlabel='messages sent', ylabel='average error from ground truth (pixels)',
                    title='error vs messages')
            axs[0,0].set(ylabel='average error from ground truth (pixels)',
                    title='error vs iterations')
            axs[1,1].set(xlabel='messages sent', ylabel='total energy of all base level factors',
                    title='energy vs messages')
            axs[0,1].set(ylabel='total energy of all base level factors',
                    title='energy vs iterations')
            plt.show(block=False)
        visualiser.reset_event.clear()
        graph = gbp.FactorGraph(nonlinear_factors=False, wild_thresh=1e-3)
        visualiser.graph = graph
        print("GRAPH RESET!")


solver_thread = Thread(target=controller_loop, daemon=True)
solver_thread.start()
visualiser.run()
solver_thread.join(timeout=1.0)
