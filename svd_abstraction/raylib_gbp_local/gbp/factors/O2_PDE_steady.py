import numpy as np

"""
Free space path loss factor for measurement in the log_10 domain h(x_1, x_2) = x_2 - x_1 and analogous form in higher dimensions.
"""


def meas_fn(x):
    J = np.array([np.concatenate(([(-len(x)+1)],np.ones(len(x)-1)))])
    return J @ x


def jac_fn(x):
    J = np.array([np.concatenate(([-len(x)+1],np.ones(len(x)-1)))])
    return J
