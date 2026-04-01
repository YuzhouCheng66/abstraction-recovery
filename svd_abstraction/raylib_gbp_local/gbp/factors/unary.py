import numpy as np

"""
Unary factor for measurement function h(x_1) = 0 and analogous form in higher dimensions.
"""


def meas_fn(x):
    J = np.array([1])
    return J @ x


def jac_fn(x):
    return np.array([1])