"""
normalisation methods
"""
from numpy import sqrt
from metric import q_metric


def space_light_timelike(position, input_velocity, params):
    """
    Checks sum of the trajectory with -+++ signature and computes the needed tt component
    """
    vv = input_velocity
    g = q_metric(position, params)
    u_t = sqrt(-(vv[0]**2 * g[1] + vv[1]**2 * g[2] + vv[2]**2 * g[3]) * g[0]**(-1))
    return u_t