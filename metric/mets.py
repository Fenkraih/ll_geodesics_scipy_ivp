"""
Metric methods for computations
"""
from numba import jit
from numpy import sin, cos, tan, array, zeros


# @jit(nopython=True)
def q_metric(position, params):
    """
    returns the metric
    """
    tt, r, theta, phi = position
    q, M = params
    g00 = -(1. - 2. * M / r)**(1+q)
    g11 = (1. - 2. * M / r)**(-q-1) * (1 + (M**2 * sin(theta)**2)/(r**2 - 2*M*r))**(-q*(2+q))
    g22 = (1. - 2. * M / r)**(-q) * (1 + (M**2 * sin(theta)**2)/(r**2 - 2*M*r))**(-q*(2+q)) * r**2
    g33 = (1. - 2. * M / r)**(-q) * r ** 2 * sin(theta) ** 2
    g = [g00, g11, g22, g33]
    return g


# @jit(nopython=True)
def christoffel_precalculated(position, params):
    """
    Copy pasted the results of the sympy based gamma() function here to save computation time
    """
    tt, r, theta, phi = position
    q, M = params
    gam = zeros([4, 4, 4])

    gam[0][0][1] = -1.0 * M * (q + 1) / (r * (2 * M - r))
    gam[0][1][0] = -1.0 * M * (q + 1) / (r * (2 * M - r))
    gam[1][0][0] = -1.0 * M * (-(2 * M - r) / r) ** (2 * q + 2) * (
                   -(M ** 2 * sin(theta) ** 2 / r - 2 * M + r) / (2 * M - r)) ** (q * (q + 2)) * (q + 1) / (
                                  r * (2 * M - r))
    gam[1][1][1] = 1.0 * M * (-M * q * (M - r) * (q + 2) * sin(theta) ** 2 + (q + 1) * (
                   -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2)) / (
                                  r * (2 * M - r) * (-M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2))
    gam[1][1][2] = 1.0 * M ** 2 * q * (q + 2) * sin(2 * theta) / (
                   M ** 2 * cos(2 * theta) - M ** 2 + 4 * M * r - 2 * r ** 2)
    gam[1][2][1] = 1.0 * M ** 2 * q * (q + 2) * sin(2 * theta) / (
                   M ** 2 * cos(2 * theta) - M ** 2 + 4 * M * r - 2 * r ** 2)
    gam[1][2][2] = (-2 * M + r) * (M ** 2 * q * (M - r) * (q + 2) * sin(theta) ** 2 - 1.0 * M * q * (
                   -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2) + (-2.0 * M + 1.0 * r) * (
                                                  -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2)) / (
                                  (2 * M - r) * (-M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2))
    gam[1][3][3] = 1.0 * ((-M ** 2 * sin(theta) ** 2 / r + 2 * M - r) / (2 * M - r)) ** (q * (q + 2)) * (
                   -2 * M + r) * (-M * q - 2 * M + r) * sin(theta) ** 2 / (2 * M - r)
    gam[2][1][1] = 1.0 * M ** 2 * q * (q + 2) * sin(2 * theta) / (
                   r * (-2 * M + r) * (-M ** 2 * cos(2 * theta) + M ** 2 - 4 * M * r + 2 * r ** 2))
    gam[2][1][2] = 1.0 * (-M ** 2 * q * (M - r) * (q + 2) * sin(theta) ** 2 + M * q * (
                   -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2) + (2 * M - r) * (
                                         -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2)) / (
                                  r * (2 * M - r) * (-M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2))
    gam[2][2][1] = 1.0 * (-M ** 2 * q * (M - r) * (q + 2) * sin(theta) ** 2 + M * q * (
                   -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2) + (2 * M - r) * (
                                         -M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2)) / (
                                  r * (2 * M - r) * (-M ** 2 * sin(theta) ** 2 + 2 * M * r - r ** 2))
    gam[2][2][2] = 1.0 * M ** 2 * q * (q + 2) * sin(2 * theta) / (
                   M ** 2 * cos(2 * theta) - M ** 2 + 4 * M * r - 2 * r ** 2)
    gam[2][3][3] = -0.5 * ((M ** 2 * cos(2 * theta) / (2 * r) - M ** 2 / (2 * r) + 2 * M - r) / (2 * M - r)) ** (
                   q * (q + 2)) * sin(2 * theta)
    gam[3][1][3] = 1.0 * (M * q + 2 * M - r) / (r * (2 * M - r))
    gam[3][2][3] = 1.0 / tan(theta)
    gam[3][3][1] = 1.0 * (M * q + 2 * M - r) / (r * (2 * M - r))
    gam[3][3][2] = 1.0 / tan(theta)
    return gam


# @jit(nopython=True)
def accel(xx, vv, params):
    """
    Calculates the acceleration via geodesic equation
    :param xx: location
    :param vv: velocity
    :param params:
    :return:
    """
    rang = range(4)
    aa = []
    gam = christoffel_precalculated(xx, params)
    for ii in rang:
        accel = 0
        for jj in rang:
            for kk in rang:
                accel += gam[ii][jj][kk]*vv[jj]*vv[kk]
        aa.append(accel)
    return -1*array(aa)
