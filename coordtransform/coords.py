"""
core functions of coordinate transforms
"""
from numba import jit
from numpy import cos, sin, sqrt, arccos, arctan2


# @jit(nopython=True)
def sph_to_karth(geschw):
    """
    takes a list of velocities in spherical coordinates and gives back in cartesian coordinates
    :param geschw:
    :return:
    """
    geschw_karth = []
    for elements in geschw:
        xx = elements[0] * cos(elements[2]) * sin(elements[1])
        yy = elements[0] * sin(elements[2]) * sin(elements[1])
        zz = elements[0] * cos(elements[1])
        geschw_karth.append([xx, yy, zz])
    return [xx, yy, zz]


def karth_to_sph(geschw):
    """
    takes a list of velocities  in cartesian coordinates and gives back in spherical
    :param geschw:
    :return:
    """
    geschw_sph = []
    for elements in geschw:
        xx = elements[0]
        yy = elements[1]
        zz = elements[2]
        rr = sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        theta = arccos(zz / rr)
        phi = arctan2(yy, xx)
        geschw_sph.append([rr, theta, phi])
    return geschw_sph
