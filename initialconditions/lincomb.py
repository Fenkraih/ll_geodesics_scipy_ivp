from numba import jit
from numpy import array, pi, sqrt
from coordtransform.coords import sph_to_karth
from metric import q_metric


# @jit(nopython=True)
def lin_comb(ort, quad_param, mass, angles, steps):
    g = q_metric(ort, [quad_param, mass])
    e1 = [-1.0 * (1 / sqrt(g[1])), 0, 0]  # forward
    e2 = [0, -1.0 * (1 / sqrt(g[2])), 0]  # upward
    e3 = [0, 0, -1.0 * (1 / sqrt(g[3]))]  # leftward

    theta_low, theta_up, phi_low, phi_up = angles
    coords = []

    if theta_low == theta_up and phi_low !=phi_up:
        for ll in range(steps + 1):
            sphere = [1.0, theta_low,
                      phi_low + ll * (phi_up - phi_low) / steps]
            sph_karth = sph_to_karth([sphere])
            ay, ax, az = sph_karth  # eigentlich ax ay az
            local_coord = array([ax * e1[0], ay * e2[1], az * e3[2]])
            coords.append(local_coord)

    if theta_low == theta_up and phi_low == phi_up:
        # print("here")
        sphere = [1.0, theta_low,
                  phi_low]
        sph_karth = sph_to_karth([sphere])
        ay, ax, az = sph_karth
        print(f"init kugel {sph_karth}")
        local_coord = array([ax * e1[0], ay * e2[1], az * e3[2]])
        coords.append(local_coord)

    elif phi_low == phi_up and theta_low != theta_up:
        for kk in range(steps + 1):
            sphere = [1.0, theta_low + kk * (theta_up - theta_low) / steps, phi_low]
            sph_karth = sph_to_karth([sphere])
            ax, ay, az = sph_karth
            local_coord = array([ax * e1[0], ay * e2[1], az * e3[2]])
            coords.append(local_coord)

    else:
        for kk in range(steps + 1):
            for ll in range(steps + 1):
                sphere = [1.0, theta_low + kk * (theta_up - theta_low) / steps,
                          phi_low + ll * (phi_up - phi_low) / steps]
                sph_karth = sph_to_karth([sphere])
                ax, ay, az = sph_karth
                local_coord = array([ax * e1[0], ay * e2[1], az * e3[2]])
                coords.append(local_coord)
    return coords


def lin_comb_2():
    return None
