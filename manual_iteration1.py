import matplotlib.pyplot as plt
import os
from copy import deepcopy
from initialconditions import lin_comb, space_light_timelike, check_tetrad, vartheta_raster
from decimal import Decimal
from metric import q_metric
import multiprocessing as mp
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy, set_printoptions, save, load
from scipy.integrate import solve_ivp, odeint
from plotting import plot_data_3d
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp
from extensisq import CKdisc


def too_deep(t, y, *args):
    return y[1] - 2.1


def too_far(t, y, *args):
    return 10 - abs(y[1])


def one_calculation(dist_this, mm_2, qq_2):
    init_ort = array([0.0, dist_this, pi / 2, 0.0])
    init_three_velocity_list = check_tetrad(init_ort, qq_2, mm_2)
    init_three_velocity = init_three_velocity_list[0]
    u_t = space_light_timelike(init_ort, init_three_velocity, [qq_2, mm_2])
    init_4_vel = [u_t, *init_three_velocity]
    e = (1 - 2 * mm_2 / dist_this) ** (1 + qq_2) * u_t  # Energy, constant of motion
    angular_mom = (1 - 2 * mm_2 / dist_this) ** (-qq_2) * dist_this ** 2 * sin(init_ort[2]) * init_4_vel[3]  # constant of motion
    u_t, u_r, u_theta, u_phi = init_4_vel
    tt, rr1, theta_1, phi1 = init_ort
    sol = solve_ivp(quadrupol_ivp,
                    # schwarzschild_ivp,
                    t_span=[0, 300],
                    t_eval=linspace(0, 300, 10000),
                    y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                    args=(mm_2, qq_2, e, angular_mom),
                    # y0=array([tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi]),
                    vectorized=True,
                    method=CKdisc,
                    atol=1e-12,
                    rtol=1e-12,
                    max_step=5e-3,
                    events=(too_deep, too_far)
                    )

    result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])
    if not os.path.isdir(dirstring):
        os.mkdir(dirstring)
    save(dirstring + f"r{dist_this}", result_spherical)

    del result_spherical


if __name__ == "__main__":
    set_printoptions(precision=16)
    num_processes = os.cpu_count()
    too_deep.terminal = True
    too_far.terminal = True
    plot_plane = "xz"
    sub_procs = []

    quad = .1
    ii = 0
    mm = 1
    lower_thing, upper_thing = 3 * mm + 2 * quad - 0.5, 3 * mm + 2 * quad + 1
    ii = 1
    init_list = linspace(lower_thing, upper_thing, 8)
    ii += 1
    init_list = linspace(init_list[1], init_list[3], 8)
    ii += 1
    init_list = linspace(init_list[5], init_list[6], 8)
    ii += 1
    init_list = linspace(init_list[3], init_list[4], 8)
    ii += 1
    init_list = linspace(init_list[4], init_list[5], 8)
    ii += 1
    init_list = linspace(init_list[4], init_list[5], 8)
    ii += 1
    init_list = linspace(init_list[1], init_list[2], 8)
    ii += 1
    init_list = linspace(init_list[1], init_list[2], 8)
    ii += 1
    init_list = linspace(init_list[3], init_list[4], 8)
    ii += 1
    # init_list = linspace(init_list[4], init_list[5], 8)
    # ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    if True:
        for dist in init_list:
            p = mp.Process(target=one_calculation,
                           args=(dist, mm, quad))
            sub_procs.append(p)
            if len(sub_procs) == 4:
                for pp in sub_procs:
                    pp.start()
                for pp in sub_procs:
                    pp.join()
                for pp in sub_procs:
                    pp.terminate()
                sub_procs = []
    bx_here = plt.axes()
    for file in os.listdir(dirstring):
        result_spherical = load(dirstring + file)
        plot_data_3d(result_spherical, bx_here)
    plt.show()

