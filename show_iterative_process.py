import matplotlib.pyplot as plt
import os
from copy import deepcopy
from initialconditions import lin_comb, space_light_timelike, check_tetrad, vartheta_raster
from decimal import Decimal
from metric import q_metric
import multiprocessing as mp
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy, set_printoptions, save
from scipy.integrate import solve_ivp, odeint
from plotting import plot_data_3d
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp


def too_deep(t, y, *args):
    return y[1] - 2.1


def too_far(t, y, *args):
    return 10 - abs(y[1])


def one_calculation(dist, mm_2, qq_2):
    bx = plt.axes()
    init_ort = array([0.0, dist, pi / 2, 0.0])
    init_three_velocity_list = check_tetrad(init_ort, qq_2, mm_2)
    init_three_velocity = init_three_velocity_list[0]
    u_t = space_light_timelike(init_ort, init_three_velocity, [qq_2, mm_2])
    init_4_vel = [u_t, *init_three_velocity]
    e = (1 - 2 * mm_2 / dist) ** (1 + qq_2) * u_t  # Energy, constant of motion
    angular_mom = (1 - 2 * mm_2 / dist) ** (-qq_2) * dist ** 2 * sin(init_ort[2]) * init_4_vel[3]  # constant of motion
    u_t, u_r, u_theta, u_phi = init_4_vel
    tt, rr1, theta_1, phi1 = init_ort
    sol = solve_ivp(quadrupol_ivp,
                    # schwarzschild_ivp,
                    t_span=[0, 30],
                    t_eval=linspace(0, 30, 5000),
                    y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                    args=(mm_2, qq_2, e, angular_mom),
                    # y0=array([tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi]),
                    vectorized=True,
                    method='RK45',
                    atol=1e-12,
                    rtol=1e-12,
                    max_step=1e-3,
                    events=too_deep
                    )
    result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])

    plot_data_3d(result_spherical, bx)
    plt.show()


# 3.0025485803049516 kaputt bei 8. iteration mit länge 30

def iterate_calculation(iter_list, qq_2, mm_2):
    falls_in = 1
    falling_list = []
    turning_point = None
    trajectories_here = []
    #aax_1 = plt.axes()
    minimal_distance = []
    for ii_1, dist in enumerate(iter_list):
        init_ort = array([0.0, dist, pi / 2, 0.0])
        init_three_velocity_list = check_tetrad(init_ort, qq_2, mm_2)
        init_three_velocity = init_three_velocity_list[0]
        u_t = space_light_timelike(init_ort, init_three_velocity, [qq_2, mm_2])
        init_4_vel = [u_t, *init_three_velocity]
        e = (1 - 2 * mm_2 / dist) ** (1 + qq_2) * u_t  # Energy, constant of motion
        angular_mom = (1 - 2 * mm_2 / dist) ** (-qq_2) * dist ** 2 * sin(init_ort[2]) * init_4_vel[3]  # constant of motion
        u_t, u_r, u_theta, u_phi = init_4_vel
        tt, rr1, theta_1, phi1 = init_ort
        sol = solve_ivp(quadrupol_ivp,
                        # schwarzschild_ivp,
                        t_span=[0, 25],
                        t_eval=linspace(0, 25, 5000),
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mm_2, qq_2, e, angular_mom),
                        # y0=array([tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi]),
                        vectorized=True,
                        method='RK45',
                        atol=1e-12,
                        rtol=1e-12,
                        max_step=1e-3,
                        events=too_deep
                        )
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])
        trajectories_here.append(deepcopy(result_spherical))

        if sol.y[1][-1] < dist:
            falls_in = 1
        elif sol.y[1][-1] > dist:
            if falls_in == 1 and turning_point is None:
                turning_point = ii_1
            falls_in = 0
        falling_list.append(falls_in)
        minimal_distance.append(deepcopy((sol.y[1][-1] - dist)))
        del result_spherical
        del sol
    print(falling_list)
    #plt.show()

    # plt.close()

    return turning_point, falling_list, trajectories_here, minimal_distance


def return_call(initial_list, quad_param, mass):
    index_thing, list_this, result, distance_from_origin_list = iterate_calculation(initial_list, quad_param, mass)
    if 0 in list_this and 1 not in list_this:
        print("warnung alles nullen")
    if 1 in list_this and 0 not in list_this:
        print("warnung alles einsen")
    upper_thing = initial_list[index_thing]
    lower_thing = initial_list[index_thing - 1]
    return upper_thing, lower_thing, result, list_this, distance_from_origin_list


def iter_loop_this(qq_1, mm_1, ii_1, jj_1):
    lower_thing, upper_thing = 3*mm_1 + 2 * qq_1 - 0.5, 3*mm_1 + 2 * qq_1 + 1
    resultate = []
    liste_des_fallens = []
    for kk in range(10):
        print(f"loop {kk}")
        init_list = linspace(lower_thing, upper_thing, 10)
        upper_thing, lower_thing, results, liste_hier, d_or_list = return_call(init_list, qq_1, mm_1)
#        if kk == 4:
#            liste_hier = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#            lower_thing, upper_thing = init_list[1], init_list[2]
        resultate.append(deepcopy(results))
        liste_des_fallens.append(deepcopy(liste_hier))
        plt.plot(d_or_list, "bo")
        plt.savefig(f"{kk}.png")
        plt.close()

    print(liste_des_fallens)
    endradius = 0.5 * (lower_thing + upper_thing)
    print(f"qq:{qq_1}:{endradius}")
    if not os.path.isdir(f"/home/altin/ll_geod_scipy_ivp/data/m{ii_1}/"):
        os.mkdir(f"/home/altin/ll_geod_scipy_ivp/data/m{ii_1}/")
    save(f"/home/altin/ll_geod_scipy_ivp/data/m{ii_1}/q{jj_1}", endradius)
    return resultate, liste_des_fallens, endradius


if __name__ == "__main__":
    set_printoptions(precision=16)
    too_deep.terminal = True
    too_far.terminal = False
    plot_plane = "xz"

    quad = .1
    ii = 0
    mm = 1
    resultate_1, liste_of_sachen, ende_gelaende = iter_loop_this(quad, mm, ii, 0)

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 10))

    for ii, geo in enumerate(resultate_1[0]):
        if liste_of_sachen[0][ii]==1:
            plot_data_3d(geo,ax[0, 0], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[0, 0], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[1]):
        if liste_of_sachen[1][ii]==1:
            plot_data_3d(geo,ax[0, 1], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[0, 1], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[2]):
        if liste_of_sachen[2][ii]==1:
            plot_data_3d(geo,ax[0, 2], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[0, 2], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[3]):
        if liste_of_sachen[3][ii]==1:
            plot_data_3d(geo,ax[0, 3], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[0, 3], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[4]):
        if liste_of_sachen[4][ii]==1:
            plot_data_3d(geo,ax[0, 4], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[0, 4], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[5]):
        if liste_of_sachen[5][ii]==1:
            plot_data_3d(geo,ax[1, 0], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[1, 0], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[6]):
        if liste_of_sachen[6][ii]==1:
            plot_data_3d(geo,ax[1, 1], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[1, 1], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[7]):
        if liste_of_sachen[7][ii]==1:
            plot_data_3d(geo,ax[1, 2], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[1, 2], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[8]):
        if liste_of_sachen[8][ii]==1:
            plot_data_3d(geo,ax[1, 3], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[1, 3], black_geodesic=False)

    for ii, geo in enumerate(resultate_1[9]):
        if liste_of_sachen[9][ii]==1:
            plot_data_3d(geo,ax[1, 4], black_geodesic=True)
        else:
            plot_data_3d(geo, ax[1, 4], black_geodesic=False)

    title_list = ["i=1", "i=2", "i=3", "i=4", "i=5", "i=6", "i=7", "i=8", "i=9", "i=10"]
    ii = 0
    for axis_list in ax:
        for axis in axis_list:
            axis.set_xlim([-5, 5])
            axis.set_ylim([-5, 5])
            axis.set_title(title_list[ii], fontsize=20)
            axis.tick_params(axis='both', which='major', labelsize=20)
            ii += 1

    for ax1 in ax.flat:
        ax1.set(xlabel='x', ylabel='z')
        ax1.xaxis.label.set_size(20)
        ax1.yaxis.label.set_size(20)

    for ax1 in ax.flat:
        ax1.label_outer()

    plt.show()

    one_calculation(ende_gelaende, mm, quad)

