import matplotlib.pyplot as plt
import os
from copy import deepcopy
from initialconditions import lin_comb, space_light_timelike, check_tetrad, vartheta_raster
from decimal import Decimal
from metric import q_metric
import multiprocessing as mp
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy, set_printoptions, save, load, sort
from scipy.integrate import solve_ivp, odeint
from plotting import plot_data_3d
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp
from extensisq import CKdisc


def too_deep(t, y, *args):
    return y[1] - 2.1


def too_far(t, y, *args):
    return 9 - abs(y[1])


def one_calculation(dist_this, mm_2, qq_2):
    init_ort = array([0.0, dist_this, pi / 2, 0.0])
    init_three_velocity_list = check_tetrad(init_ort, qq_2, mm_2)
    init_three_velocity = init_three_velocity_list[0]
    u_t = space_light_timelike(init_ort, init_three_velocity, [qq_2, mm_2])
    init_4_vel = [u_t, *init_three_velocity]
    e = (1 - 2 * mm_2 / dist_this) ** (1 + qq_2) * u_t  # Energy, constant of motion
    angular_mom = (1 - 2 * mm_2 / dist_this) ** (-qq_2) * dist_this ** 2 * sin(init_ort[2]) * init_4_vel[
        3]  # constant of motion
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


def cald_this(init_list, mm, quad):
    sub_procs = []
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


def generate_date():
    global dirstring
    lower_thing, upper_thing = 3 * mm + 2 * quad - 0.5, 3 * mm + 2 * quad + 1

    ii = 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(lower_thing, upper_thing, 8)
    dark = [1, 1, 1, 0, 0, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[1], init_list[3], 8)
    dark = [1, 1, 1, 1, 1, 1, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[5], init_list[6], 8)
    dark = [1, 1, 1, 1, 0, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[3], init_list[4], 8)
    dark = [1, 1, 1, 1, 1, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[4], init_list[5], 8)
    dark = [1, 1, 1, 1, 1, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[4], init_list[5], 8)
    dark = [1, 1, 0, 0, 0, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[1], init_list[2], 8)
    dark = [1, 1, 0, 0, 0, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)

    ii += 1
    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{ii}/"
    init_list = linspace(init_list[1], init_list[2], 8)
    dark = [1, 1, 1, 1, 0, 0, 0, 0]
    print(ii)
    print(array(init_list[-1] - init_list[0]))
    # cald_this(init_list, mm, quad)


def plotting_data():
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(5, 5))

    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{1}/"
    dark = [1, 1, 1, 0, 0, 0, 0, 0]
    list_of_files = []
    for elements in os.listdir(dirstring):
        elements = elements.replace("r", "")
        elements = elements.replace(".npy", "")
        list_of_files.append(float(elements))
    list_of_files = sort(list_of_files)
    list_of_files = [str(x) for x in list_of_files]
    #print(list_of_files)
    for index_thing, file in enumerate(list_of_files):
        result_spherical = load(dirstring + f"r{file}.npy")
        plot_data_3d(result_spherical, ax[0], black_geodesic=dark[index_thing])

    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{3}/"
    dark = [1, 1, 1, 1, 0, 0, 0, 0]
    list_of_files = []
    for elements in os.listdir(dirstring):
        elements = elements.replace("r", "")
        elements = elements.replace(".npy", "")
        list_of_files.append(float(elements))
    list_of_files = sort(list_of_files)
    list_of_files = [str(x) for x in list_of_files]
    #print(list_of_files)
    for index_thing, file in enumerate(list_of_files):
        result_spherical = load(dirstring + f"r{file}.npy")
        plot_data_3d(result_spherical, ax[1], black_geodesic=dark[index_thing])

    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{5}/"
    dark = [1, 1, 1, 1, 1, 0, 0, 0]
    list_of_files = []
    for elements in os.listdir(dirstring):
        elements = elements.replace("r", "")
        elements = elements.replace(".npy", "")
        list_of_files.append(float(elements))
    list_of_files = sort(list_of_files)
    list_of_files = [str(x) for x in list_of_files]
    #print(list_of_files)
    for index_thing, file in enumerate(list_of_files):
        result_spherical = load(dirstring + f"r{file}.npy")
        plot_data_3d(result_spherical, ax[2], black_geodesic=dark[index_thing])

    dirstring = f"/home/altin/ll_geod_scipy_ivp/data/it_{8}/"
    dark = [1, 1, 1, 1, 0, 0, 0, 0]
    list_of_files = []
    for elements in os.listdir(dirstring):
        elements = elements.replace("r", "")
        elements = elements.replace(".npy", "")
        list_of_files.append(float(elements))
    list_of_files = sort(list_of_files)
    list_of_files = [str(x) for x in list_of_files]
    #print(list_of_files)
    for index_thing, file in enumerate(list_of_files):
        result_spherical = load(dirstring + f"r{file}.npy")
        plot_data_3d(result_spherical, ax[3], black_geodesic=dark[index_thing])

    for bx in ax:
        xval = linspace(-2.1, 2.1, 1000)
        yval = sqrt(2.1 ** 2 - xval ** 2)
        bx.plot(xval, yval, "r")
        bx.plot(xval, -yval, "r")

        surface = 9
        xval = linspace(-surface, surface, 1000)
        yval = sqrt(surface ** 2 - xval ** 2)
        bx.plot(xval, yval, "c")
        bx.plot(xval, -yval, "c")

    title_list = ["i=1, l≈1 e0", "i=3, l≈7 e-2", "i=5, l≈2 e-3", "i=8, l≈4 e-6"]
    for kk, axis in enumerate(ax):
        axis.set_title(title_list[kk], fontsize=20)
        axis.tick_params(axis='both', which='major', labelsize=20)

    for ax1 in ax.flat:
        ax1.set(xlabel='x', ylabel='z')
        ax1.xaxis.label.set_size(20)
        ax1.yaxis.label.set_size(20)

    for ax1 in ax.flat:
        ax1.label_outer()

    plt.show()

if __name__ == "__main__":
    set_printoptions(precision=16)
    num_processes = os.cpu_count()
    too_deep.terminal = True
    too_far.terminal = True
    plot_plane = "xz"
    quad = .1
    mm = 1
    #generate_date()
    plotting_data()

