import matplotlib.pyplot as plt
import os
from initialconditions import lin_comb, space_light_timelike, check_tetrad, vartheta_raster
from metric import q_metric
import multiprocessing as mp
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy, set_printoptions, save
from scipy.integrate import solve_ivp, odeint
from plotting import plot_data_3d
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp


def too_deep(t, y, *args):
    return y[1] - 2.5


def too_far(t, y, *args):
    return 10 - abs(y[1])


# 3.300056116722783
# 3.3000005668355836

def iterate_calculation(iter_list, qq_2, mm_2):
    falls_in = 1
    falling_list = []
    turning_point = None
    trajectories_here = []
    #aax_1 = plt.axes()
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
                        t_span=[0, 19],
                        t_eval=linspace(0, 19, 2000),
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mm_2, qq_2, e, angular_mom),
                        # y0=array([tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi]),
                        vectorized=True,
                        method='DOP853',
                        atol=1e-12,
                        rtol=1e-12,
                        max_step=1e-3,
                        events=too_deep
                        )
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])
        trajectories_here.append(result_spherical)
        #plot_data_3d(result_spherical, aax_1, r_anzeige=dist + 0.5,
        #             plot_plane=plot_plane, axins=axins)

        if sol.y[1][-1] < dist:
            falls_in = 1
            falling_list.append(falls_in)
        else:
            if falls_in == 1 and turning_point is None:
                turning_point = ii_1
            falls_in = 0
            falling_list.append(falls_in)
    # print(falling_list)
    # plt.show()
    # plt.close()

    return turning_point, falling_list, trajectories_here


def return_call(initial_list, quad_param, mass):
    index_thing, list_this, result = iterate_calculation(initial_list, quad_param, mass)
    if 0 in list_this and 1 not in list_this:
        print("warnung alles nullen")
    if 1 in list_this and 0 not in list_this:
        print("warnung alles einsen")
    upper_thing = initial_list[index_thing]
    lower_thing = initial_list[index_thing - 1]
    return upper_thing, lower_thing


def iter_loop_this(qq_1, mm_1, ii_1, jj_1):
    lower_thing, upper_thing = 3*mm_1 + 2 * qq_1 - 0.1, 3*mm_1 + 2 * qq_1 + 1
    for kk in range(10):
        print(f"{qq_1} loop {kk}")
        init_list = linspace(lower_thing, upper_thing, 8)
        upper_thing, lower_thing = return_call(init_list, qq_1, mm_1)

    endradius = 0.5 * (lower_thing + upper_thing)
    print(f"qq:{qq_1}:{endradius}")
    if not os.path.isdir(f"/home/altin/ll_geod_scipy_ivp/data/m{ii_1}/"):
        os.mkdir(f"/home/altin/ll_geod_scipy_ivp/data/m{ii_1}/")
    save(f"/home/altin/ll_geod_scipy_ivp/data/m{ii_1}/q{jj_1}", endradius)


if __name__ == "__main__":
    set_printoptions(precision=15)
    too_deep.terminal = True
    too_far.terminal = False
    num_processes = os.cpu_count()
    plot_plane = "xz"
    r_values = []
    if plot_plane == "3d":
        aax = plt.axes(projection='3d')
        axins = None
    else:
        aax = plt.axes()
        axins = None
    sub_procs = []
    ii = 0
    mm = 1

    if True:
        for jj, qq in enumerate(linspace(0, 1, 100)):
            p = mp.Process(target=iter_loop_this,
                           args=(qq,mm, ii, jj))
            sub_procs.append(p)
            if len(sub_procs)==4:
                for pp in sub_procs:
                    pp.start()
                for pp in sub_procs:
                    pp.join()
                for pp in sub_procs:
                    pp.terminate()
                sub_procs = []



# result m = 1 er iteration [2.999999433164416, 3.051914400423299, 3.103628707885562, 3.1551468902358755, 3.206474615830075, 3.2576186866951655, 3.308587038529318, 3.359389874373039, 3.4100305952398307, 3.460517136827863, 3.510850632808305, 3.5610356178658247, 3.6110754930139253, 3.66097252559494, 3.710730116622372, 3.7603516671097235, 3.809840578070496, 3.8592025178605267, 3.908439753822149, 3.9575568206400336, 4.006559386670017, 4.055449719254433, 4.10423348674912, 4.152915223838745, 4.201498331536811, 4.249988478199156, 4.298387931168114, 4.346703492470688, 4.394937429449214, 4.4430942767883606, 4.491179702843964, 4.539198242300696, 4.587153296172055, 4.635049399142713, 4.682891085897339, 4.730685158462937, 4.77843388418184, 4.826142931409887, 4.873817968502913, 4.921462396474421, 4.9690807500090814, 5.016679831133896, 5.064264174533536, 5.111837181221501, 5.159404519553631, 5.2069707242145915, 5.254540329889055, 5.302119004932858, 5.349709016688335, 5.397314899840153]
