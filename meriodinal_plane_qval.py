import matplotlib.pyplot as plt
from initialconditions import lin_comb, space_light_timelike, check_tetrad, vartheta_raster
from metric import q_metric
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy, set_printoptions
from scipy.integrate import solve_ivp, odeint
from plotting import plot_data_3d
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp


def too_deep(t, y, *args):
    return y[1] - 2.5


def too_far(t,y,*args):
    return 10 - abs(y[1])


def iterate_calculation(iter_list, qq=0):
    mm = 1
    falls_in = 1
    falling_list = []
    turning_point = None
    trajectories_here = []
    aax = plt.axes()
    for ii, dist in enumerate(iter_list):
        if ii%50==0:
            print(ii)
        init_ort = array([0.0, dist, pi / 2, 0.0])
        init_three_velocity_list = check_tetrad(init_ort, qq, mm)
        init_three_velocity = init_three_velocity_list[0]
        u_t = space_light_timelike(init_ort, init_three_velocity, [qq, mm])
        init_4_vel = [u_t, *init_three_velocity]
        e = (1 - 2 * mm / dist) ** (1 + qq) * u_t  # Energy, constant of motion
        angular_mom = (1 - 2 * mm / dist) ** (-qq) * dist ** 2 * sin(init_ort[2]) * init_4_vel[3]  # constant of motion
        u_t, u_r, u_theta, u_phi = init_4_vel
        tt, rr1, theta_1, phi1 = init_ort
        sol = solve_ivp(quadrupol_ivp,
                        #schwarzschild_ivp,
                        t_span=[0, 20],
                        t_eval=linspace(0, 20, 2000),
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mm, qq, e, angular_mom),
                        # y0=array([tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi]),
                        vectorized=True,
                        method='DOP853',
                        atol=1e-12,
                        rtol=1e-12,
                        events=too_deep
                        )
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])
        trajectories_here.append(result_spherical)
        plot_data_3d(result_spherical, aax, r_anzeige=dist + 0.5,
                     plot_plane=plot_plane, axins=axins)

        if sol.y[1][-1] < dist:
            falls_in = 1
            falling_list.append(falls_in)
        else:
            if falls_in == 1 and turning_point is None:
                turning_point = ii
            falls_in = 0
            falling_list.append(falls_in)
    plt.show()
    #plt.close()

    return turning_point, falling_list, trajectories_here



if __name__ == "__main__":
    set_printoptions(precision=15)
    too_deep.terminal = True
    too_far.terminal = False
    plot_plane = "xz"
    r_values = []
    if plot_plane == "3d":
        aax = plt.axes(projection='3d')
        axins = None
    else:
        aax = plt.axes()
        axins = None

    plot_this_shit = []
    for qq in linspace(.0, 1, 20):
        print(f"qq bei {qq}")
        init_list = linspace(3+2*qq-0.1, 3+2*qq + 1, 20)

        index_thing, list_this, result = iterate_calculation(init_list, qq)
        print(index_thing)
        if 0 in list_this and 1 not in list_this:
            print("warnung alles nullen")
        if 1 in list_this and 0 not in list_this:
            print("warnung alles einsen")
        upper_thing = init_list[index_thing]
        lower_thing = init_list[index_thing-1]

        init_list = linspace(lower_thing, upper_thing, 20)
        index_thing, list_this, result = iterate_calculation(init_list, qq)
        print(index_thing)
        if 0 in list_this and 1 not in list_this:
            print("warnung alles nullen")
        if 1 in list_this and 0 not in list_this:
            print("warnung alles einsen")
        upper_thing = init_list[index_thing]
        lower_thing = init_list[index_thing - 1]
        endradius = 0.5 * (lower_thing + upper_thing)
        print(endradius)
        plot_this_shit.append(endradius)

    plt.plot(linspace(0,1,20), plot_this_shit, "b+")
    plt.show()
