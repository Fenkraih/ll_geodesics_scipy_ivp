import matplotlib.pyplot as plt
from initialconditions import lin_comb, space_light_timelike, check_tetrad
from metric import q_metric
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy
from scipy.integrate import solve_ivp, odeint
from plotting import plot_data_3d
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp


def too_deep(t, y, *args):
    return y[1] - 2.5


if __name__ == "__main__":
    dist = 3.002
    init_ort = [0, dist, pi / 2, 0]  # for some reason r phi theta
    mm = 1
    qq = 0.001
    grid_steps = 1
    lw_theta, hg_theta, lw_phi, hg_phi = 90, 90, 30, 60
    angles = [lw_theta, hg_theta, lw_phi, hg_phi]
    g = q_metric(init_ort, [qq, mm])
    too_deep.terminal = True
    plot_plane = "3d"
    if plot_plane == "3d":
        aax = plt.axes(projection='3d')
        axins = None
    else:
        aax = plt.axes()
        axins = aax.inset_axes([0.7, 0.7, 0.25, 0.25])
        x1, x2, y1, y2 = dist-.1, dist+.1, -5e-7, 5e-7
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        aax.indicate_inset_zoom(axins, edgecolor="black")

    init_three_velocity_list = check_tetrad(init_ort, qq, mm)
    for init_three_velocity in init_three_velocity_list:
        u_t = space_light_timelike(init_ort, init_three_velocity, [qq, mm])
        init_4_vel = [u_t, *init_three_velocity]
        e = (1 - 2 * mm / dist) ** (1 + qq) * u_t  # Energy, constant of motion
        angular_mom = (1 - 2 * mm / dist) ** (-qq) * dist ** 2 * sin(init_ort[2]) * init_4_vel[3]  # constant of motion
        u_t, u_r, u_theta, u_phi = init_4_vel
        tt, rr1, theta_1, phi1 = init_ort
        sol = solve_ivp(quadrupol_ivp,
                        t_span=[0, 50],
                        t_eval=linspace(0, 50, 200),
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mm, qq, e, angular_mom),
                        vectorized=True,
                        method='RK45',
                        atol=1e-10,
                        rtol=1e-10,
                        dense_output=True,
                        events=too_deep
                        )
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])

        plot_data_3d(result_spherical, aax, r_anzeige=dist + 0.5,
                     black_geodesic=sol.message == "A termination event occurred.",
                     plot_plane=plot_plane, axins=axins)

    plt.show()
