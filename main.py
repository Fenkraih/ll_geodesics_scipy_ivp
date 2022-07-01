import matplotlib.pyplot as plt
from initialconditions import lin_comb, space_light_timelike
from metric import q_metric
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy
from scipy.integrate import solve_ivp, odeint
from differential_equations import quadrupol_odeint, schwarzschild_odeint, quadrupol_ivp, schwarzschild_ivp


def init_params_2(distance):
    mass = 1
    quad = 0
    grid_steps = 1
    init_theta = 1 * pi / 2
    init_phi = 1 * pi / 2

    init_pos = [0, distance, pi / 2, pi]
    g = q_metric(init_pos, [quad, mass])

    sphere = [1.0, init_theta, init_phi]                                    # define initial momentum on observer sphere
    sph_karth = [sphere[0] * cos(sphere[1]) * sin(sphere[2]),               # define initial momentum in karthesian
                 sphere[0] * sin(sphere[1]) * sin(sphere[2]),               # to prepare for multiplication with tetrad
                 sphere[0] * cos(sphere[2])]                                # new vector is again in spherical coords
    ax, ay, az = sph_karth
    local_coord = array([ax * 1/sqrt(g[1]), ay * 1/sqrt(g[2]), az * 1/sqrt(g[3])])
    # multiplication with 1/sqrt(g) for local representation

    init_three_velocity = local_coord
    u_t = space_light_timelike(init_pos, init_three_velocity, [quad, mass])
    init_four_vel = [u_t, *init_three_velocity]
    e = (1 - 2 * mass / distance) ** (1 + quad) * u_t                       # Energy, constant of motion
    angular_mom = (1 - 2 * mass / distance) ** (-quad) * distance ** 2 * sin(init_pos[2]) * init_four_vel[3]    # constant of motion
    return mass, quad, e, angular_mom, init_four_vel, init_pos


def too_deep(t, y):
    return y[1] - 2.5


def check_tetrad(init_pos, quad, mass):
    g = q_metric(init_pos, [quad, mass])
    triad = [[0, -1 * (1 / sqrt(g[2])), 0],
                [0, 0, 1 * (1 / sqrt(g[3]))],
                [-1 * (1 / sqrt(g[1])), 0, 0]
                ]
    return triad


def plot_data(result_sp, ax):
    r = result_sp[0]
    theta = result_sp[1]
    phi = result_sp[2]
    x = r * cos(phi) * sin(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(theta)
    ax.plot3D(x, y, z, "b-")
    r_anzeige = 12
    ax.set_xlim([-r_anzeige, r_anzeige])
    ax.set_ylim([-r_anzeige, r_anzeige])
    ax.set_zlim([-r_anzeige, r_anzeige])


def solve_this(metric: str, solver: str, init_pos: array, init_four_vel: array, t_span, mass, quad, energy, lz):
    """
    Method to try out other solvers
    :param metric:
    :param solver:
    :param init_pos:
    :param init_four_vel:
    :param t_span:
    :param mass:
    :param quad:
    :param energy:
    :param lz:
    :return:
    """
    u_t, u_r, u_theta, u_phi = init_four_vel
    tt, rr1, theta_1, phi1 = init_pos
    if metric == "quadrupol" and solver == "ivp":
        sol = solve_ivp(quadrupol_ivp,
                        t_span=t_span,
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mass, quad, energy, lz),
                        vectorized=True,
                        method='RK45',
                        atol=1e-7,
                        rtol=1e-7,
                        dense_output=True
                        )
        print(sol)
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])

    elif metric == "quadrupol" and solver == "odeint":
        t_span = linspace(t_span[0], t_span[1], 200)
        sol = odeint(quadrupol_odeint,
                     y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                     t=t_span,
                     args=(mass, quad, energy, lz),
                     atol=1e-7,
                     rtol=1e-7
                     )
        result_spherical = vstack([sol[:, 1], sol[:, 3], sol[:, 5]])

    elif metric == "schwarzschild" and solver == "odeint":
        t_span = linspace(t_span[0], t_span[1], 200)
        sol = odeint(schwarzschild_odeint,
                     y0=[tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi],
                     t=t_span,
                     atol=1e-4,
                     rtol=1e-4
                     )
        result_spherical = vstack([sol[:, 2], sol[:, 4], sol[:, 6]])

    elif metric == "schwarzschild" and solver == "ivp":
        sol = solve_ivp(schwarzschild_ivp,
                        t_span=t_span,
                        y0=[tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi],
                        atol=1e-7,
                        rtol=1e-7,
                        dense_output=True,
                        events=too_deep)
        result_spherical = vstack([sol.y[2], sol.y[4], sol.y[6]])

    else:
        sol = solve_ivp(quadrupol_ivp,
                        t_span=t_span,
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mm, qq, e, angular_mom),
                        vectorized=True,
                        method='RK45',
                        atol=1e-5,
                        rtol=1e-5,
                        dense_output=True
                        )
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])

    return result_spherical


if __name__ == "__main__":
    dist = 10
    init_ort = [0, dist, pi/2, 0]          # for some reason r phi theta
    mm = 1
    qq = 0
    grid_steps = 1
    lw_theta, hg_theta, lw_phi, hg_phi = 90, 90, 30, 60
    angles = [lw_theta, hg_theta, lw_phi, hg_phi]
    g = q_metric(init_ort, [qq, mm])
    too_deep.terminal = True
    aax = plt.axes(projection='3d')
    init_three_velocity_list = lin_comb(init_ort, qq, mm, angles, grid_steps)
    init_three_velocity_list = check_tetrad(init_ort, qq, mm)
    for init_three_velocity in init_three_velocity_list:
        u_t = space_light_timelike(init_ort, init_three_velocity, [qq, mm])
        init_4_vel = [u_t, *init_three_velocity]
        e = (1 - 2 * mm / dist) ** (1 + qq) * u_t  # Energy, constant of motion
        angular_mom = (1 - 2 * mm / dist) ** (-qq) * dist ** 2 * sin(init_ort[2]) * init_4_vel[3] # constant of motion
        u_t, u_r, u_theta, u_phi = init_4_vel
        tt, rr1, theta_1, phi1 = init_ort
        sol = solve_ivp(quadrupol_ivp,
                        t_span=[0,10],
                        y0=array([tt, rr1, u_r, theta_1, u_theta, phi1]),
                        args=(mm, qq, e, angular_mom),
                        vectorized=True,
                        method='RK45',
                        atol=1e-7,
                        rtol=1e-7,
                        dense_output=True
                        )
        print(sol)
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])
        plot_data(result_spherical, aax)
        break
    plt.show()

