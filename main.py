import matplotlib.pyplot as plt
from initialconditions import lin_comb, space_light_timelike
from metric import q_metric
from numpy import array, sin, cos, sum, pi, linspace, vstack, sqrt, tan, copy
from scipy.integrate import solve_ivp


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

    # local_coord = array([ax, ay, az])
    local_coord = array([ax * 1/sqrt(g[1]), ay * 1/sqrt(g[2]), az * 1/sqrt(g[3])])
    # multiplication with 1/sqrt(g) for local representation

    init_three_velocity = local_coord
    u_t = space_light_timelike(init_pos, init_three_velocity, [quad, mass])
    init_four_vel = [u_t, *init_three_velocity]
    e = (1 - 2 * mass / distance) ** (1 + quad) * u_t                       # Energy, constant of motion
    angular_mom = (1 - 2 * mass / distance) ** (-quad) * distance ** 2 * sin(init_pos[2]) * init_four_vel[3]    # constant of motion
    return mass, quad, e, angular_mom, init_four_vel, init_pos


def rsdgl(s: float, y_prev: array, q, m, energy, angular_momentum):
    t1, r1, theta1, phi, r2, theta2 = y_prev

    alpha = 1 - (2 * m / r1)
    beta = 1 + (sin(theta1) ** 2 * m ** 2 / (r1 ** 2 - 2 * m * r1))

    t_dot = alpha ** (-(1 + q)) * energy
    r1_dot = r2
    r2_a = -(1 + q) * m / (r1 ** 2) * beta ** (q * (q + 2)) * energy ** 2
    r2_b = (2 * (q + 1) * m - (1 + q) * m / (r1 ** 2 - 2 * m * r1) - (
            q * (2 + q) * (r1 - m) * m ** 2 * sin(theta1) ** 2) / (
                    (r1 ** 2 - 2 * m * r1 + m ** 2 * sin(theta1) ** 2) * (r1 ** 2 - 2 * m * r1))) * r2 ** 2
    r2_c = ((2 * m ** 2 * q * (2 + q) * sin(theta1) * cos(theta1)) / (
            r1 ** 2 - 2 * m * r1 + m ** 2 * sin(theta1) ** 2)) * r2 * theta2
    r2_d = ((q * (2 + q) * (r1 - m) * m ** 2 * sin(theta1) ** 2) / (
            r1 ** 2 - 2 * m * r1 + m ** 2 * sin(theta1) ** 2) + r1 - 2 * m - q * m) * theta2 ** 2
    r2_e = (1 / r1 - q * m / (r1 ** 2 - 2 * m * r1)) * alpha ** (2 * q + 1) * beta ** (
            q + (q + 2)) * angular_momentum ** 2 * (1 / (r1 ** 2 * sin(theta1) ** 2))

    r2_dot = r2_a + r2_b + r2_c + r2_d + r2_e

    theta1_dot = theta2

    theta2_a = ((2 * m * q) / (r1 ** 2 - 2 * m * r1) - 2 / r1 - (
            (2 * m ** 2 * q * (2 + q)) / (r1 ** 2 - 2 * m * r1 + m ** 2 * sin(theta1) ** 2)) * (
                        sin(theta1) ** 2 * (r1 - m)) / (r1 ** 2 - 2 * m * r1))
    theta2_b = - (
            (q * (2 + q) * m ** 2 * sin(theta1) * cos(theta1)) / (r1 ** 2 - 2 * m * r1 + m ** 2 * sin(theta1) ** 2))
    theta2_c = alpha ** (2 * q) * beta ** (q * (2 + q)) * (sin(theta1) * cos(theta1) * angular_momentum ** 2) / (
            r1 ** 4 * sin(theta1) ** 4)
    theta2_dot = theta2_a * r2 * theta2 + theta2_b * ((r2 ** 2) / (r1 ** 2 - 2 * m * r1) - theta2 ** 2) + theta2_c

    phi_dot = alpha ** q * (angular_momentum / (r1 ** 2 * sin(theta1) ** 2))

    y_step = array([t_dot, r1_dot, r2_dot, theta1_dot, theta2_dot, phi_dot])
    return y_step


def too_deep(t, y):
    return y[1] - 2.5


def schwarzschild(s, y, m=1):
    t, td, r, rd, th, thd, phi, phid = y

    # for convenience:
    alpha = 1 - 2 * m / r
    f = [td,
         - 2 * m * td * rd / (r ** 2 * alpha),
         rd,
         -m * td ** 2 / r ** 2 * alpha + m / r ** 2 * rd ** 2 / alpha + r * alpha * thd ** 2 + alpha * r * sin(
             th) ** 2 * phid ** 2,
         thd,
         -2 / r * rd * thd + cos(th) * sin(th) * phid ** 2,
         phid,
         - 2 / r * rd * phid - 2 / tan(th) * thd * phid]
    return f


def plot_data(result_sp, ax):
    r = result_sp[0]
    theta = result_sp[1]
    phi = result_sp[2]
    x = r * cos(theta) * sin(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(phi)
    ax.plot3D(x, y, z, "b-")
    # ax.plot3D(y, x, z, "b-")
    r_anzeige = 10
    ax.set_xlim([-r_anzeige, r_anzeige])
    ax.set_ylim([-r_anzeige, r_anzeige])
    ax.set_zlim([-r_anzeige, r_anzeige])


if __name__ == "__main__":
    dist = 10
    init_pos = [0, dist, pi, pi/2]          # for some reason r phi theta
    mm = 1
    qq = 0
    grid_steps = 1
    lw_theta = 90
    hg_theta = 90
    lw_phi = 30
    hg_phi = 60
    angles = [lw_theta, hg_theta, lw_phi, hg_phi]
    g = q_metric(init_pos, [qq, mm])
    too_deep.terminal = True

    aax = plt.axes(projection='3d')
    init_three_velocity_list = lin_comb(init_pos, qq, mm, angles, grid_steps)
    if True:
        init_three_velocity_list = [[0, -1 * (1 / sqrt(g[2])), 0],
                                    [0, 0, -1 * (1 / sqrt(g[3]))],
                                    [-1 * (1 / sqrt(g[1])), 0, 0]
                                    ]
    for init_three_velocity in init_three_velocity_list:
        u_t = space_light_timelike(init_pos, init_three_velocity, [qq, mm])
        init_four_vel = [u_t, *init_three_velocity]
        e = (1 - 2 * mm / dist) ** (1 + qq) * u_t  # Energy, constant of motion
        angular_mom = (1 - 2 * mm / dist) ** (-qq) * dist ** 2 * sin(init_pos[2]) * init_four_vel[3]  # constant of motion
        u_t, u_r, u_theta, u_phi = init_four_vel
        tt, rr1, theta_1, phi1 = init_pos
        if False:
            sol = solve_ivp(rsdgl,
                            t_span=[0, 2],
                            y0=array([tt, rr1, theta_1, phi1, u_r, u_theta]),
                            args=(mm, qq, e, angular_mom),
                            atol=1e-7,
                            rtol=1e-7,
                            dense_output=True
                            )
            result_spherical = vstack([sol.y[1], sol.y[2], sol.y[3]])
            plot_data(result_spherical, aax)

        if True:
            sol = solve_ivp(schwarzschild,
                            t_span=[0, 15],
                            y0=[tt, u_t, rr1, u_r, theta_1, u_theta, phi1, u_phi],
                            atol=1e-7,
                            rtol=1e-7,
                            dense_output=True,
                            events=too_deep)
            result_spherical = vstack([sol.y[2], sol.y[4], sol.y[6]])
            plot_data(result_spherical, aax)
    plt.show()
