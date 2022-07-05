import matplotlib.pyplot as plt
from initialconditions import lin_comb, space_light_timelike, check_tetrad
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


def too_deep(t, y, *args):
    return y[1] - 2.5


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


if __name__ == "__main__":
    dist = 5
    init_ort = [0, dist, pi/2, 0]          # for some reason r phi theta
    mm = 1
    qq = 1
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
                        t_span=[0,50],
                        t_eval=linspace(0,50,200),
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
        plot_data(result_spherical, aax)
    plt.show()

