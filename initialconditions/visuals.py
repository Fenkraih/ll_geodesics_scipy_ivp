"""
visuals for the initial condition stuff
"""
import matplotlib.pyplot as plt
from numpy import sqrt, pi, cos, sin, size, linspace, outer, ones, array, arccos, arctan2, arcsin
from metric import q_metric


def vartheta_raster(init_three_velocity_list, g):
    """
    vartheta = 0 means to the left
    pi/2 means straight upwards
    :param init_three_velocity_list:
    :return:
    """
    for var_theta in linspace(.9*pi/2,1.1*pi/2,100):
        init_three_velocity_list.append([1 * cos(var_theta) * (1 / sqrt(g[1])),
                                        -1 * sin(var_theta) * (1 / sqrt(g[2])),
                                        0])
    return init_three_velocity_list


def check_tetrad(init_pos, quad, mass):
    g = q_metric(init_pos, [quad, mass])
    triad = [[0, -1 * (1 / sqrt(g[2])), 0],
                [0, 0, 1 * (1 / sqrt(g[3]))],
                [-1 * (1 / sqrt(g[1])), 0, 0]
                ]
    return triad


def angle_calc(mass, quad_param, distance, grid_steps):
    if quad_param == -0.5:
        syng_angle = arcsin(sqrt(mass ** 2 * (2 * quad_param + 3) ** 2 * (1 / distance ** 2))) * 180 * pi ** (-1)
    else:
        syng_angle = arcsin(sqrt(mass ** 2 * (2 * quad_param + 3) ** 2 * (1 / distance ** 2) * (
                    (2 * quad_param + 3) * (distance - 2 * mass) / ((2 * quad_param + 1) * distance)) ** (
                                             2 * quad_param + 1))) * 180 * pi ** (-1)
    schwarzschild_angle = arcsin(sqrt((27 * (2 * mass) ** 2 * (distance - 2 * mass)) / (4 * distance ** 3))) * 180 * (
        pi) ** (-1)
    print(f"Outer shadow angle in equatorial plane (q= {quad_param})-metric: {syng_angle}")
    angle_intervall = .0015
    lower_phi_angle = 90 - syng_angle - angle_intervall
    high_phi_angle = 90 - syng_angle + angle_intervall
    angle_steps = (lower_phi_angle - high_phi_angle) / grid_steps
    return lower_phi_angle, high_phi_angle, angle_steps


def shadow_angle_calc(lower_phi_angle, light_or_dark, angle_steps):
    shadow_angle = lower_phi_angle
    for number in light_or_dark:
        if number == 1:
            shadow_angle += angle_steps
    shadow_angle += 0.5 * angle_steps
    shadow_angle = 90 - shadow_angle
    print(f"Numerical value of shadow angle: {shadow_angle}")
    return shadow_angle


def eq_plane_grid(steps):
    karth_plane = []
    for kk in range(steps):
        karth_plane.append([10, 0, (-0.1 + 0.2 * kk / steps)])

    return karth_plane


def example_angle_selection(forward_backward):
    if forward_backward == "forward":
        angles = [pi / 2, pi / 2, 250 / 360 * 2 * pi, 290 / 360 * 2 * pi]  # sample for phi raster mit const theta
    else:
        angles = [pi / 2, pi / 2, 78.315 / 360 * 2 * pi, 78.318 / 360 * 2 * pi]


def sph_make(ax):
    """
    visual to print a 3d sphere on a plt plot
    :param ax:
    :return:
    """
    coefs_1 = (0.8, 0.8, 0.8)
    rx_1, ry_1, rz_1 = 1 / sqrt(coefs_1)
    rz_1 = 1 / (rx_1 * ry_1)

    # Make data
    u_space = linspace(0, 2 * pi, 100)
    v_space = linspace(0, pi, 100)

    x_space = rx_1 * outer(cos(u_space), sin(v_space))
    y_space = ry_1 * outer(sin(u_space), sin(v_space))
    z_space = rz_1 * outer(ones(size(u_space)), cos(v_space))

    # Plot the surface
    surf_1 = ax.plot_surface(x_space, y_space, z_space, alpha=0.8, label='Sphäre')
    surf_1._edgecolors2d = surf_1._edgecolor3d
    surf_1._facecolors2d = surf_1._facecolor3d


def square_to_polar_grid_v2(steps, distance, width):
    karth_quadrat = []
    karth_kugel = []
    sph_kugel = []

    for kk in range(steps):
        karth_quadrat.append([-distance, (-1 * width + 2 * width * kk / steps), -1 * width])
        for ll in range(steps):
            karth_quadrat.append(
                [-distance, (-1 * width + 2 * width * kk / steps), (-1 * width + 2 * width * ll / steps)])

    for elements in karth_quadrat:
        length = sqrt(elements[0] ** 2 + elements[1] ** 2 + elements[2] ** 2)
        karth_kugel.append(array(elements) / length * 0.1)

    if False:
        for elements in karth_kugel:
            xx = elements[0]
            yy = elements[1]
            zz = elements[2]
            # spiegelung an x=-0.1
            xx = -xx - 0.2 - 29.9
            rr = sqrt(xx ** 2 + yy ** 2 + zz ** 2)
            theta = arccos(zz / rr)
            phi = arctan2(yy, xx)
            sph_kugel.append([rr, theta, phi])

    return karth_quadrat, sph_kugel, karth_kugel


def square_to_polar_grid(steps):
    karth_quadrat = []
    karth_kugel = []
    sph_kugel = []

    if False:
        for kk in range(steps + 1):
            karth_quadrat.append([10, (-0.1 + 0.2 * kk / steps), (-0.1)])
            for ll in range(steps):
                karth_quadrat.append([10, (-0.1 + 0.2 * kk / steps), (-0.1 + 0.2 * ll / steps)])

    if True:
        for kk in range(steps + 1):
            karth_quadrat.append([10, 0, (-0.1 + 0.2 * kk / steps)])

    for elements in karth_quadrat:
        length = sqrt(elements[0] ** 2 + elements[1] ** 2 + elements[2] ** 2)
        karth_kugel.append(array(elements) / length * 0.1)

    for elements in karth_kugel:
        xx = elements[0]
        yy = elements[1]
        zz = elements[2]
        # spiegelung an x=-0.1
        # xx = -xx - 0.2 - 29.9
        rr = sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        theta = arccos(zz / rr)
        phi = arctan2(yy, xx)
        sph_kugel.append([rr, theta, phi])

    return karth_quadrat, sph_kugel, karth_kugel


def plot_karth_multi_vel(karth_geschw_multi):
    r_anzeige = 0.2
    ax = plt.axes(projection='3d')
    ax.set_xlim([-r_anzeige, r_anzeige])
    ax.set_ylim([-r_anzeige, r_anzeige])
    ax.set_zlim([-r_anzeige, r_anzeige])
    for karth_geschw in karth_geschw_multi:
        x_coord = []
        y_coord = []
        z_coord = []
        for elements in karth_geschw:
            x_coord.append(elements[0])
            y_coord.append(elements[1])
            z_coord.append(elements[2])
        ax.plot3D(x_coord, y_coord, z_coord, "+")
    plt.show()


def sph_to_karth(geschw):
    geschw_karth = []
    for elements in geschw:
        xx = elements[0] * cos(elements[2]) * sin(elements[1])
        yy = elements[0] * sin(elements[2]) * sin(elements[1])
        zz = elements[0] * cos(elements[1])
        geschw_karth.append([xx, yy, zz])
    return geschw_karth


def square_to_sphere_tester():
    karth_quadrat, sph_kugel, karth_kugel = square_to_polar_grid(40)
    kugel = sph_to_karth(sph_kugel)
    plot_karth_multi_vel([karth_quadrat, kugel])


if __name__ == '__main__':
    square_to_sphere_tester()
