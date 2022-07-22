from numpy import sin, cos
import matplotlib.pyplot as plt


def plot_data_3d(result_sp, ax, black_geodesic=None, r_anzeige=None, plot_plane="xz", axins=None):
    if black_geodesic == 1:
        black_geodesic = True
    r = result_sp[0]
    theta = result_sp[1]
    phi = result_sp[2]
    x = r * cos(phi) * sin(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(theta)
    if plot_plane not in ["3d", "xy", "xz", "yz"]:
        print("Plot plane key not supported")
    if black_geodesic:
        if plot_plane == "3d":
            ax.plot3D(x, y, z, "k-")
        if plot_plane == "xy":
            ax.plot(x, y, "k-")
            if axins is not None:
                axins.plot(x, y, "k-")
        if plot_plane == "xz":
            ax.plot(x, z, "k-")
            if axins is not None:
                axins.plot(x, z, "k-")
        if plot_plane == "yz":
            ax.plot(y, z, "k-")
            if axins is not None:
                axins.plot(x, z, "k-")
    else:
        if plot_plane == "3d":
            ax.plot3D(x, y, z, "b-")
        if plot_plane == "xy":
            ax.plot(x, y, "b-")
            if axins is not None:
                axins.plot(x, y, "b-")
        if plot_plane == "xz":
            ax.plot(x, z, "b-")
            if axins is not None:
                axins.plot(x, z, "b-")
        if plot_plane == "yz":
            ax.plot(y, z, "b-")
            if axins is not None:
                axins.plot(y, z, "b-")

    if r_anzeige is None:
        r_anzeige = 10
    if plot_plane == "3d":
        ax.set_xlim([-r_anzeige, r_anzeige])
        ax.set_ylim([-r_anzeige, r_anzeige])
        ax.set_zlim([-r_anzeige, r_anzeige])
    else:
        ax.set_xlim([-r_anzeige, r_anzeige])
        ax.set_ylim([-r_anzeige, r_anzeige])