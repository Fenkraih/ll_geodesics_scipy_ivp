from numpy import array, linspace, vstack
from scipy.integrate import solve_ivp, odeint
from core import quadrupol_odeint, quadrupol_ivp, schwarzschild_ivp, schwarzschild_odeint


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
                        args=(mass, quad, energy, lz),
                        vectorized=True,
                        method='RK45',
                        atol=1e-5,
                        rtol=1e-5,
                        dense_output=True
                        )
        result_spherical = vstack([sol.y[1], sol.y[3], sol.y[5]])

    return result_spherical