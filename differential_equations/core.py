from numpy import copy, sin, cos, tan, array


def quadrupol_ivp(s: float, y_prev: array, m, q, energy, angular_momentum):
    t1, r1, r2,  theta1, theta2, phi = y_prev

    alpha = copy(1 - (2 * m / r1))
    beta = copy(1 + (sin(theta1) ** 2 * m ** 2 / (r1 ** 2 - 2 * m * r1)))
    gamma = copy((q*(q+2)*(r1-m)*m**2 * sin(theta1)**2)/(r1**2 - 2*m*r1 + m**2 * sin(theta1)**2))

    t_dot = alpha ** (-(1 + q)) * energy
    phi_dot = alpha ** q * (angular_momentum / (r1 ** 2 * sin(theta1) ** 2))

    r1_dot = r2
    r2_a = -(1 + q) * m / (r1 ** 2) * alpha**(2*q+1) * beta ** (q * (q + 2)) * t_dot**2
    r2_b = ((1+q)*m + gamma) * (r2 ** 2 / (r1**2-2*m*r1))
    r2_c = ((r1-2*m-q*m) - gamma) * theta2**2
    r2_d = 2*((q*(q+2)*m**2 * sin(theta1) * cos(theta1))/(r1**2 - 2*m*r1 + m**2 * sin(theta1)**2))*theta2*r2
    r2_e = (r1-2*m-q*m)*beta**(q*(2+q))*sin(theta1)**2*phi_dot**2
    r2_dot = r2_a + r2_b + r2_c + r2_d + r2_e

    theta1_dot = theta2
    theta2_a = - ((q*(q+2)*m**2 * sin(theta1) * cos(theta1))/(r1**2 - 2*m*r1 + m**2 * sin(theta1)**2)) * (r2**2 /(r1**2-2*m*r1) - theta2**2)
    theta2_b = beta**(q*(2+q)) * sin(theta1) * cos(theta1) * phi_dot**2
    theta2_c = ((2*q*m)/(r1**2-2*m*r1) - 2/r1 - (2*(q+2)*m**2)/(r1**2-2*m*r1 + m**2 * sin(theta1)**2) * ((r1-m)*sin(theta1)**2)/(r1**2-2*r1*m))*r2*theta2
    theta2_dot = theta2_a + theta2_b + theta2_c

    y_step = array([t_dot, r1_dot, r2_dot, theta1_dot, theta2_dot, phi_dot])
    return y_step


def quadrupol_odeint(y_prev: array, s: float, m, q, energy, angular_momentum):
    t1, r1, r2,  theta1, theta2, phi = y_prev
    alpha = copy(1 - (2 * m / r1))
    beta = copy(1 + (sin(theta1) ** 2 * m ** 2 / (r1 ** 2 - 2 * m * r1)))
    gamma = copy((q*(q+2)*(r1-m)*m**2 * sin(theta1)**2)/(r1**2 - 2*m*r1 + m**2 * sin(theta1)**2))

    t_dot = alpha ** (-(1 + q)) * energy
    phi_dot = alpha ** q * (angular_momentum / (r1 ** 2 * sin(theta1) ** 2))

    r1_dot = r2
    r2_a = -(1 + q) * m / (r1 ** 2) * alpha**(2*q+1) * beta ** (q * (q + 2)) * t_dot**2
    r2_b = ((1+q)*m + gamma) * (r2 ** 2 / (r1**2-2*m*r1))
    r2_c = ((r1-2*m-q*m) - gamma) * theta2**2
    r2_d = 2*((q*(q+2)*m**2 * sin(theta1) * cos(theta1))/(r1**2 - 2*m*r1 + m**2 * sin(theta1)**2))*theta2*r2
    r2_e = (r1-2*m-q*m)*beta**(q*(2+q))*sin(theta1)**2*phi_dot**2
    r2_dot = r2_a + r2_b + r2_c + r2_d + r2_e

    theta1_dot = theta2
    theta2_a = - ((q*(q+2)*m**2 * sin(theta1) * cos(theta1))/(r1**2 - 2*m*r1 + m**2 * sin(theta1)**2)) * (r2**2 /(r1**2-2*m*r1) - theta2**2)
    theta2_b = beta**(q*(2+q)) * sin(theta1) * cos(theta1) * phi_dot**2
    theta2_c = ((2*q*m)/(r1**2-2*m*r1) - 2/r1 - (2*(q+2)*m**2)/(r1**2-2*m*r1 + m**2 * sin(theta1)**2) * ((r1-m)*sin(theta1)**2)/(r1**2-2*r1*m))*r2*theta2
    theta2_dot = theta2_a + theta2_b + theta2_c

    y_step = array([t_dot, r1_dot, r2_dot, theta1_dot, theta2_dot, phi_dot])
    return y_step


def schwarzschild_odeint(y, s):    # für ODEINT  y, s für solve_ivp s,y
    m = 1
    t, td, r, rd, th, thd, phi, phid = y
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


def schwarzschild_ivp(s, y):    # für ODEINT  y, s für solve_ivp s,y
    m = 1
    t, td, r, rd, th, thd, phi, phid = y
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

