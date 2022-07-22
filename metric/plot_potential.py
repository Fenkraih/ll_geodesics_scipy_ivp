import matplotlib.pyplot as plt
import numpy as np


def main(q,):
    m = 1
    r = np.linspace(3.5, 6.5, 100)
    b = m*(2*q+3)*np.sqrt(((2*q+3)/(2*q+1))**(2*q+1))
    alpha = 1-2*m/r
    beta = 1+m**2/(r**2-2*m*r)
    #x = (2*q+3)*m
    #b = np.sqrt((x**4/(x**2 - 2*m*x))*(1-2*m/x)**(-2*q))
    v_b = -.5 * (1+m**2/(r**2-2*m*r))**(q*(2+q)) * ((r**4/b**2)*(1-2*m/r)**(-2*q) - (r**2 - 2*m*r))
    v_b_prime_1 = -q*(2+q)*((r-m)*m**2)/(r**2 - 2*m*r)**2 * beta ** (q*(2+q)-1) * (r**4/b**2 * alpha**(-2*q) - (r**2 - 2 *m *r))
    v_b_prime_2 = - beta**(q*(2+q)) * (2*r**3 / b**2 * alpha ** (-2*q) - 2*q*m*r**2/b**2 * alpha**(-2*q-1) - (r-m))
    v_b_prime = v_b_prime_1 + v_b_prime_2
    ax = plt.axes()
    ax.plot(r,v_b, label="V(r)", linewidth=2)
    ax.plot(r,v_b_prime, label="V(r)'", linewidth=2)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=20)
    plt.xlabel("r")
    ax.xaxis.label.set_size(20)
    plt.ylim([-3,3])
    plt.show()


if __name__ == "__main__":
    main(1)