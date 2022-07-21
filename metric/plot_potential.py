import matplotlib.pyplot as plt
import numpy as np


def main(q,):
    m = 1
    r = np.linspace(3, 6, 100)
    b = m*(2*q+1)*np.sqrt(((2*q+3)/(2*q+1))**(2*q+1))
    x = (2*q+3)*m
    b = np.sqrt((x**4/(x**2 - 2*m*x))*(1-2*m/x)**(-2*q))
    v_b = -.5 * (1+m**2/(r**2-2*m*r))**(q*(2+q)) * ((r**4/b**2)*(1-2*m/r)**(-2*q) - (r**2 - 2*m*r))
    plt.plot(r,v_b)
    plt.ylabel("Vb(r)")
    plt.xlabel("r")
    plt.show()


if __name__ == "__main__":
    main(1)