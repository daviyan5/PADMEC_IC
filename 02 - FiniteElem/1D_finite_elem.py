# Aplication of the Finite Element Method for 1D heat equation
# du/dt - alpha * d^2u/dx^2 = 0, 0 < x < 1
# u(x, 0) = u0(x)
# u0(0) = u0(1) = 0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation



def main():
    nx = 10
    nt = 100
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    alpha = 1.0
    dt = 0.01
    # Initial conditions
    u = np.zeros((nt, nx))
    u0 = np.zeros(nx)



if __name__ == "__main__":
    main()

