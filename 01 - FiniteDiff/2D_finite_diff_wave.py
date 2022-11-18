# Solving the 2D wave equation
# d^2u/dt^2 - c^2 * (d^2u/dx^2 + d^2u/dy^2) = 0, 0 < x < 1, 0 < y < 1
# u(x, y, 0) = u0(x, y)
# u(x, y, dt) = u(x, y, 0)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


# Finite difference method
# d^2u/dt² = (u[t+1][dx][dy] - 2u[t][dx][dy] + u[t-1][dx][dy]) / dt²
# d^2u/dx² = (u[t][dx+1][dy] - 2u[t][dx][dy] + u[t][dx-1][dy]) / dx²
# d^2u/dy² = (u[t][dx][dy+1] - 2u[t][dx][dy] + u[t][dx][dy-1]) / dy²
# d^2u/dt² = c²(d^2u/dx² + d^2u/dy²)
# u[t+1][dx][dy] = 2u[t][dx][dy] - u[t-1][dx][dy] + c²(dt²/dx²)(u[t][dx+1][dy] - 2u[t][dx][dy] + u[t][dx-1][dy]) + 
#                  c²(dt²/dy²)(u[t][dx][dy+1] - 2u[t][dx][dy] + u[t][dx][dy-1])
# c²(dt²/dx²) = ctx
# c²(dt²/dy²) = cty
# u[t+1] = 2u[t] - u[t-1] + ctx * (np.roll(u[t], -1, axis=0) - 2*u[t] + np.roll(u[t], 1, axis=0)) +
#          cty * (np.roll(u[t], -1, axis=1) - 2*u[t] + np.roll(u[t], 1, axis=1))
# u[t+1] = u[t] * (2 - 2ctx - 2cty) - u[t-1] + ctx * (np.roll(u[t], -1, axis=0) + np.roll(u[t], 1, axis=0)) +
#          cty * (np.roll(u[t], -1, axis=1) + np.roll(u[t], 1, axis=1))

def solve_2d_mt_get_next(ctx, cty, u0, u00):
    return (u0 * (2 - 2*ctx - 2*cty) - u00 + ctx * (np.roll(u0, -1, axis=0) + np.roll(u0, 1, axis=0)) +
            cty * (np.roll(u0, -1, axis=1) + np.roll(u0, 1, axis=1)))

def reset_boundary_conditions(u):
    u_top = 1
    u_left = 0
    u_bottom = 0
    u_right = 0
    nx = len(u)
    ny = len(u[0])
    u[(nx-1):, :] = u_top
    u[:, :1] = u_left
    u[:1, 1:] = u_bottom
    u[:, (ny-1):] = u_right
    return u

def solve_2d_mt(nt, dt, dx, dy, c, u):
    ctx = (c ** 2) * (dt ** 2) / (dx ** 2)
    cty = (c ** 2) * (dt ** 2) / (dy ** 2)

    for k in range(1, nt - 1, 1):
        u[k+1] = solve_2d_mt_get_next(ctx, cty, u[k], u[k-1])
        # Reset the boundary conditions
        u[k+1] = reset_boundary_conditions(u[k+1])
    return u

                
    return u   

def plotheatmap(u_k, k, dt):
    # Clear the current plot figure
    plt.clf()
    plt.title(f"Wave Amplitude at t = {k*dt:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap = plt.cm.coolwarm)
    plt.colorbar()
    
    return plt

def animate(k, u, dt):
        plotheatmap(u[k], k, dt)


def main():
    # Grid parameters
    nx = 200
    ny = 200
    nt = 1000
    lx = 10
    ly = 10
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    

    # Equation parameters   
    c = 1

    # Initial conditions
    u = np.empty((nt, nx, ny))
    u0 = np.load("../Random/image_array.npy")
    u0 = np.zeros((nx, ny))
    # Rotate 180 degrees
    u0 = np.rot90(u0, 2)
    u[0] = u0
    u[0] = reset_boundary_conditions(u[0])
    u[1] = u[0]

    # Time parameters
    dt = (dx * dy)/ (c * (dx + dy))
    print("Parameters: nx = ", nx, ", ny = ", ny, ", nt = ", nt, ", lx = ", lx, ", ly = ", ly, ", dx = ", dx, ", dy = ", dy, ", c = ", c, ", dt = ", dt)
    
    # Solve the equation
    u = solve_2d_mt(nt, dt, dx, dy, c, u)

    print("Done!")
    # Plot the solution
    anim = animation.FuncAnimation(plt.figure(), animate, fargs=[u, dt], interval=1, frames=nt, repeat=False)
    plt.show()
    #anim.save('heat.mp4', fps=24, extra_args=['-vcodec', 'libx264'])

if __name__ == "__main__":
    main()