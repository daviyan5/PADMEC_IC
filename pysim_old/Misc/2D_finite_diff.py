# Solving the 2D heat equation 
# du/dt - alpha * (d^2u/dx^2 + d^2u/dy^2) = 0, 0 < x < 1, 0 < y < 1
# u(x, y, 0) = u0(x, y)
# u0(0, y) = u0(1, y) = u0(x, 0) = 0
# u0(x, 1) = 0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Finite difference method
# u[t+1][dx][dy] = dt * alpha * ((u[t][dx+1][dy] - 2 * u[t][dx][dy] + u[t][dx-1][dy]) / dx² ) + 
#                                (u[t][dx][dy+1] - 2 * u[t][dx][dy] + u[t][dx][dy-1]) / dy² )) + u[t][dx][dy]
# alphat = alpha * dt
# alphatx = alphat / dx²
# alphaty = alphat / dy²
# u[t+1][dx][dy] = alphatx * (u[t][dx+1][dy] - 2 * u[t][dx][dy] + u[t][dx-1][dy]) +
#                  alphaty * (u[t][dx][dy+1] - 2 * u[t][dx][dy] + u[t][dx][dy-1]) + u[t][dx][dy]
# u[t+1] = alphatx * (np.roll(u[t], -1, axis=1) - 2 * u[t] + np.roll(u[t], 1, axis=1)) +
#          alphaty * (np.roll(u[t], -1, axis=0) - 2 * u[t] + np.roll(u[t], 1, axis=0)) + u[t]

def solve_2d_mt_get_next(alphat, alphatx, alphaty, u0):
    return (alphatx * (np.roll(u0, -1, axis=1) + np.roll(u0, 1, axis=1)) + 
            alphaty * (np.roll(u0, -1, axis=0) + np.roll(u0, 1, axis=0)) + u0 * (1 - 2 * alphatx - 2 * alphaty)) 

def reset_boundary_conditions(u):
    u_top = 100
    u_left = -2
    u_bottom = -2
    u_right = -2
    nx = len(u)
    ny = len(u[0])
    u[(nx-1):, :] = u_top
    u[:, :1] = u_left
    u[:1, 1:] = u_bottom
    u[:, (ny-1):] = u_right
    return u

def solve_2d_mt(nt, dt, dx, dy, alpha, u):
    alphat = alpha * dt
    alphatx = alphat / dx**2
    alphaty = alphat / dy**2

    for k in range(0, nt - 1, 1):
        u[k+1] = solve_2d_mt_get_next(alphat, alphatx, alphaty, u[k])
        # Reset the boundary conditions
        #u[k+1] = reset_boundary_conditions(u[k+1])
    return u


def solve_2d_fl(nt, dt,  x, y, alpha, u):
    alphat = alpha * dt
    for k in range(0, nt - 1, 1):
        for i in range(1, len(x) - 1):  
            alphatx = alphat / x[i]**2
            for j in range(1, len(y) - 1):
                print("Solving for u[", k, "][", i, "][", j, "]")
                alphaty = alphat / y[j]**2
                u[k+1][i][j] = u[k][i][j] + (alphatx * (u[k][i+1][j] - 2 * u[k][i][j] + u[k][i-1][j]) +
                                             alphaty * (u[k][i][j+1] - 2 * u[k][i][j] + u[k][i][j-1]) )
                
    return u   

def plotheatmap(u_k, k, dt):
    # Clear the current plot figure
    plt.clf()
    plt.title(f"Temperature at t = {k*dt:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap = plt.cm.jet)
    plt.colorbar()
    
    return plt

def animate(k, u, dt):
        plotheatmap(u[k], k, dt)

def image_to_numpy(image):
    from PIL import Image
    import numpy as np
    im = Image.open(image)
    im = im.convert('L')
    im = im.resize((200, 200))
    im = np.array(im)
    return im

def main():
    # Grid parameters
    nx = 200
    ny = 200
    nt = 100
    lx = 1000
    ly = 1000
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    

    # Equation parameters   
    alpha = 1.0

    # Initial conditions
    u = np.empty((nt, nx, ny))
    u0 = np.zeros((nx, ny))
    #u0[nx//4:3*nx//4, ny//4:3*ny//4] = np.random.rand(250,250) * 500
    u0 = image_to_numpy("./silk.jpg")
    # Rotate 180 degrees
    u0 = np.rot90(u0, 2)
    u[0] = u0
    #u[0] = reset_boundary_conditions(u[0])
    u1 = u
    u2 = u

    # Time parameters
    dt = (dx * dy)/ (4 * alpha)
    print("Parameters: nx = ", nx, ", ny = ", ny, ", nt = ", nt, ", lx = ", lx, ", ly = ", ly, ", dx = ", dx, ", dy = ", dy, ", alpha = ", alpha, ", dt = ", dt)
    # Solve the equation
    u1 = solve_2d_mt(nt, dt, dx, dy, alpha, u1)

    print("Done!")
    # Plot the solution
    anim = animation.FuncAnimation(plt.figure(), animate, fargs=[u1, dt], interval=1, frames=nt, repeat=False)
    anim.save('heat.mp4', fps=dt, extra_args=['-vcodec', 'libx264'])
    

if __name__ == "__main__":
    main()