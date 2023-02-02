import numpy as np
import time

from .solver import simulate_tpfa, get_random_tensor
from memory_profiler import profile

@profile
def exemplo1():
    # Example 1
    Lx, Ly = 0.6, 1
    nx, ny, nz = 500, 500, 1
    dx, dy, dz = Lx/nx, Ly/ny, 0.1
    all_indices = np.arange(nx * ny * nz)

    k_time = time.time()
    K1 = get_random_tensor(52, 52, size = (nz, ny, nx))
    print("Time to create K1: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()

    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1):]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[0: nx]
    print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd[up_border] = fd[down_border] = fd[right_border] = True
    fd_values = np.zeros((nx * ny * nz))
    fd_values[up_border] ,fd_values[down_border] ,fd_values[right_border] = 0, 100, 0
    fd1 = (fd, fd_values)
    print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[left_border] = True
    fn_values = np.zeros((nx * ny * nz))
    fn_values[left_border] = 0
    fn1 = (fn, fn_values)
    print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.zeros((nx * ny * nz))
    print("-----------------------------------------------------------------------")
    mesh1 = simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "Exemplo 1", K = K1, q = q,
                        fd = fd1, maskd = True, fn = fn1, maskn = True, create_vtk=True)

def exemplo2():
    # Example 1
    Lx, Ly = 20, 10
    nx, ny, nz = 50, 50, 1
    dx, dy, dz = Lx/nx, Ly/ny, 0.1
    all_indices = np.arange(nx * ny * nz)
    k_time = time.time()
    K2 = get_random_tensor(15, 15, size = (nz, ny, nx))
    K_left = [[50., 0., 0.], [0., 50., 0.], [0., 0., 50.]]
    K2[:, :, : nx // 2] = K_left
    print("Time to create K2: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()
    
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1):]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[0: nx]
    print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd[left_border] = fd[right_border] = True
    fd_values = np.zeros((nx * ny * nz))
    fd_values[left_border], fd_values[right_border] = 100, 30
    fd2 = (fd, fd_values)
    print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[up_border] = fn[down_border] = True
    fn_values = np.zeros((nx * ny * nz))
    fn_values[up_border] = fn_values[down_border] = 0
    fn2 = (fn, fn_values)
    print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.zeros((nx * ny * nz))
    
    print("-----------------------------------------------------------------------")
    mesh2 = simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "Exemplo 2", K = K2, q = q,
                        fd = fd2, maskd = True, fn = fn2, maskn = True, create_vtk=True)
