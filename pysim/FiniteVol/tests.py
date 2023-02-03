import numpy as np
import time

from solver import simulate_tpfa, get_random_tensor
from memory_profiler import profile


def compare():
    names = ["volumes_trans","faces_trans", 
             "A", "p",
             "dirichlet_points", "dirichlet_values", 
             "neumann_points", "neumann_values", "faces_adjacents"]
    
    for name in names:
        legacy = np.load("legacy" + name + ".npy")
        new = np.load(name + ".npy")
        print("Comparing {}...".format(name))
        print("Legacy: \t", legacy.shape)
        print("New: \t\t", new.shape)
        print("Equal: \t\t", np.array_equal(legacy, new))
        print("-----------------------------------------------------------------------")
    
@profile
def all_examples(nx = 100, ny = 100, nz = 1, check = False):
    example1(nx, ny, nz, check)
    example2(nx, ny, nz, check)
    example3(nx, ny, nz, check)

def example1(nx = 100, ny = 100, nz = 1, check = False):
    # Example 1
    Lx, Ly, Lz = 0.6, 1, 0.05
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)

    k_time = time.time()
    K1 = get_random_tensor(52, 52, size = (nz, ny, nx))
    print("Time to create K1: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()

    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
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
    mesh1, solver1 = simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 1", K = K1, q = q,
                        fd = fd1, maskd = True, fn = fn1, maskn = True, create_vtk=True, check=check)

def example2(nx = 100, ny = 100, nz = 1, check = False):
    # Example 1
    Lx, Ly, Lz = 20, 10, 0.1
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)
    k_time = time.time()
    K2 = get_random_tensor(15, 15, size = (nz, ny, nx))
    K_left = [[10., 0., 0.], [0., 50., 0.], [0., 0., 5.]]
    K2[:, :, : nx // 2] = K_left
    print("Time to create K2: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()
    
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
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
    mesh2, solver2 = simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 2", K = K2, q = q,
                        fd = fd2, maskd = True, fn = fn2, maskn = True, create_vtk=True, check=check)

def example3(nx = 100, ny = 100, nz = 1, check = False):
    Lx, Ly, Lz = 10, 15, 20
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)
    k_time = time.time()
    K3 = get_random_tensor(10, 15, size = (nz, ny, nx))
    print("Time to create K3: \t\t {} s".format(round(time.time() - k_time, 5)))


    border_time = time.time()
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
    print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd[left_border] = fd[right_border] = fd[back_border] = True
    fd_values = np.zeros((nx * ny * nz))
    fd_values[left_border] = np.random.uniform(30, 100, size = len(left_border))
    fd_values[right_border] = np.random.uniform(50, 100, size = len(right_border))
    fd_values[back_border] = np.random.uniform(4, 100, size = len(back_border))
    fd3 = (fd, fd_values)
    print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))

    n_time = time.time()
    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[up_border] = fn[down_border] = True
    fn_values = np.zeros((nx * ny * nz))
    fn_values[up_border] = fn_values[down_border] = 0
    fn3 = (fn, fn_values)
    print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.zeros((nx * ny * nz))

    print("-----------------------------------------------------------------------")
    mesh3, solver3 = simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 3", K = K3, q = q,
                        fd = fd3, maskd = True, fn = fn3, maskn = True, create_vtk=True, check=check)
    

if __name__ == '__main__':
    all_examples(40, 40, 40, check = True)