import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import solver
import pickle

from sklearn.neural_network import MLPRegressor
from memory_profiler import profile
from datetime import timedelta

def create_estimator(fit_interations):
    nvols = 2
    max_value = 1.2e6
    num_tests = 3

    # Estimating max time with 3 order polynomial
    nvols_increase = (max_value / nvols) ** (1 / fit_interations) 
    x = np.append([2.], nvols * np.logspace(1, fit_interations, base = nvols_increase, num = fit_interations))
    y = np.zeros(fit_interations + 1)
    for i in range(fit_interations + 1):
        for j in range(num_tests):
            t = time.time()
            example_random(int(x[i]))
            y[i] += time.time() - t
        y[i] /= num_tests
        print("Iteration: {}/{} - {} x {}".format(i, fit_interations, y[i], num_tests))
    
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    estimated_max_time = p(x * num_tests).sum()
    eta = int(estimated_max_time)
    td = timedelta(seconds = eta if eta > 0 else 0)
    print("Estimated max time: {}".format(td))
    np.save("estimator", z)

def plot_time(f):
    nvols = 2
    num_tests = 3
    nvols_increase = 1.02
    max_value = int(1.2e6)
    max_interations = int(np.log(max_value / nvols) / np.log(nvols_increase))
    solver.print_times = False

    if not os.path.exists("estimator.npy"):
        create_estimator(100)
    z = np.load("estimator.npy")
    p = np.poly1d(z)
    x = np.append([2.], nvols * np.logspace(1, max_interations, base = nvols_increase, num = max_interations))
    estimated_max_time = p(x * num_tests).sum()

    eta = int(estimated_max_time)
    td = timedelta(seconds = eta if eta > 0 else 0)
    print("Estimated max time: {}".format(td))
    
    times = []
    vols = []
    iteration = np.arange(0, max_interations)
    
    print(max_value, len(str(max_value)))
    start_time = time.time()
    for i in range(max_interations + 1):
        total_time = 0
        for j in range(num_tests):
            t = time.time()
            f(int(nvols))
            total_time += time.time() - t
        
        times.append(total_time / num_tests)
        vols.append(int(nvols))

        eta = int(estimated_max_time - (time.time() - start_time))
        td = timedelta(seconds = eta if eta > 0 else 0)
        str_i = str(i)
        str_nvols = str(int(nvols))
        #Add leading zeroes 
        str_i = "0" * (len(str(max_interations)) - len(str_i)) + str_i
        str_nvols = "0" * (len(str(max_value)) - len(str_nvols)) + str_nvols
        round_time = np.around(times[-1], 5)
        # Make sure len(str(round_time)) == 7
        round_time = str(round_time) + "0" * (7 - len(str(round_time)))
        print("Iteration : {}/{} -- \t Vols: {} - {} x {} -- \t ETA: {}".format(str_i, max_interations, 
                                                                                str_nvols, round_time, num_tests, td))
        nvols *= nvols_increase
        
    return vols, times, iteration

def plot1D(mesh, solver, name):
    pass

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
def all_examples(nx = 100, ny = 100, nz = 1, check = False, create_vtk = True):
    example1(nx, ny, nz, check)
    example2(nx, ny, nz, check)
    example3(nx, ny, nz, check)

def example1(nx = 100, ny = 100, nz = 1, check = False, create_vtk = True):
    # Example 1
    Lx, Ly, Lz = 20, 20, 20
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)

    k_time = time.time()
    K1 = solver.get_random_tensor(52, 52, size = (nz, ny, nx))
    if solver.print_times: 
        print("Time to create K1: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()

    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
    if solver.print_times: 
        print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd[up_border] = fd[down_border] = fd[right_border] = True
    fd_values = np.zeros((nx * ny * nz))
    fd_values[up_border] ,fd_values[down_border] ,fd_values[right_border] = 0, 100, 0
    fd1 = (fd, fd_values)
    if solver.print_times: 
        print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[left_border] = True
    fn_values = np.zeros((nx * ny * nz))
    fn_values[left_border] = 0
    fn1 = (fn, fn_values)
    if solver.print_times: 
        print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.zeros((nx * ny * nz))
    if solver.print_times:
        print("-----------------------------------------------------------------------")
    mesh1, solver1 = solver.simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 1", K = K1, q = q,
                        fd = fd1, maskd = True, fn = fn1, maskn = True, create_vtk=create_vtk, check=check)

def example2(nx = 100, ny = 100, nz = 1, check = False, create_vtk = True):
    # Example 1
    Lx, Ly, Lz = 20, 10, 0.1
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)
    k_time = time.time()
    K2 = solver.get_random_tensor(15, 15, size = (nz, ny, nx))
    K_left = [[10., 0., 0.], [0., 50., 0.], [0., 0., 5.]]
    K2[:, :, : nx // 2] = K_left
    if solver.print_times: 
        print("Time to create K2: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()
    
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
    if solver.print_times: 
        print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd[left_border] = fd[right_border] = True
    fd_values = np.zeros((nx * ny * nz))
    fd_values[left_border], fd_values[right_border] = 100, 30
    fd2 = (fd, fd_values)
    if solver.print_times: 
        print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[up_border] = fn[down_border] = True
    fn_values = np.zeros((nx * ny * nz))
    fn_values[up_border] = fn_values[down_border] = 0
    fn2 = (fn, fn_values)
    if solver.print_times: 
        print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.zeros((nx * ny * nz))
    
    if solver.print_times:
        print("-----------------------------------------------------------------------")
    mesh2, solver2 = solver.simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 2", K = K2, q = q,
                        fd = fd2, maskd = True, fn = fn2, maskn = True, create_vtk=create_vtk, check=check)

def example_random(nvols):
    Lx, Ly, Lz = np.random.randint(1, 100), np.random.randint(1, 100), np.random.randint(1, 100)
    part1 = np.random.randint(1, math.ceil(nvols ** (1/2)))
    part2 = np.random.randint(1, math.ceil(nvols ** (1/2)))
    part3 = int(nvols / part1 / part2)

    # Making sure that if
    # nx == 1, ny == 1, nz != 1 -> nx != 1, ny = 1, nz = 1
    # nx == 1, ny != 1, nz = 1 -> nx != 1, ny = 1, nz = 1
    # nx == 1, ny != 1, nz != 1  -> nx != 1, ny != 1, nz = 1
    # nx == a, ny != 1, nz != 1 ->  nx != 1, ny != 1, nz = 1

    # if part1 == 1 and part2 == 1 and part3 != 1:
    #     #Swap part 1 and 3
    #     part1, part3 = part3, part1
    # elif part1 == 1 and part2 != 1 and part3 == 1:
    #     #Swap part 1 and 2
    #     part1, part2 = part2, part1
    # elif part1 == 1 and part2 != 1 and part3 != 1:
    #     #Swap part 1 and 3
    #     part1, part3 = part3, part1
    # elif part1 != 1 and part2 == 1 and part3 != 1:
    #     #Swap part 2 and 3
    #     part2, part3 = part3, part2

    nx, ny, nz = part1, part2, part3
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    
    a, b = np.random.randint(1, 100), np.random.randint(1, 100)
    K = solver.get_random_tensor(a, b, size = (nz, ny, nx))

    all_indices = np.arange(nx * ny * nz)
    all_borders = all_indices[(all_indices % nx == 0) | 
                              (all_indices % nx == nx - 1) | 
                              (all_indices % (nx * ny) < nx) | 
                              (all_indices % (nx * ny) >= nx * (ny - 1)) | 
                              (all_indices < nx * ny) | 
                              (all_indices >= nx * ny * (nz - 1))]
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd_indices = np.random.choice(all_borders, size = np.random.randint(1, len(all_borders)), replace = False)
    fn_indices = np.setdiff1d(all_borders, fd_indices)

    fd[fd_indices] = True
    a,b = np.random.uniform(0, 100), np.random.uniform(0, 100)
    fd_values = np.random.uniform(a, b, size = len(fd_indices))
    fd = (fd, fd_values)

    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[fn_indices] = True
    a,b = np.random.uniform(0, 100), np.random.uniform(0, 100)
    fn_values = np.random.uniform(a, b, size = len(fn_indices))
    fn = (fn, fn_values)

    q = np.zeros((nx * ny * nz))
    a, b = np.random.uniform(0, 100), np.random.uniform(0, 100)
    q = np.random.uniform(1, 100, size = len(q))
    axes = [(nx, dx), (ny, dy), (nz, dz)]
    meshr, solverr = solver.simulate_tpfa(axes, "random e", K = K, q = q, 
                                          fd = fd, maskd = True, fn = fn, maskn = True, 
                                          create_vtk=False, check=False)
    

if __name__ == '__main__':
    #Plot time x number of cells and number of cells x interation
    vols1, time1, iterations1 = plot_time(example_random)

    np.save("vols1", vols1)
    np.save("time1", time1)
    np.save("iterations1", iterations1)
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Time x Number of cells')

    # Plot time x number of cells
    ax1.plot(vols1, time1, label = "")
    ax1.set_xlabel('Number of cells')
    ax1.set_ylabel('Time (s)')
    ax1.legend()
    ax1.grid()

    # Plot number of cells x iterations
    ax2.plot(iterations1, vols1, label = "Example 1")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Number of cells')
    ax2.legend()
    ax2.grid()
    

    plt.show()
