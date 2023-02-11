import numpy as np
import time
import os
import math
import solver

from datetime import timedelta



def time_tests(f, num_tests = 3, nvols = 2, nvols_increase = 1.004, max_value = int(1.05e6)):
    max_interations = int(np.log(max_value / nvols) / np.log(nvols_increase))
    solver.verbose = False
    estimated_max_time = 0.
    estimated_min_time = 0.
    estimated_time = 0.

    if not os.path.exists("./important/estimator.npy"):
        print("Estimator not found, will be created")
    else:
        z = np.load("./important/estimator.npy")
        up = np.load("./important/up_estimator.npy")
        low = np.load("./important/low_estimator.npy")
        p = np.poly1d(z)
        x = np.append([2.], nvols * np.logspace(1, max_interations, base = nvols_increase, num = max_interations))
        estimated_time = (num_tests * p(x)).sum()
        estimated_max_time = (num_tests * np.poly1d(up)(x)).sum()
        estimated_min_time = (num_tests * np.poly1d(low)(x)).sum()

    
    times = []
    vols = []
    
    
    start_time = time.time()
    for i in range(max_interations + 1):
        total_time = 0
        for j in range(num_tests):
            t = time.time()
            f(int(nvols))
            total_time += time.time() - t

        times.append(total_time / num_tests)
        vols.append(int(nvols))

        np.save("./tmp/times.npy", times)
        np.save("./tmp/vols.npy", vols)

        eta = int(estimated_time - (time.time() - start_time))
        td = timedelta(seconds = eta if eta > 0 else 0)

        etamx = int(estimated_max_time - (time.time() - start_time))
        tdmx = timedelta(seconds = etamx if etamx > 0 else 0)

        etamn = int(estimated_min_time - (time.time() - start_time))
        tdmn = timedelta(seconds = etamn if etamn > 0 else 0)

        ##--------------- VERBOSE ---------------##
        str_i = str(i)
        str_nvols = str(int(nvols))

        #Add leading zeroes 
        str_i = "0" * (len(str(max_interations)) - len(str_i)) + str_i
        str_nvols = "0" * (len(str(max_value)) - len(str_nvols)) + str_nvols
        round_time = np.around(times[-1], 5)
        # Make sure len(str(round_time)) == 7
        round_time = str(round_time) + "0" * (7 - len(str(round_time)))
        ##---------------------------------------##

        print("Iteration : {}/{} -- \t Vols: {} - {} x {} -- \t ETA: {} -> MAX: {} -- MIN : {}".format(str_i, max_interations, 
                                                                                                       str_nvols, round_time, 
                                                                                                       num_tests, td, tdmx,tdmn))
        nvols = nvols * nvols_increase
        
    return vols, times
def do_time_tests():
    
    vols, time = time_tests(example_random, 
                           num_tests=3, nvols=2, nvols_increase=1.004, max_value=int(1e6))

   

    # Create estimator if not already created
    if not os.path.exists("./important/estimator.npy"):
        # Fit linear model to time x vols
        vols = np.load("./important/vols.npy")
        times = np.load("./important/times.npy")

        vols = np.array(vols)
        times = np.array(times)
        
        # Fit linear model to time x vols
        z = np.polyfit(vols, times, 1)
        p = np.poly1d(z)
       
        np.save("./important/estimator.npy", z)
        # Get the peak time every 20 num_vols and the corresponding number of cells
        num_vols = 20
        peaks = [[],[]]
        valleys = [[],[]]

        for i in range(0, len(times), num_vols):
            peak = np.max(times[i:i+num_vols])
            peaks[0].append(peak)
            peaks[1].append(vols[i+np.argmax(times[i:i+num_vols])])

            valley = np.min(times[i:i+num_vols])
            valleys[0].append(valley)
            valleys[1].append(vols[i+np.argmin(times[i:i+num_vols])])
            
        # Fit linear polynomial to the peaks
        up_p = np.poly1d(np.polyfit(peaks[1], peaks[0], 1))
        low_p = np.poly1d(np.polyfit(valleys[1], valleys[0], 1))
        up_estimated = up_p(vols)
        low_estimated = low_p(vols)
        
        
        np.save("./important/up_estimator.npy", up_p)
        np.save("./important/low_estimator.npy", low_p)
def all_examples(nx = 100, ny = 100, nz = 100, check = False, create_vtk = True):
    solver.verbose = True
    example1(nx, ny, nz, check)
    example2(nx, ny, nz, check)
    example_random(nvols = nx * ny * nz)
def example1(nx = 10, ny = 10, nz = 10, check = False, create_vtk = True):
    # Example 1
    Lx, Ly, Lz = 0.6, 1, 0.00001
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)

    k_time = time.time()
    K1 = solver.get_random_tensor(52, 52, size = (nz, ny, nx))
    if solver.verbose: 
        print("Time to create K1: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()

    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
    if solver.verbose: 
        print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.hstack((up_border, down_border, right_border))
    
    fd_values = np.zeros((nx * ny * nz))
    fd_values[up_border] ,fd_values[down_border] ,fd_values[right_border] = 0, 100, 0
   

    fd1 = (fd, fd_values)
    if solver.verbose: 
        print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.hstack((left_border))
    fn_values = np.zeros((nx * ny * nz))
    fn_values[left_border] = 0

    fn1 = (fn, fn_values)
    if solver.verbose: 
        print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.full((nx * ny * nz), 0.)
    if solver.verbose:
        print("-----------------------------------------------------------------------")
    mesh1, solver1 = solver.simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 1", K = K1, q = q,
                                          fd = fd1, fn = fn1, create_vtk=create_vtk, check=check)    
def example2(nx = 100, ny = 100, nz = 1, check = False, create_vtk = True):
    # Example 1
    Lx, Ly, Lz = 20, 20,20
    nx, ny, nz = nx, ny, nz
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    all_indices = np.arange(nx * ny * nz)
    k_time = time.time()
    K2 = solver.get_random_tensor(15., 15., size = (nz, ny, nx))
    K_left = [[50., 0., 0.], [0., 50., 0.], [0., 0., 50.]]
    K2[:, :, : nx // 2] = K_left
    if solver.verbose: 
        print("Time to create K2: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]
    if solver.verbose: 
        print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.hstack((left_border, right_border))

    fd_values = np.zeros((nx * ny * nz))
    fd_values[left_border], fd_values[right_border] = 100., 30.

    fd2 = (fd, fd_values)
    if solver.verbose: 
        print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.hstack((up_border, down_border))
    
    fn_values = np.zeros((nx * ny * nz))
    fn_values[up_border] = fn_values[down_border] = 0.
    fn2 = (fn, fn_values)
    if solver.verbose: 
        print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    q = np.zeros((nx * ny * nz))
    
    if solver.verbose:
        print("-----------------------------------------------------------------------")
    mesh2, solver2 = solver.simulate_tpfa([(nx, dx), (ny, dy), (nz, dz)], "example 2", K = K2, q = q,
                        fd = fd2, fn = fn2, create_vtk=create_vtk, check=check)
    
    #compare with impress
    #impress_tpfa = np.load("./tmp/impress_tpfa.npy", allow_pickle=True)
    #impress_p = np.load("./tmp/impress_p.npy", allow_pickle=True)
    #impress_q = np.load("./tmp/impress_q.npy", allow_pickle=True)

    #from scipy.sparse import csr_matrix
    #assert np.allclose(impress_tpfa, solver2.A.todense())
    #impress_p = impress_p.reshape((20, 20, 20)).flatten()
    #print(impress_tpfa[:3], solver2.A.todense()[:3])
    
    #assert np.allclose(impress_p, solver2.p)
    #assert np.allclose(impress_q, q)

    ## Plotar pressão por x
    if solver.verbose:
        print("-----------------------------------------------------------------------")
        print("Plotting pressure by x")
        from matplotlib import pyplot as plt
        plt.plot(mesh2.volumes.x[:nx], solver2.p[:nx], label = "Numerical")
        plt.text(mesh2.volumes.x[nx//2], solver2.p[nx//2], "p = {:.2f}".format(solver2.p[nx//2]))
        an_p = 100 - (100 - 30) * (mesh2.volumes.x[:nx//2] / 10) * (15/65)
        an_p = np.hstack((an_p, (100 + 70 * (50 - 15)/65) - (100 - 30) * (mesh2.volumes.x[nx//2:nx] / 10) * (50/65)))
        plt.plot(mesh2.volumes.x[:nx], an_p[:nx], label = "Analytical")
        plt.plot(mesh2.volumes.x[:nx], solver2.p[:nx] - an_p, label = "Error")
        plt.legend()
        print("Max error: {}".format(np.max(np.abs(solver2.p[:nx] - an_p))))
        plt.grid()
        plt.show()
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
    
    fd_indices = np.random.choice(all_borders, size = np.random.randint(1, len(all_borders)), replace = False)
    fn_indices = np.setdiff1d(all_borders, fd_indices)

    a,b = np.random.uniform(0, 100), np.random.uniform(0, 100)
    fd_values = np.zeros((nx * ny * nz))
    fd_values[fd_indices] = np.random.uniform(a, b, size = len(fd_indices))
    fd = (fd_indices, fd_values)

    
    a,b = np.random.uniform(0, 100), np.random.uniform(0, 100)
    fn_values = np.zeros((nx * ny * nz))
    fn_values[fn_indices] = np.random.uniform(a, b, size = len(fn_indices))
    fn = (fn_indices, fn_values)

    q = np.zeros((nx * ny * nz))
    a, b = np.random.uniform(0, 100), np.random.uniform(0, 100)
    q = np.random.uniform(1, 100, size = len(q))

    axes = [(nx, dx), (ny, dy), (nz, dz)]
    meshr, solverr = solver.simulate_tpfa(axes, "random e", K = K, q = q, 
                                          fd = fd, fn = fn,
                                          create_vtk=True, check=True)   
def example_comparative(nx = 50, ny = 50, nz = 1, check = False, create_vtk = True):
    # Let p = sen(x) + 2 * cos(y) + 3 * tan(z)
    # grad(p) = (cos(x), -2 * sin(y), 3 / (cos(z) ** 2))
    # -div(gra(p)) = sin(x) - 2cos(y) + 6tan(z)sec²(z) = d
    # q = d * A * k
    Lx, Ly, Lz = 100, 100, 1
    dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
    x = np.linspace(dx, Lx - dx, nx)
    y = np.linspace(dy, Ly - dy, ny)
    z = np.linspace(dz, Lz - dz, nz)

    K = 1
    X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")
    p = np.sin(X) + 2 * np.cos(Y) 
    d = np.sin(X) - 2 * np.cos(Y)
    q = d * K * dx * dy * dz

    Kt = solver.get_random_tensor(K, K, size = (nz, ny, nx))

    all_indices = np.arange(nx * ny * nz)
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1) <= all_indices  % (nx * ny)]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[(all_indices % (nx * ny)) < nx]
    front_border = all_indices[all_indices < nx * ny]
    back_border = all_indices[nx * ny * (nz - 1) <= all_indices]

    fd = np.hstack((left_border, up_border, right_border, down_border, front_border, back_border))

    fd = (fd, fd_values)

    fn = np.empty()
    fn_values = np.zeros((nx * ny * nz))
    fn = (fn, fn_values)

    axes = [(nx, dx), (ny, dy), (nz, dz)]
    meshr, solverr = solver.simulate_tpfa(axes, "test e", K = Kt, q = q, 
                                          fd = fd, fn = fn,
                                          create_vtk=True, check=True)
    
    import matplotlib.pyplot as plt


    
if __name__ == '__main__':
    pass
    
    
