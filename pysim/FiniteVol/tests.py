import numpy as np
import time
import os
import math
import solver
import matplotlib.pyplot as plt


from datetime import timedelta

def exemplo1D(nx, ny, nz):

    solver.verbose = True
    Lx, Ly, Lz = 20, 10, 0.01
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    solver1D = solver.Solver()
    mesh = solver1D.create_mesh(nx, ny, nz, dx, dy, dz)
    nvols = mesh.nvols

    v1, v2 = 15., 100.
    d1, d2 = 100., 30.
    K = solver.get_tensor(v1, (nz, ny, nx))
    K_left = solver.get_tensor(v2, ())
    K[:, :, : nx // 2] = K_left
    q = np.full(nvols, 0)

    fd = lambda x, y, z: np.where(x == 0, d1, np.where(x == Lx, d2, None))
    fn = lambda x, y, z: np.where(x == 0, 0, np.where(x == Lx, 0, None))

    p = solver1D.solve(mesh, K, q, fd, fn, create_vtk = True, check = False)

    x = mesh.volumes.x[:nx]
    plt.plot(x, p[:nx], label = "TPFA")

    a_p = d1 - (d1 - d2) * (x[:nx//2] / (Lx/2)) * (v1/(v1 + v2))
    a_p = np.hstack((a_p, 
                    (d1 + (d1 - d2) * (v2 - v1)/(v1 + v2)) - (d1 - d2) * (x[nx//2:] / (Lx/2)) * (v2/(v1+v2)) ))

    plt.plot(x, a_p, label = "Analítico")
    plt.plot(x, np.abs(a_p - p[:nx]), label = "Erro")
    plt.text(0.7, 0.7, "Erro máximo: {}".format(np.max(np.abs(a_p - p[:nx]))))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("p")

    plt.grid()
    plt.show()




    
if __name__ == '__main__':
    exemplo1D(100000, 10, 1)
    
    
