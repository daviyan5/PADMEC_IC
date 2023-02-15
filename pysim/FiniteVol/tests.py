import numpy as np
import time
import os
import math
import solver
import matplotlib.pyplot as plt


from datetime import timedelta

def exemplo1D(nx, ny, nz):

    Lx, Ly, Lz = 20, 10, 0.01
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    solver1D = solver.Solver()
    mesh = solver1D.create_mesh(nx, ny, nz, dx, dy, dz, name = "exemplo1D")
    nvols = mesh.nvols

    v1, v2 = 15., 100.
    d1, d2 = 100., 30.
    vq = 0.

    K = solver.get_tensor(v1, (nz, ny, nx))
    K_left = solver.get_tensor(v2, ())
    K[:, :, : nx // 2] = K_left
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(x == 0, d1, np.where(x == Lx, d2, None))
    fn = lambda x, y, z: np.where(x == 0, 0, np.where(x == Lx, 0, None))

    p = solver1D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False)

    x = mesh.volumes.x[:nx]
    

    a_p = d1 - (d1 - d2) * (x[:nx//2] / (Lx/2)) * (v1/(v1 + v2))
    a_p = np.hstack((a_p, 
                    (d1 + (d1 - d2) * (v2 - v1)/(v1 + v2)) - (d1 - d2) * (x[nx//2:] / (Lx/2)) * (v2/(v1+v2)) ))
    
    if solver.verbose:
        plt.plot(x, p[:nx], label = "TPFA")
        plt.title("Solução 1D - Caso Linear com {} elementos".format(nvols))
        plt.plot(x, a_p, label = "Analítico")
        plt.plot(x, np.abs(a_p - p[:nx]), label = "Erro")
        plt.text(0.7, 0.7, "Erro máximo: {}".format(np.max(np.abs(a_p - p[:nx]))))
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("p")

        plt.grid()
        plt.show()
    
    return mesh, solver1D

def exemplo2D(nx, ny, nz):
    
    Lx, Ly, Lz = 0.6, 1, 0.01
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    solver2D = solver.Solver()
    mesh = solver2D.create_mesh(nx, ny, nz, dx, dy, dz, name = "exemplo2D")
    nvols = mesh.nvols

    v1 = 52.
    d1, d2, d3 = 0., 100., 0.
    vq = 0.

    K = solver.get_tensor(v1, (nz, ny, nx))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, None)))
    fn = lambda x, y, z: np.where(x == 0, 0, None)

    p = solver2D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False)

    if solver.verbose:
        x = mesh.volumes.x[:nx*ny]
        y = mesh.volumes.y[:nx*ny]

        plt.title("Solução 2D - {} elementos".format(nvols))
        # Plot contorno
        plt.contourf(x.reshape((nx, ny)), y.reshape((nx, ny)), p[:nx*ny].reshape((nx, ny)), 100)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.show()

    return mesh, solver2D

def exemplo3D(nx, ny, nz):
        
    Lx, Ly, Lz = 1, 2, 3
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    solver3D = solver.Solver()
    mesh = solver3D.create_mesh(nx, ny, nz, dx, dy, dz, name = "exemplo3D")
    nvols = mesh.nvols

    v1 = 10.
    d1, d2, d3, d4 = 0., 100., 10., 40.
    n1, n2 = 0.3, 0.7
    vq = 0.

    K = solver.get_tensor(v1, (nz, ny, nx))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, np.where(z == 0, d4, None))))
    fn = lambda x, y, z: np.where(x == 0, n1, np.where(z == Lz, n2, None))

    p = solver3D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False)
    return mesh, solver3D

def exemploAleatorio(nx, ny, nz):
    nvols = nx * ny * nz
    nx = np.random.randint(1, int(nvols ** (1/2)))
    ny = np.random.randint(1, int(nvols ** (1/2)))
    nz = nvols // (nx * ny)

    Lx, Ly, Lz = np.random.uniform(1, 10), np.random.uniform(1, 10), np.random.uniform(1, 10)
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    solverA = solver.Solver()
    meshA = solverA.create_mesh(nx, ny, nz, dx, dy, dz, name = "exemploAleatorio")
    nvols = meshA.nvols
    a, b = np.random.uniform(1, 100, 2)
    K = solver.get_random_tensor(a, b, size = (nz, ny, nx))
    q = np.random.uniform(0, 100, nvols)
    fd_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fn_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fd = lambda x, y, z: fd_v
    fn = lambda x, y, z: fn_v

    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = False)
    
    return meshA, solverA

def exemploAnalitico(nx, ny, nz):
    # P = sin(x) + cos(y) + e^(z)
    #grad(p) = (cos(x), -sin(y), e^(z))
    # div(K * grad(p)) = K * (-sen(x) - cos(y) + e^(z)
    
    Lx, Ly, Lz = 1, 2, 3
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    solverA = solver.Solver()
    meshA = solverA.create_mesh(nx, ny, nz, dx, dy, dz, name = "exemploAnalitico")
    nvols = meshA.nvols
    K = solver.get_tensor(1., (nz, ny, nx))
    K_vols = 1.
    pa = lambda x, y, z: np.sin(x) + np.cos(y) + np.exp(z)
    ga = lambda x, y, z: np.array([np.cos(x), -np.sin(y), np.exp(z)]).T
    qa = lambda x, y, z: -K_vols * (-np.sin(x) - np.cos(y) + np.exp(z))
    normal = lambda x, y, z: meshA.faces.normal[meshA.faces.boundary]

    q = qa(meshA.volumes.x, meshA.volumes.y, meshA.volumes.z) * meshA.volume
    # faces_flux = qa(meshA.faces.x, meshA.faces.y, meshA.faces.z) * meshA.faces.areas
    # q = np.empty(nvols)
    # np.add.at(q, meshA.faces.adjacents[:, 0], faces_flux)
    # np.add.at(q, meshA.faces.adjacents[:, 1], -faces_flux)
    
    fd = lambda x, y, z: pa(x, y, z)
    fn = lambda x, y, z: (normal(x, y, z) * ga(x, y, z)).sum(axis = 1)

    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = solver.verbose)
    
    if solver.verbose:
        # Plot 3D solution
        x, y, z = meshA.volumes.x, meshA.volumes.y, meshA.volumes.z
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(x, y, z, c = p, cmap = 'jet')
        ax.title.set_text('Solução numérica')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Plot analytical solution
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(x, y, z, c = pa(x, y, z), cmap = 'jet')
        ax.title.set_text('Solução analítica')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        solverA.p[:] = pa(x, y, z)
        solverA._create_vtk("exemploAnalitico_A")
        plt.show()

    return meshA, solverA


def testes_tempo():
    nvols0 = 10
    nvolsMAX = int(1e5)
    incremento = 1.2
    nvols0 = int(max(nvols0, 1/(incremento - 1) + 1))
    ntestes = 100
    tempos = np.zeros(ntestes)
    niteracoes = int(np.ceil(np.log(float(nvolsMAX)/nvols0) / np.log(incremento))) + 1
    
    solver.verbose = False

    f = [exemplo1D, exemplo2D, exemplo3D, exemploAleatorio]
    
    def run_iterations(f):
        tempos = np.empty((niteracoes, 4, 3))
        # Tempos[i][0] = [MIN, MEDIO, MAX] -> Montar Malha
        # Tempos[i][1] = [MIN, MEDIO, MAX] -> Montar Matriz
        # Tempos[i][2] = [MIN, MEDIO, MAX] -> Resolver Sistema
        # Tempos[i][3] = [MIN, MEDIO, MAX] -> Total
        nvols = nvols0
        vols = []


        iter_time = time.time()
        for i in range(niteracoes):
            str_i = "0" * (len(str(niteracoes)) - len(str(i))) + str(i)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(nvols))) + str(nvols)

            
            nx, ny, nz = int(nvols ** (1/3)), int(nvols ** (1/3)), int(nvols ** (1/3))
            tempos_aux = np.empty(ntestes, dtype = object)
            for j in range(ntestes):
                total_time = time.time()
                mesh, solver = f(nx, ny, nz)
                mesh.times["Total"] = time.time() - total_time
                tempos_aux[j] = mesh.times
            
            t1 = np.array([t["Montar Malha"] for t in tempos_aux])
            t2 = np.array([t["Montar Matriz A do TPFA"] for t in tempos_aux])
            t3 = np.array([t["Resolver o Sistema TPFA"] for t in tempos_aux])
            t4 = np.array([t["Total"] for t in tempos_aux])
            tempos[i][0][:] = [t1.min(), t1.mean(), t1.max()]
            tempos[i][1][:] = [t2.min(), t2.mean(), t2.max()]
            tempos[i][2][:] = [t3.min(), t3.mean(), t3.max()]
            tempos[i][3][:] = [t4.min(), t4.mean(), t4.max()]

            t_str = timedelta(seconds = time.time() - iter_time)
            print("Iteration : {}/{}  \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
            vols.append(nvols)
            nvols = int(nvols * incremento)
        return tempos, vols
    
    
    for i in range(len(f)):
        start_time = time.time()
        tempos, vols = run_iterations(f[i])
        tempos = tempos[1:]
        vols = vols[1:]

        fig, ax = plt.subplots(2, 2, figsize = (10, 5))
        plt.suptitle(f[i].__name__)
       

        
        ax[0][0].plot(vols, tempos[:, 3, 0], linestyle = "dotted", color = "black", label = "Mínimo")
        ax[0][0].plot(vols, tempos[:, 3, 1], linestyle = "solid", color = "black", label = "Média")
        ax[0][0].plot(vols, tempos[:, 3, 2], linestyle = "dashed", color = "black", label = "Máximo")
        ax[0][0].set_xlabel("Número de volumes")
        ax[0][0].set_ylabel("Tempo (s)")
        ax[0][0].grid()
        ax[0][0].legend()
        ax[0][0].title.set_text("Total")

        ax[0][1].plot(vols, tempos[:, 0, 0], linestyle = "dotted", color = "blue", label = "Mínimo")
        ax[0][1].plot(vols, tempos[:, 0, 1], linestyle = "solid", color = "blue", label = "Média")
        ax[0][1].plot(vols, tempos[:, 0, 2], linestyle = "dashed", color = "blue", label = "Máximo")
        ax[0][1].set_xlabel("Número de volumes")
        ax[0][1].set_ylabel("Tempo (s)")
        ax[0][1].grid()
        ax[0][1].legend()
        ax[0][1].title.set_text("Montar Malha")

        ax[1][0].plot(vols, tempos[:, 1, 0], linestyle = "dotted", color = "red", label = "Mínimo")
        ax[1][0].plot(vols, tempos[:, 1, 1], linestyle = "solid", color = "red", label = "Média")
        ax[1][0].plot(vols, tempos[:, 1, 2], linestyle = "dashed", color = "red", label = "Máximo")
        ax[1][0].set_xlabel("Número de volumes")
        ax[1][0].set_ylabel("Tempo (s)")
        ax[1][0].grid()
        ax[1][0].legend()
        ax[1][0].title.set_text("Montar Matriz A")

        ax[1][1].plot(vols, tempos[:, 2, 0], linestyle = "dotted", color = "green", label = "Mínimo")
        ax[1][1].plot(vols, tempos[:, 2, 1], linestyle = "solid", color = "green", label = "Média")
        ax[1][1].plot(vols, tempos[:, 2, 2], linestyle = "dashed", color = "green", label = "Máximo")
        ax[1][1].set_xlabel("Número de volumes")
        ax[1][1].set_ylabel("Tempo (s)")
        ax[1][1].grid()
        ax[1][1].legend()
        ax[1][1].title.set_text("Resolver Sistema")

        plt.tight_layout()
        plt.savefig("./tmp/{}.png".format(f[i].__name__))
        #plt.show()
        print("Total Time for {}: \t {}".format(f[i].__name__, timedelta(seconds = time.time() - start_time)))
    
    
    
if __name__ == '__main__':
    solver.verbose = True
    #exemplo1D(1000, 1, 1)
    #exemplo2D(300, 300, 1)
    #exemplo3D(60, 60, 60)
    #exemploAleatorio(50,50,50)
    exemploAnalitico(40, 40, 40)
    #testes_tempo()
    
    
