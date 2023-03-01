import numpy as np
import time
import os
import math
import solver
import matplotlib.pyplot as plt
from datetime import timedelta

def exemplo1D(meshfile):

    Lx, Ly, Lz = 20, 20, 20
    solver1D = solver.Solver()
    mesh = solver1D.create_mesh(meshfile)
    nvols = len(mesh.volumes)

    v1, v2 = 15., 100.
    d1, d2 = 100., 30.
    vq = 0.

    x = mesh.volumes.center[:][:, 0]
    K = solver.get_tensor(v1, (nvols))
    K_left = solver.get_tensor(v2, ())
    K[x < Lx / 2] = K_left
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(x == 0., d1, np.where(x == Lx, d2, None))
    fn = lambda x, y, z: np.where(y == 0., 0., np.where(y == Lx, 0., None))

    a_p = lambda x, y, z : np.where(x < Lx/2, d1 - (d1 - d2) * (x / (Lx/2)) * (v1/(v1 + v2)), 
                                             (d1 + (d1 - d2) * (v2 - v1)/(v1 + v2)) - (d1 - d2) * (x / (Lx/2)) * (v2/(v1+v2)))
                                
    p = solver1D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False, an_sol=a_p)

    
    return mesh, solver1D

def exemplo2D(meshfile):
    
    Lx, Ly, Lz = 20, 20, 20

    solver2D = solver.Solver()
    mesh = solver2D.create_mesh(meshfile)
    nvols = len(mesh.volumes)

    v1 = 52.
    d1, d2, d3 = 0., 100., 0.
    vq = 0.

    K = solver.get_tensor(v1, (nvols))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, None)))
    fn = lambda x, y, z: np.where(x == 0, 0, None)

    p = solver2D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False)

    return mesh, solver2D

def exemplo3D(meshfile):
        
    Lx, Ly, Lz = 1, 2, 3

    solver3D = solver.Solver()
    mesh = solver3D.create_mesh(meshfile)
    nvols = len(mesh.volumes)

    v1 = 10.
    d1, d2, d3, d4 = 0., 100., 10., 40.
    n1, n2 = 0.3, 0.7
    vq = 0.

    K = solver.get_tensor(v1, (nvols))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, np.where(z == 0, d4, None))))
    fn = lambda x, y, z: np.where(x == 0, n1, np.where(z == Lz, n2, None))

    p = solver3D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False)
    return mesh, solver3D

def exemploAleatorio(meshfile):

    Lx, Ly, Lz = 20, 20, 20
    

    solverA = solver.Solver()
    meshA = solverA.create_mesh(meshfile)
    nvols = len(meshA.volumes)
    a, b = np.random.uniform(1, 100, 2)
    K = solver.get_random_tensor(a, b, size = (nvols))
    q = np.random.uniform(0, 100, nvols)
    fd_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fn_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fd = lambda x, y, z: fd_v
    fn = lambda x, y, z: fn_v

    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = False)
    
    return meshA, solverA

def exemploAnalitico(meshfile, pa, ga, qa, K_vols):
    # P = sin(x) + cos(y) + sen(x)
    #grad(p) = (cos(x), -sin(y), cos(z))
    # div(K * grad(p)) = K * (-sen(x) - cos(y) - sen(z))
    
    Lx, Ly, Lz = 1, 1, 1

    solverA = solver.Solver()
    meshA = solverA.create_mesh(meshfile)
    nvols = len(meshA.volumes)
    K = solver.get_tensor(K_vols, (nvols))
    
    normal = lambda x, y, z: meshA.faces.normal[meshA.faces.boundary]
    coords = meshA.volumes.center[:]
    xv, yv, zv = coords[:, 0], coords[:, 1], coords[:, 2]

    q = qa(xv, yv, zv) * meshA.volumes.volume[:]
    
    fd = lambda x, y, z: pa(x, y, z)
    fn = lambda x, y, z: (normal(x, y, z) * ga(x, y, z) * -K_vols).sum(axis = 1)
    #np.zeros_like(x)#
    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = solver.verbose, an_sol=pa)
    
    if solver.verbose:
        # Plot 3D solution
        fig = plt.figure()
        fig.suptitle('Erro Médio Quadrático: ' + str(np.mean((p - pa(xv, yv, zv)) ** 2)))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(xv, yv, zv, c = p, cmap = 'jet')
        ax.title.set_text('Solução numérica')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Plot analytical solution
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(xv, yv, zv, c = pa(xv, yv, zv), cmap = 'jet')
        ax.title.set_text('Solução analítica')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.savefig("teste.png")
        #plt.show() if nvols < 10000 else None

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
  
def testes_precisao(solutions, K_vols):
    # Solutions = tuple([p0, g0, q0], [p1, g1, q1], ...)
    nvols0 = 10
    nvolsMAX = int(2e5)
    incremento = 1.2
    nvols0 = int(max(nvols0, 1/(incremento - 1) + 1))
    ntestes = 1
    tempos = np.zeros(ntestes)
    niteracoes = int(np.ceil(np.log(float(nvolsMAX)/nvols0) / np.log(incremento))) + 1
    
    solver.verbose = False

    
    def run_iterations(f, pa, ga, qa, K_vols):
        
        nvols = nvols0
        vols = []
        precisao_media = []
        erro_medio = []

        iter_time = time.time()
        for i in range(niteracoes):
            str_i = "0" * (len(str(niteracoes)) - len(str(i + 1))) + str(i + 1)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(nvols))) + str(nvols)

            
            nx, ny, nz = int(nvols ** (1/3)), int(nvols ** (1/3)), int(nvols ** (1/3))
            tempos_aux = np.empty(ntestes, dtype = object)
            mesh, solver = f(nx, ny, nz, pa, ga, qa, K_vols)

            p_a = pa(mesh.volumes.x, mesh.volumes.y, mesh.volumes.z)
            error = np.sqrt(np.sum((p_a - solver.p)**2 * mesh.volumes.volumes) / np.sum(p_a**2 * mesh.volumes.volumes))

            erro_medio.append(error)
            precisao_media.append(1. - error)
            vols.append(nvols)

            t_str = timedelta(seconds = time.time() - iter_time)
            print("Iteration : {}/{}  \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
            nvols = int(nvols * incremento)

        return precisao_media, erro_medio, vols
    
    fig, ax = plt.subplots(len(solutions), figsize = (10, 5))
    fig.suptitle("Comparação com Soluções Analíticas")
    for i, solution in enumerate(solutions):
        start_time = time.time()
        precisao, erro_medio, vols = run_iterations(exemploAnalitico, solution[0], solution[1], solution[2], K_vols)
        ax[i].plot(vols, erro_medio, label = "I²rel", color = "blue", marker = "o")
        ax[i].set_xlabel("Número de volumes")
        ax[i].set_ylabel("I²rel")
        ax[i].set_xscale("log")
        ax[i].grid()
        ax[i].legend()
        print("Total Time : \t {}".format(timedelta(seconds = time.time() - start_time)))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    solver.verbose = True
    meshfile = "./mesh/20.h5m"
    #exemplo1D(meshfile)
    #exemplo2D(meshfile)
    #exemplo3D(meshfile)
    #exemploAleatorio(meshfile)
    
    K_vols = 112.435
    pa1 = lambda x, y, z: np.sin(x) + np.cos(y) + np.exp(z)
    ga1 = lambda x, y, z: np.array([np.cos(x), -np.sin(y), np.exp(z)]).T
    qa1 = lambda x, y, z: -K_vols * (-np.sin(x) - np.cos(y) + np.exp(z))
    pa2 = lambda x, y, z: x**2 + y**2 + z**2
    ga2 = lambda x, y, z: np.array([2*x, 2*y, 2*z]).T
    qa2 = lambda x, y, z: -K_vols * (2) * np.ones_like(x)
    exemploAnalitico(meshfile, pa2, ga2, qa2, K_vols)
    #testes_tempo()
    #testes_precisao([(pa1, ga1, qa1), (pa2, ga2, qa2)], K_vols)
    
    
