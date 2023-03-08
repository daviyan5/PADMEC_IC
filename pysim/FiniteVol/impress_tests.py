import numpy as np
import time
import sys
import os
import math
import solver
import matplotlib.pyplot as plt
from mesh_gen import create_box, legacy_create_box
from datetime import timedelta
from memory_profiler import profile


def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__

def exemplo1D(order, Lx, Ly, Lz, meshfile = None):

    solver1D = solver.Solver()
    montar_time = time.time()
    if not meshfile:
        meshfile = create_box(Lx, Ly, Lz, order)
    mesh = solver1D.create_mesh(meshfile)
    solver1D.times["Montar Malha"] = time.time() - montar_time

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

    return solver1D
    
def exemplo2D(order, Lx, Ly, Lz):

    solver2D = solver.Solver()

    montar_time = time.time()
    meshfile = create_box(Lx, Ly, Lz, order)
    mesh = solver2D.create_mesh(meshfile)
    solver2D.times["Montar Malha"] = time.time() - montar_time

    nvols = len(mesh.volumes)

    v1 = 52.
    d1, d2, d3 = 0., 100., 0.
    vq = 0.

    K = solver.get_tensor(v1, (nvols))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, None)))
    fn = lambda x, y, z: np.where(x == 0, 0, None)

    p = solver2D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False)

    return solver2D

def exemplo3D(order, Lx, Ly, Lz):

    solver3D = solver.Solver()

    montar_time = time.time()
    meshfile = create_box(Lx, Ly, Lz, order)
    mesh = solver3D.create_mesh(meshfile)
    solver3D.times["Montar Malha"] = time.time() - montar_time

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
    return solver3D

def exemploAleatorio(order, Lx, Ly, Lz):

    solverA = solver.Solver()

    montar_time = time.time()
    meshfile = create_box(Lx, Ly, Lz, order)
    meshA = solverA.create_mesh(meshfile)
    solverA.times["Montar Malha"] = time.time() - montar_time

    nvols = len(meshA.volumes)
    a, b = np.random.uniform(1, 100, 2)
    K = solver.get_random_tensor(a, b, size = (nvols))
    q = np.random.uniform(0, 100, nvols)
    fd_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fn_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fd = lambda x, y, z: fd_v
    fn = lambda x, y, z: fn_v

    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = False)
    
    return solverA

def exemploAnalitico(order, pa, ga, qa, K_vols, Lx, Ly, Lz, meshfile = None):
    # P = sin(x) + cos(y) + sen(x)
    #grad(p) = (cos(x), -sin(y), cos(z))
    # div(K * grad(p)) = K * (-sen(x) - cos(y) - sen(z))
    
    solverA = solver.Solver()

    montar_time = time.time()
    if not meshfile:
        meshfile = create_box(Lx, Ly, Lz, order)
    meshA = solverA.create_mesh(meshfile)
    solverA.times["Montar Malha"] = time.time() - montar_time

    nvols = len(meshA.volumes)
    K = solver.get_tensor(K_vols, (nvols))
    
    normal = meshA.faces.normal[meshA.faces.boundary]
    coords = meshA.volumes.center[:]
    xv, yv, zv = coords[:, 0], coords[:, 1], coords[:, 2]

    q = qa(xv, yv, zv) * solverA.volumes
    
    fd = lambda x, y, z: pa(x, y, z)
    fn = lambda x, y, z: (normal * ga(x, y, z) * -K_vols).sum(axis = 1)
    #np.zeros_like(x)#
    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = solver.verbose, an_sol=pa)
    #np.savetxt('d_nodes.txt', solverA.d_nodes.astype('int'))
    #np.savetxt('n_nodes.txt', solverA.n_nodes.astype('int'))
    #np.savetxt('d_vals.txt', solverA.d_values)
    #np.savetxt('n_vals.txt', solverA.n_values)
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

    return solverA

def testes_tempo(Lx, Ly, Lz):
    
    nvols0 = 100
    nvolsMAX = int(5e2)
    incremento = 1.7
    nvols0 = int(max(nvols0, 1/(incremento - 1) + 1))
    ntestes = 10
    tempos = np.zeros(ntestes)
    niteracoes = int(np.ceil(np.log(float(nvolsMAX)/nvols0) / np.log(incremento))) + 1
    
    solver.verbose = False

    f = [exemplo1D, exemplo2D, exemplo3D, exemploAleatorio]
    
    fp = open('memory_profiler.log','w+')
    @profile(stream=fp)
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
            tempos_aux = np.empty(ntestes, dtype = object)
            for j in range(ntestes):
                total_time = time.time()
                solverF = f(nvols, Lx, Ly, Lz)
                solverF.times["Total"] = time.time() - total_time
                tempos_aux[j] = solverF.times
            
            t1 = np.array([t["Montar Malha"] for t in tempos_aux])
            t2 = np.array([t["Montar Matriz A do TPFA"] for t in tempos_aux])
            t3 = np.array([t["Resolver o Sistema TPFA"] for t in tempos_aux])
            t4 = np.array([t["Total"] for t in tempos_aux])
            tempos[i][0][:] = [t1.min(), t1.mean(), t1.max()]
            tempos[i][1][:] = [t2.min(), t2.mean(), t2.max()]
            tempos[i][2][:] = [t3.min(), t3.mean(), t3.max()]
            tempos[i][3][:] = [t4.min(), t4.mean(), t4.max()]

            real_nvols = len(solverF.mesh.volumes)
            t_str = timedelta(seconds = time.time() - iter_time)
            str_i = "0" * (len(str(niteracoes)) - len(str(i))) + str(i)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(real_nvols))) + str(real_nvols)

            print("Iteration : {}/{}  \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
            vols.append(real_nvols)
            nvols = int(nvols * incremento)
        return tempos, vols
    
    sub = {"exemplo1D": "Fluxo Unidimensional",
           "exemplo2D": "Fluxo Bidimensional",
           "exemplo3D": "Fluxo Tridimensional",
           "exemploAleatorio": "Fluxo Aleatório"}
    c = ["black", "blue", "green", "red", "gray", "purple"]

    fig, ax = plt.subplots(2, 2, figsize = (11, 6))
    plt.suptitle("Tempo x Número de Elementos")
    for i in range(len(f)):
        start_time = time.time()
        tempos, vols = run_iterations(f[i])
        tempos = tempos[1:]
        vols = vols[1:]

        
        #ax[0][0].plot(vols, tempos[:, 3, 0], linestyle = "dotted", color = c[i])
        ax[0][0].plot(vols, tempos[:, 3, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__])
        #ax[0][0].plot(vols, tempos[:, 3, 2], linestyle = "dashed", color = c[i])
        ax[0][0].set_xlabel("Número de elementos")
        ax[0][0].set_ylabel("Tempo (s)")
        ax[0][0].title.set_text("Total")

        #ax[0][1].plot(vols, tempos[:, 0, 0], linestyle = "dotted", color = c[i])
        ax[0][1].plot(vols, tempos[:, 0, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__])
        #ax[0][1].plot(vols, tempos[:, 0, 2], linestyle = "dashed", color = c[i])
        ax[0][1].set_xlabel("Número de elementos")
        ax[0][1].set_ylabel("Tempo (s)")
        ax[0][1].title.set_text("Montar Malha")

        #ax[1][0].plot(vols, tempos[:, 1, 0], linestyle = "dotted", color = c[i])
        ax[1][0].plot(vols, tempos[:, 1, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__])
        #ax[1][0].plot(vols, tempos[:, 1, 2], linestyle = "dashed", color = c[i])
        ax[1][0].set_xlabel("Número de elementos")
        ax[1][0].set_ylabel("Tempo (s)")
        ax[1][0].title.set_text("Montar Matriz A")

        #ax[1][1].plot(vols, tempos[:, 2, 0], linestyle = "dotted", color = c[i])
        ax[1][1].plot(vols, tempos[:, 2, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__])
        #ax[1][1].plot(vols, tempos[:, 2, 2], linestyle = "dashed", color = c[i])
        ax[1][1].set_xlabel("Número de elementos")
        ax[1][1].set_ylabel("Tempo (s)")
        ax[1][1].title.set_text("Resolver Sistema")

        
        #plt.show()
        print("Total Time for {}: \t {}".format(f[i].__name__, timedelta(seconds = time.time() - start_time)))
    for i in range(2):
        for j in range(2):
            ax[i][j].grid()
            ax[i][j].legend(loc = "upper left")
    plt.tight_layout()
    plt.savefig("./tmp/testes_tempo.png")
    fp.close()

def testes_precisao(solutions, K_vols, Lx, Ly, Lz):
    # Solutions = tuple([p0, g0, q0], [p1, g1, q1], ...)
    nvols0 = 100
    nvolsMAX = int(2e4)
    incremento = 1.7
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

        
        for i in range(niteracoes):
            iter_time = time.time()

            total_time = time.time()
            solver = f(nvols, pa, ga, qa, K_vols, Lx, Ly, Lz)
            solver.times["Total"] = time.time() - total_time

            p_a = solver.mesh.an_sol[:]
            error = np.sqrt(np.sum((p_a - solver.p)**2 * solver.volumes) / np.sum((p_a**2) * solver.volumes))
            print("Error:", error)
            #print(solver.times)
            #print(np.mean(abs(solver.p - p_a)))
            erro_medio.append(error)
            precisao_media.append(1. - error)
            #print(solver.times)
            
            real_nvols = len(solver.mesh.volumes)
            vols.append(real_nvols)

            t_str = timedelta(seconds = time.time() - iter_time)
            str_i = "0" * (len(str(niteracoes)) - len(str(i + 1))) + str(i + 1)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(real_nvols))) + str(real_nvols)

            print("Iteration : {}/{}  \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
            nvols = int(nvols * incremento)

        return precisao_media, erro_medio, vols
    
    fig, ax = plt.subplots(len(solutions), figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas")
    c = ["black", "blue"]
    for i, solution in enumerate(solutions):
        start_time = time.time()
        precisao, erro_medio, vols = run_iterations(exemploAnalitico, solution[0], solution[1], solution[2], K_vols)
        ax[i].plot(vols, erro_medio, label = "I²rel", color = c[i % len(c)], marker = "o")
        ax[i].set_title("Função {}".format(i))
        ax[i].set_xlabel("Número de volumes")
        ax[i].set_ylabel("I²rel")
        ax[i].grid()
        ax[i].legend()
        print("Total Time : \t {}".format(timedelta(seconds = time.time() - start_time)))
    
    plt.tight_layout()
    plt.savefig("./tmp/testes_precisao.png")
    #plt.show()

def exemplo1(order, Lx, Ly, Lz, meshfile = None):
    solverA = solver.Solver()

    montar_time = time.time()
    if not meshfile:
        meshfile = create_box(Lx, Ly, Lz, order)
    meshA = solverA.create_mesh(meshfile)
    solverA.times["Montar Malha"] = time.time() - montar_time

    nvols = len(meshA.volumes)
    K_single = np.array([[1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1]]).reshape((3, 3))
    K = np.full((solverA.nvols, 3, 3), 1.)
    #K[:] = K_single
    
    q = np.zeros(nvols)
    fn = lambda x, y, z: np.full_like(x, 0.)
    fd = lambda x, y, z: 1 + np.sin(np.pi * x) * np.sin(np.pi * (y + 1/2))  * np.sin(np.pi * (z + 1/3))

    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = False, an_sol = fd)
    
    return solverA
if __name__ == '__main__':
    solver.verbose = True
    Lx, Ly, Lz = 1., 1., 1.
    order = 100
    #exemplo1D(order, Lx, Ly, Lz, "./mesh/20.h5m")
    #exemplo2D(order, Lx, Ly, Lz)
    #exemplo3D(order, Lx, Ly, Lz)
    #exemploAleatorio(order, Lx, Ly, Lz)
    #exemplo1(order, Lx, Ly, Lz)
    #exit()
    K_vols = 1.
    pa1 = lambda x, y, z: np.sin(x) + np.cos(y) + np.exp(z)
    ga1 = lambda x, y, z: np.array([np.cos(x), -np.sin(y), np.exp(z)]).T
    qa1 = lambda x, y, z: -K_vols * (-np.sin(x) - np.cos(y) + np.exp(z))

    pa2 = lambda x, y, z: x**2 + y**2 + z**2
    ga2 = lambda x, y, z: np.array([2*x, 2*y, 2*z]).T
    qa2 = lambda x, y, z: -K_vols * (2) * np.ones_like(x)

    pa3 = lambda x, y, z: x + y + z
    ga3 = lambda x, y, z: np.ones(shape = (len(x), 3))
    qa3 = lambda x, y, z: -K_vols * (0)
    
    #exemploAnalitico(order, pa1, ga1, qa1, K_vols, Lx, Ly, Lz)
    testes_tempo(Lx, Ly, Lz)
    #testes_precisao([(pa1, ga1, qa1), (pa2, ga2, qa2), (pa3, ga3, qa3)], K_vols, Lx, Ly, Lz)
    exit()
    meshfiles = ["./mesh/20.h5m", "./mesh/30.h5m"] #,"./mesh/40.h5m", "./mesh/50.h5m", "./mesh/60.h5m", "./mesh/80.h5m"]
    sz = {meshfiles[0]: (20, 20, 20),
          meshfiles[1]: (30, 30, 30)
          #meshfiles[2]: (40, 40, 40),
          #meshfiles[3]: (50, 50, 50),
          #meshfiles[4]: (60, 60, 60),
          #meshfiles[5]: (80, 80, 80)
          }
    for meshfile in meshfiles:
        Lx, Ly, Lz = sz[meshfile]
        precisao_media = []
        erro_medio = []
        nvols = 0
        solverE = exemploAnalitico(nvols, pa2, ga2, qa2, K_vols, Lx, Ly, Lz, meshfile)
        p_a = solverE.mesh.an_sol[:]
        error = np.sqrt(np.sum((p_a - solverE.p)**2 * solverE.volumes) / np.sum((p_a**2) * solverE.volumes))
        
        print(len(solverE.mesh.volumes))
        print(error)
        print(np.mean(abs(solverE.p - p_a)))
        