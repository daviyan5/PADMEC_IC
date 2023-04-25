import numpy as np
import time
import os
import math
import solver
import matplotlib.pyplot as plt
from datetime import timedelta

def exemplo1D(nx, ny, nz):

    Lx, Ly, Lz = 10, 10, 0.01
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

    fd = lambda x, y, z: np.where(x == 0., d1, np.where(x == Lx, d2, None))
    fn = lambda x, y, z: np.zeros_like(x)

    a_p = lambda x, y, z : np.where(x < Lx/2, d1 - (d1 - d2) * (x / (Lx/2)) * (v1/(v1 + v2)), 
                                             (d1 + (d1 - d2) * (v2 - v1)/(v1 + v2)) - (d1 - d2) * (x / (Lx/2)) * (v2/(v1+v2)))
                                
    p = solver1D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False, an_sol=a_p)

    x = mesh.volumes.x[:nx]
    
    
    
    if solver.verbose:
        plt.plot(x, p[:nx], label = "TPFA")
        plt.title("Solução 1D - Caso Linear com {} elementos".format(nvols))
        plt.plot(x, a_p(x,None,None), label = "Analítico")
        plt.plot(x, np.abs(a_p(x, None, None) - p[:nx]), label = "Erro")
        plt.text(0.7, 0.7, "Erro máximo: {}".format(np.max(np.abs(a_p(x, None, None) - p[:nx]))))
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

def exemploAnalitico(nx, ny, nz, pa, ga, qa, K_vols):
    # P = sin(x) + cos(y) + sen(x)
    #grad(p) = (cos(x), -sin(y), cos(z))
    # div(K * grad(p)) = K * (-sen(x) - cos(y) - sen(z))
    
    Lx, Ly, Lz = 20, 20, 20
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    solverA = solver.Solver()
    meshA = solverA.create_mesh(nx, ny, nz, dx, dy, dz, name = "exemploAnalitico")
    nvols = meshA.nvols
    K = solver.get_tensor(K_vols, (nz, ny, nx))
    
    normal = meshA.faces.normal[meshA.faces.boundary]

    q = qa(meshA.volumes.x, meshA.volumes.y, meshA.volumes.z) * meshA.volume
    
    fd = lambda x, y, z: pa(x, y, z)
    fn = lambda x, y, z: (normal * ga(x, y, z) * -K_vols).sum(axis = 1)
    #np.zeros_like(x)#
    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = solver.verbose, an_sol=pa)

    #if nvols == 8000:
    #    np.savetxt('d_nodes.txt', solverA.d_nodes.astype('int'))
    #    np.savetxt('n_nodes.txt', solverA.n_nodes.astype('int'))
    #    np.savetxt('d_vals.txt', solverA.d_values)
    #    np.savetxt('n_vals.txt', solverA.n_values)
    if solver.verbose:
        # Plot 3D solution
        x, y, z = meshA.volumes.x, meshA.volumes.y, meshA.volumes.z
        fig = plt.figure()
        fig.suptitle('Erro Médio Quadrático: ' + str(np.mean((p - pa(x, y, z)) ** 2)))
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
        

        plt.show() if nvols < 100000 else None

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
    nvolsMAX = int(1e4)
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

            
            nx, ny, nz = int(nvols ** (1/3)), int(nvols ** (1/3)), int(nvols ** (1/3))
            tempos_aux = np.empty(ntestes, dtype = object)
            mesh, solver = f(nx, ny, nz, pa, ga, qa, K_vols)

            p_a = pa(mesh.volumes.x, mesh.volumes.y, mesh.volumes.z)
            error = np.sqrt(np.sum(((p_a - solver.p)**2) * mesh.volumes.volumes) / np.sum((p_a**2) * mesh.volumes.volumes))

            #print(error)
            #print(np.mean(abs(solver.p - p_a)))

            real_vols = mesh.nvols
            erro_medio.append(error)
            precisao_media.append(1. - error)
            vols.append(real_vols)

            
            t_str = timedelta(seconds = time.time() - iter_time)
            str_i = "0" * (len(str(niteracoes)) - len(str(i + 1))) + str(i + 1)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(real_vols))) + str(real_vols)

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
        ax[i].set_yscale("log")
        ax[i].grid()
        ax[i].legend()
        print("Total Time : \t {}".format(timedelta(seconds = time.time() - start_time)))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    solver.verbose = True
    
    exemplo1D(9, 9, 1)
    #exemplo2D(300, 300, 1)
    #exemplo3D(60, 60, 60)
    #exemploAleatorio(50,50,50)
    #testes_tempo()

    K_vols = 1.
    pa1 = lambda x, y, z: np.sin(x) + np.cos(y) + np.exp(z)
    ga1 = lambda x, y, z: np.array([np.cos(x), -np.sin(y), np.exp(z)]).T
    qa1 = lambda x, y, z: -K_vols * (-np.sin(x) - np.cos(y) + np.exp(z))
    pa2 = lambda x, y, z: x**2 + y**2 + z**2
    ga2 = lambda x, y, z: np.array([2*x, 2*y, 2*z]).T
    qa2 = lambda x, y, z: -K_vols * (2) * np.ones_like(x)


    #exemploAnalitico(50, 50, 50, pa1, ga1, qa1, K_vols)
    #print("Testes de Precisão")
    #testes_precisao([(pa1, ga1, qa1), (pa2, ga2, qa2)], K_vols)
    