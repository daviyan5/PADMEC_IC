import numpy as np
import time
import sys
import os
import math
import solver
import re
import objgraph
import gc
import matplotlib.pyplot as plt
from scipy.integrate import tplquad
from mesh_gen import create_box, legacy_create_box
from datetime import timedelta
from memory_profiler import profile

def exemplo1D(order, Lx, Ly, Lz, meshfile = None):

    solver1D = solver.Solver()
    montar_time = time.time()
    if meshfile is None:
        meshfile = create_box(Lx, Ly, Lz, order)
    mesh = solver1D.create_mesh(meshfile)
    solver1D.times["Pré-Processamento"] = time.time() - montar_time

    nvols = len(mesh.volumes)

    v1, v2 = 1., 1.
    d1, d2 = 100., 30.
    vq = 0.

    x = mesh.volumes.center[:][:, 0]
    K = solver.get_tensor(v1, (nvols))
    K_left = solver.get_tensor(v2, ())
    K[x < Lx / 2] = K_left
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(x == 0., d1, np.where(x == Lx, d2, None))
    fn = lambda x, y, z: np.where(y == 0., 0., np.where(y == Ly, 0., None))

    a_p = lambda x, y, z : np.where(x < Lx/2, d1 - (d1 - d2) * (x / (Lx/2)) * (v1/(v1 + v2)), 
                                             (d1 + (d1 - d2) * (v2 - v1)/(v1 + v2)) - (d1 - d2) * (x / (Lx/2)) * (v2/(v1+v2)))
    
    
    solver1D.times["Total"] = time.time()   
    p = solver1D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False, an_sol=a_p, name = "1D")
    solver1D.times["Total"] = time.time() - solver1D.times["Total"] + solver1D.times["Pré-Processamento"]
    del fd, fn, mesh, meshfile
    return solver1D

def exemplo2D(order, Lx, Ly, Lz, meshfile = None):

    solver2D = solver.Solver()

    montar_time = time.time()
    if meshfile is None:
        meshfile = create_box(Lx, Ly, Lz, order)
    mesh = solver2D.create_mesh(meshfile)
    solver2D.times["Pré-Processamento"] = time.time() - montar_time

    nvols = len(mesh.volumes)

    v1 = 1.
    d1, d2, d3 = 0., 100., 0.
    vq = 0.

    K = solver.get_tensor(v1, (nvols))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, None)))
    fn = lambda x, y, z: np.where(x == 0, 0, None)

    solver2D.times["Total"] = time.time()
    p = solver2D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False, name = "2D")
    solver2D.times["Total"] = time.time() - solver2D.times["Total"] + solver2D.times["Pré-Processamento"]
    del fd, fn, mesh, meshfile
    return solver2D

def exemplo3D(order, Lx, Ly, Lz, meshfile = None):

    solver3D = solver.Solver()

    montar_time = time.time()
    if meshfile is None:
        meshfile = create_box(Lx, Ly, Lz, order)
    mesh = solver3D.create_mesh(meshfile)
    solver3D.times["Pré-Processamento"] = time.time() - montar_time

    nvols = len(mesh.volumes)

    v1 = 1.
    d1, d2, d3, d4 = 0., 100., 10., 40.
    n1, n2 = 0, 0
    vq = 0.

    K = solver.get_tensor(v1, (nvols))
    q = np.full(nvols, vq)

    fd = lambda x, y, z: np.where(y == Ly, d1, np.where(y == 0, d2, np.where(x == Lx, d3, np.where(z == 0, d4, None))))
    fn = lambda x, y, z: np.where(x == 0, n1, np.where(z == Lz, n2, None))

    solver3D.times["Total"] = time.time()
    p = solver3D.solve(mesh, K, q, fd, fn, create_vtk = solver.verbose, check = False, name = "3D")
    solver3D.times["Total"] = time.time() - solver3D.times["Total"] + solver3D.times["Pré-Processamento"]
    del fd, fn, mesh, meshfile
    return solver3D

def exemploAleatorio(order, Lx, Ly, Lz, meshfile = None):

    solverA = solver.Solver()

    montar_time = time.time()
    if meshfile is None:
        meshfile = create_box(Lx, Ly, Lz, order)
    meshA = solverA.create_mesh(meshfile)
    solverA.times["Pré-Processamento"] = time.time() - montar_time

    nvols = len(meshA.volumes)
    a, b = np.random.uniform(1, 100, 2)
    K = solver.get_random_tensor(a, b, size = (nvols))
    q = np.random.uniform(0, 100, nvols)
    fd_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fn_v = np.random.choice(np.append([None] * 10, np.random.uniform(1, 100, 3)), size = len(meshA.faces.boundary))
    fd = lambda x, y, z: fd_v
    fn = lambda x, y, z: fn_v

    solverA.times["Total"] = time.time()
    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = False)
    solverA.times["Total"] = time.time() - solverA.times["Total"] + solverA.times["Pré-Processamento"]
    del fd, fn, mesh, meshfile
    return solverA

def exemploAnalitico(order, pa, ga, qa, K_vols, Lx, Ly, Lz, meshfile = None, factor = 1.5):
    solverA = solver.Solver()

    montar_time = time.time()
    if meshfile is None:
        meshfile = create_box(Lx, Ly, Lz, order, factor = factor)
    meshA = solverA.create_mesh(meshfile)
    solverA.times["Pré-Processamento"] = time.time() - montar_time
    
    nvols = len(meshA.volumes)
    K = solver.get_tensor(K_vols, (nvols))
    boundary_faces = meshA.faces.boundary[:]
    normal, vL, vR = solverA._get_normal(boundary_faces)
    normal = np.abs(normal) / np.linalg.norm(normal, axis = 1).reshape((len(normal), 1))
    
    coords = meshA.volumes.center[:]
    xv, yv, zv = coords[:, 0], coords[:, 1], coords[:, 2]

    q = qa(xv, yv, zv, K) * solverA.volumes
    fd = lambda x, y, z: pa(x, y, z)
    fn = lambda x, y, z: (np.einsum("ij,ij->i", normal , -ga(x, y, z)))
    #fd = lambda x, y, z: np.zeros_like(x)
    #fn = lambda x, y, z: np.zeros_like(x)

    
    p = solverA.solve(meshA, K, q, fd, fn, create_vtk = solver.verbose, check = solver.verbose, an_sol=pa)
    
    if solver.verbose:
        # Plot 3D solution
        
        plt.axis('scaled')
        fig = plt.figure(figsize = (10,10))
        p_a = pa(xv,yv,zv)
        error = np.sum((p_a - solverA.p)**2 * solverA.volumes) / np.sum((p_a**2) * solverA.volumes)
        print(str(np.sqrt(error)))
        print(solverA.nvols)
        fig.suptitle('Erro Médio Quadrático: ' + str(np.sqrt(error)))
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(xv, yv, zv, c = p, cmap = 'jet')
        # Add Values on the plot
        for i in range(len(xv)):
            ax.text(xv[i], yv[i], zv[i], str(np.round(p[i], 5)), size = 5, zorder = 1, color = 'k')
        ax.title.set_text('Solução numérica')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Plot analytical solution
        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(xv, yv, zv, c = pa(xv, yv, zv), cmap = 'jet')
        # Add Values on the plot
        for i in range(len(xv)):
            ax.text(xv[i], yv[i], zv[i], str(np.round(pa(xv[i], yv[i], zv[i]), 5)), size = 5, zorder = 1, color = 'k')

        ax.title.set_text('Solução analítica')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Plot normal vectors 
        ax = fig.add_subplot(223, projection='3d')
        coords = meshA.faces.center[solverA.internal_faces]
        xf, yf, zf = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.quiver(xf, yf, zf, solverA.N[:, 0], solverA.N[:, 1], solverA.N[:, 2], length = 0.1, normalize = False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.title.set_text('Vetores normais internos')
        
        
        # Plot normal vectors 
        ax = fig.add_subplot(224, projection='3d')
        coords = meshA.faces.center[boundary_faces]
        xf, yf, zf = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.quiver(xf, yf, zf, normal[:, 0], normal[:, 1], normal[:, 2], length = 0.1, normalize = False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.title.set_text('Vetores normais superficie')

        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.title.set_text('Fluxo volumes númerico')

        

        plt.tight_layout()
        index = np.random.randint(0,10)
        index = 0
        plt.show()
        plt.savefig("teste{}.png".format(index))
        plt.clf()
        #plt.show() if nvols < 10000 else None

    return solverA



def testes_tempo(Lx, Ly, Lz):
    print("RUNNING TIME TESTS")
    
    nvols0, nvolsMAX, incremento, ntestes, niteracoes = get_tests_args()
    tempos = np.zeros(ntestes)
    
    factor = 0.8
    solver.verbose = False

    f = [exemplo1D, exemplo2D, exemplo3D]
    def run_iterations(f):
        tempos = np.empty((niteracoes, 4, 3))
        # Tempos[i][0] = [MIN, MEDIO, MAX] -> Pré-Processamento
        # Tempos[i][1] = [MIN, MEDIO, MAX] -> Montar Matriz
        # Tempos[i][2] = [MIN, MEDIO, MAX] -> Resolver Sistema
        # Tempos[i][3] = [MIN, MEDIO, MAX] -> Total
        nvols = nvols0
        vols = []
       

        iter_time = time.time()
        dim = 3 if f == exemplo3D else 2 
        for i in range(niteracoes):
            mesh_create_time = time.time()
            meshfile = create_box(Lx, Ly, Lz, nvols, factor = factor)
            mesh_args = get_args_mesh(meshfile)
            mesh_create_time = time.time() - mesh_create_time
            tempos_aux = np.empty(ntestes, dtype = object)
        
        
            for j in range(ntestes):
                solverF = f(nvols, Lx, Ly, Lz, mesh_args)
                solverF.times["Pré-Processamento"] += mesh_create_time
                tempos_aux[j] = solverF.times
                t_str = timedelta(seconds = solverF.times["Total"])
                s = "Test {} of {}: {}".format(j + 1, ntestes, t_str)
                real_nvols = solverF.nvols
                if len(vols) > 0 and real_nvols == vols[-1]:
                    print("ERROR")
                    exit(1)
                #objgraph.show_backrefs([solverF.mesh], filename='sample-mesh.png')_
                #objgraph.show_refs([solverF], filename='sample-graph.png')
                del solverF
                gc.collect()
                print(s, end = "\r")
            
            print(" " * len(s), end = "\r")
            t1 = np.array([t["Pré-Processamento"] for t in tempos_aux])
            t2 = np.array([t["Montar Sistema TPFA"] for t in tempos_aux])
            t3 = np.array([t["Resolver o Sistema TPFA"] for t in tempos_aux])
            t4 = np.array([t["Total"] for t in tempos_aux])
            tempos[i][0][:] = [t1.min(), t1.mean(), t1.max()]
            tempos[i][1][:] = [t2.min(), t2.mean(), t2.max()]
            tempos[i][2][:] = [t3.min(), t3.mean(), t3.max()]
            tempos[i][3][:] = [t4.min(), t4.mean(), t4.max()]

            
            t_str = timedelta(seconds = time.time() - iter_time)
            str_i = "0" * (len(str(niteracoes)) - len(str(i + 1))) + str(i + 1)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(real_nvols))) + str(real_nvols)

            print("Iteration : {}/{}  \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
            vols.append(real_nvols)
            nvols = get_next_nvols(nvols, incremento)
        return tempos, vols
    
    sub = {"exemplo1D": "Fluxo 1D",
           "exemplo2D": "Fluxo 2D",
           "exemplo3D": "Fluxo 3D",
           "exemploAleatorio": "Fluxo Aleatório"}
    c = ["black", "blue", "green", "red", "gray", "purple"]
    marker = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d", "P", "X"]
    fig, ax = plt.subplots(2, 2, figsize = (11, 6))
    plt.suptitle("Tempo x Número de Volumes")
    for i in range(len(f)):
        start_time = time.time()
        tempos, vols = run_iterations(f[i])
        tempos = tempos[1:]
        vols = vols[1:]

        
        #ax[0][0].plot(vols, tempos[:, 3, 0], linestyle = "dotted", color = c[i])
        ax[0][0].plot(vols, tempos[:, 3, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__], marker = marker[i])#o")
        #ax[0][0].plot(vols, tempos[:, 3, 2], linestyle = "dashed", color = c[i])
        ax[0][0].set_xlabel("Número de Volumes")
        ax[0][0].set_ylabel("Tempo (s)")
        ax[0][0].title.set_text("Total")

        #ax[0][1].plot(vols, tempos[:, 0, 0], linestyle = "dotted", color = c[i])
        ax[0][1].plot(vols, tempos[:, 0, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__], marker = marker[i])#v")
        #ax[0][1].plot(vols, tempos[:, 0, 2], linestyle = "dashed", color = c[i])
        ax[0][1].set_xlabel("Número de Volumes")
        ax[0][1].set_ylabel("Tempo (s)")
        ax[0][1].title.set_text("Pré-Processamento")

        #ax[1][0].plot(vols, tempos[:, 1, 0], linestyle = "dotted", color = c[i])
        ax[1][0].plot(vols, tempos[:, 1, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__], marker = marker[i])#s")
        #ax[1][0].plot(vols, tempos[:, 1, 2], linestyle = "dashed", color = c[i])
        ax[1][0].set_xlabel("Número de Volumes")
        ax[1][0].set_ylabel("Tempo (s)")
        ax[1][0].title.set_text("Montar Sistema TPFA")

        #ax[1][1].plot(vols, tempos[:, 2, 0], linestyle = "dotted", color = c[i])
        ax[1][1].plot(vols, tempos[:, 2, 1], linestyle = "solid", color = c[i], label = sub[f[i].__name__], marker = marker[i])#p")
        #ax[1][1].plot(vols, tempos[:, 2, 2], linestyle = "dashed", color = c[i])
        ax[1][1].set_xlabel("Número de Volumes")
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
    plt.clf()

def testes_precisao(solutions, K_vols, Lx, Ly, Lz, names):
    # Solutions = tuple([p0, g0, q0], [p1, g1, q1], ...)
    print("RUNNING PRECISION TESTS")

    nvols0, nvolsMAX, incremento, ntestes, niteracoes = get_tests_args()

    solver.verbose = False

    def run_iterations(f, pa, ga, qa, K_vols, ia = None):
        
        nvols = nvols0
        vols = []
        precisao_media = []
        erro_medio = []

        
        for i in range(niteracoes):
            iter_time = time.time()

            total_time = time.time()
            solver = f(nvols, pa, ga, qa, K_vols, Lx, Ly, Lz)
            solver.times["Total"] = time.time() - total_time
 
            # We need to integrate p_a over the volume
            # How:
            # 1. Find dx, dy, dz for each volume
            # 2. Get the coordinates of the point P of each volume, for (xc, yc, zc) in each volume:
            #    P = (xc - dx/2, yc - dy/2, zc - dz/2)
            # 3. Integrate p_a(P) * dx * dy * dz from P[0] to P[0] + dx, P[1] to P[1] + dy, P[2] to P[2] + dz
            coords = solver.mesh.volumes.center[:]
            xv, yv, zv = coords[:, 0], coords[:, 1], coords[:, 2]
            
            faces_coords = solver.mesh.faces.center[solver.faces_by_volume.flatten()].reshape(solver.faces_by_volume.shape + (3,))
            new_coords = coords[:, np.newaxis, :].repeat(solver.faces_by_volume.shape[1], axis = 1)

            distances = np.abs(faces_coords - new_coords).sum(axis = 1)
            P = coords - distances / 2 
            dx, dy, dz = distances[:, 0], distances[:, 1], distances[:, 2]

            
            # lamban_sol = lambda z, y, x: solver.an_sol(x, y, z)

            # for i in range(P.shape[0]):
            #     res = tplquad(lamban_sol, P[i][0], P[i][0] + dx[i], P[i][1], P[i][1] + dy[i], P[i][2], P[i][2] + dz[i])[0]
            #     res_list.append(res)
            p_a = solver.an_sol(xv, yv, zv) 
            p_n = solver.p 

            if ia is not None:
                res_list = ia(P[:, 0], P[:, 0] + dx, P[:, 1], P[:, 1] + dy, P[:, 2], P[:, 2] + dz)
                p_a  = np.array(res_list) / solver.volumes
                # print(p_a)
                # print("PN")
                # print(p_n)
                # input()
           
            
            
            error = (np.sum(((p_a - p_n) ** 2) *  solver.volumes) / np.sum((p_a ** 2) * solver.volumes))
            
            print("Error:", np.sqrt(error))
            #print(solver.times)
            #print(np.mean(abs(solver.p - p_a)))
            erro_medio.append(np.sqrt(error))
            precisao_media.append(1. - np.sqrt(error))
            #print(solver.times)
            
            real_nvols = solver.nvols
            if len(vols) > 0 and real_nvols == vols[-1]:
                print("ERROR")
                exit(1)
            vols.append(real_nvols)
           

            t_str = timedelta(seconds = time.time() - iter_time)
            str_i = "0" * (len(str(niteracoes)) - len(str(i + 1))) + str(i + 1)
            str_nvols = "0" * (len(str(nvolsMAX)) - len(str(real_nvols))) + str(real_nvols)

            print("Iteration : {}/{}  \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
            nvols = get_next_nvols(nvols, incremento)
            del solver
            gc.collect()

        return precisao_media, erro_medio, vols
    
    n = int(np.ceil(len(solutions) / 2))
    m = int(np.ceil(len(solutions) / n))
    fig, ax = plt.subplots(n, m, figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas")
    c = ["black", "blue", "green", "red", "grey"]
    for i, solution in enumerate(solutions):
        start_time = time.time()
        precisao, erro_medio, vols = run_iterations(exemploAnalitico, solution[0], solution[1], solution[2], K_vols, ia = solution[3] if len(solution) > 3 else None)
        a, b = i // m, i % m
        vols = np.array(vols)

        ax[a][b].set_xscale('log')
        ax[a][b].set_yscale('log')
        ratio = ax[a][b].get_data_ratio()
        print("ratio: ", ratio)
        tgalpha = ratio * (np.log10(erro_medio[0]) - np.log10(erro_medio[-1])) / (np.log10(vols[-1]) - np.log10(vols[0]))
        print("tgalpha: ", tgalpha)
        ax[a][b].plot(vols, erro_medio, label = "I²rel", color = c[i % len(c)], marker = "p")

        #ratio = ax[a][b].get_data_ratio()
        #scale_vols = erro_medio[0] * 10**(-2 * ratio * (np.log10(vols) - np.log10(vols[0]))) 
        scale_vols = (erro_medio[0] * (vols[0] ** 2)) / vols ** 2
        if i != 0:
            ax[a][b].plot(vols, scale_vols, label = "O(n²)", color = 'purple', linestyle = "--")
        else:
            ax[a][b].set_ylim(1e-18, 1e-14)
        ax[a][b].set_title(names[i])
        ax[a][b].set_xlabel("Número de Volumes")
        ax[a][b].set_ylabel("I²rel")
        ax[a][b].grid()
        ax[a][b].legend()
        print("Total Time : \t {}".format(timedelta(seconds = time.time() - start_time)))
    
    plt.tight_layout()
    plt.savefig("./tmp/testes_precisao.png")
    plt.clf()
    #plt.show()

def testes_memoria(Lx, Ly, Lz):
    print("RUNNING MEMORY TESTS")

    nvols0, nvolsMAX, incremento, ntestes, niteracoes = get_tests_args()

    solver.verbose = False
    
    nvols = nvols0
    vols = []
    memory = []
    for i in range(niteracoes):
        meshfile = create_box(Lx, Ly, Lz, nvols)
        mesh_args = get_args_mesh(meshfile)
        tempos_aux = np.empty(ntestes, dtype = object)
        log_filename = 'temp_memory.log'
        fp = open(log_filename,'w+')
        
        s = ""
        @profile(stream=fp)
        def run_tests(j, s):
            total_time = time.time()
            solverF = exemplo3D(nvols, Lx, Ly, Lz, mesh_args)
            t_str = timedelta(seconds = time.time() - total_time)
            s = "Test {} of {}: {}".format(j + 1, ntestes, t_str)
            print(s, end = "\r")
            return solverF, s
        
        iter_time = time.time()
        for j in range(ntestes):
            solverF, s = run_tests(j, s)
            real_nvols = solverF.nvols
            del solverF
            gc.collect()
        t_str = timedelta(seconds = time.time() - iter_time)

        print(" " * len(s), end = "\r")
        fp.close()

        # pattern to match the memory usage
        memory_aux = []
        with open(log_filename, 'r+') as f:
            for line in f:
                #print(line)
                #print("AAA")
                if 'solverF = exemplo3D(nvols, Lx, Ly, Lz, mesh_args)' in line:
                    match = re.findall(r'\b(\d+\.\d+\s*(?:MiB|GiB))\b', line)
                    #print(match)
                    if match:
                        size_str = match[1]
                        size_str = size_str.strip()
                        if size_str.endswith('MiB'):
                            size_bytes = int(float(size_str[:-3]) * 1024 * 1024)
                        elif size_str.endswith('GiB'):
                            size_bytes = int(float(size_str[:-3]) * 1024 * 1024 * 1024)
                        else:
                            size_bytes = int(size_str)
                        memory_aux.append(size_bytes)
                        print(f'{size_str} = {size_bytes} bytes')
            f.truncate(0)
        str_i = "0" * (len(str(niteracoes)) - len(str(i + 1))) + str(i + 1)
        str_nvols = "0" * (len(str(nvolsMAX)) - len(str(real_nvols))) + str(real_nvols)

        if len(vols) > 0 and real_nvols == vols[-1]:
            print("ERROR")
            exit(1)
        
        print("Iteration : {}/{} \t Vols: {} \t Total Time : {}".format(str_i, niteracoes, str_nvols, t_str))
        vols.append(real_nvols)
        memory.append(np.mean(memory_aux))
        nvols = get_next_nvols(nvols, incremento)
    plt.grid()
    plt.xlabel("Número de Volumes")
    plt.ylabel("Memória Alocada pela Simulação (MBs)")
    plt.plot(vols, np.array(memory) / 1e6, label = "Memória por Número de Volumes", marker = "p", color = "blue")
    plt.tight_layout()
    plt.legend()
    plt.savefig("./tmp/testes_memoria.png")
    plt.clf()

def get_args_mesh(meshfile, dim = 3):
    aux_solver = solver.Solver()
    aux_mesh = aux_solver.create_mesh(meshfile, dim = dim)
    return np.array([meshfile, aux_solver.areas, aux_solver.volumes], dtype=object)

def get_tests_args():
    nvols0 = 4 ** 3
    nvolsMAX = 19 ** 3
    incremento = 1.1
    ntestes = 10
    niteracoes = int((nvolsMAX ** (1/3) - nvols0 ** (1/3) + 1)) 

    return nvols0, nvolsMAX, incremento, ntestes, niteracoes

def get_next_nvols(nvols, incremento):
    return int(np.ceil((nvols ** (1/3)) + incremento) ** 3)

if __name__ == '__main__':
    solver.verbose = True
    Lx, Ly, Lz = 600., 0.2, 0.2
    order = 1
    factor = 1
    #exemplo1D(order, Lx, Ly, Lz)
    #exemplo2D(order, Lx, Ly, Lz)
    #exemplo3D(order, Lx, Ly, Lz)
    #exemploAleatorio(order, Lx, Ly, Lz)
    #exemplo1(order, Lx, Ly, Lz)
    #exit()
    ex, ey, ez = 1., 1., 1.
    K = solver.get_tensor(1., ())
    K[:, 0] = np.array([ex, 0, 0])
    K[:, 1] = np.array([0, ey, 0])
    K[:, 2] = np.array([0, 0, ez])

    names = ["x + y + z", "sin(x) + cos(y) + exp(z)", "x³ + y³ + z³", "xln(x) + 1/(y+1) + z²"]
    pa1 = lambda x, y, z:       x + y + z
    ga1 = lambda x, y, z:       np.array([np.ones_like(x), np.ones_like(y), np.ones_like(z)]).T
    qa1 = lambda x, y, z, K:    0

    pa2 = lambda x, y, z:       np.sin(x) + np.cos(y) + np.exp(z) # sin(x) + cos(y) + exp(z)
    ga2 = lambda x, y, z:       np.array([np.cos(x), -np.sin(y), np.exp(z)]).T
    qa2 = lambda x, y, z, K:    -(-K[:,0,0] * np.sin(x) - K[:,1,1] * np.cos(y) + K[:,2,2] * np.exp(z))

    pa3 = lambda x, y, z:       x**3 + y**3 + z**3 # x³ + y³ + z³
    ga3 = lambda x, y, z:       np.array([3*(x**2), 3*(y**2), 3*(z**2)]).T
    qa3 = lambda x, y, z, K:    -6 * (K[:,0,0] * x + K[:,1,1] * y + K[:,2,2] * z)
    ia3 = lambda x0, x1, y0, y1, z0, z1: 1/4 * ((x1**4 - x0**4) * (y1 - y0) * (z1 - z0) + 
                                                (y1**4 - y0**4) * (x1 - x0) * (z1 - z0) +
                                                (z1**4 - z0**4) * (x1 - x0) * (y1 - y0))
    pa4 = lambda x, y, z:       (x + 1) * np.log(1 + x) + 1/(y + 1) + z**2 # xln(x) + 1/(y+1) + z²
    ga4 = lambda x, y, z:       np.array([np.log(1 + x) + 1,-1/(y+1)**2, 2*z]).T
    qa4 = lambda x, y, z, K:    -(K[:,0,0] * (1/(x+1)) + K[:,1,1] * (2/((y+1)**3)) + K[:,2,2] * 2)

    pa5 = lambda x, y, z:       np.exp(x)
    ga5 = lambda x, y, z:       np.array([np.exp(x), 0, 0]).T
    qa5 = lambda x, y, z, K:    -(K[:,0,0] * np.exp(x) + K[:,1,1] * 0 + K[:,2,2] * 0)
    
    
    #exemploAnalitico(order, pa3, ga3, qa3, K, Lx, Ly, Lz, factor = 1)
    #exit()
    #testes_memoria(Lx, Ly, Lz)
    #testes_tempo(Lx, Ly, Lz)
    
    solutions = [(pa1, ga1, qa1), (pa2, ga2, qa2), (pa3, ga3, qa3), (pa4, ga4, qa4)]
    Lx, Ly, Lz = 1., 2., 3.
    #solutions = [(pa5, ga5, qa5)]
    testes_precisao(solutions, K, Lx, Ly, Lz, names)
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
        