import numpy as np
import matplotlib.pyplot as plt
import helpers
import tracemalloc

from solver import TPFAsolver
from mesh_generator import MeshGenerator



def run_test1D(box_dimensions : tuple, order : int, mesh_args : tuple = None, verbose : bool = False) -> dict:
    Lx, Ly, Lz = box_dimensions
    mesh, mesh_params = None, None
    if mesh_args is None:
        meshfile = MeshGenerator().create_box(box_dimensions, order)
        mesh = helpers.load_mesh(meshfile)
    else:
        mesh, mesh_params = mesh_args
    name = "1D"
    solver = TPFAsolver(verbose, verbose, name)

    v1, v2 = 1., 1.
    d1, d2 = 100., 30.
    vq = 0.

    x = mesh.volumes.center[:][:, 0]
    nvols = len(mesh.volumes)
    K = helpers.get_tensor(v1, nvols)
    K_left = helpers.get_tensor(v2, ())
    K[x < Lx / 2] = K_left
    
    def neumann(x, y, z, faces, mesh):
        return np.full(len(faces), 0.)
    
    def dirichlet(x, y, z, faces, mesh):
        return np.where(x == 0, d1, np.where(x == Lx, d2, None))
    
    def analytical(x, y, z):
        return np.where(x < Lx/2, d1 - (d1 - d2) * (x / (Lx/2)) * (v1/(v1 + v2)), 
                       (d1 + (d1 - d2) * (v2 - v1)/(v1 + v2)) - (d1 - d2) * (x / (Lx/2)) * (v2/(v1+v2)))
    
    def q(x, y, z, K):
        return np.full_like(x, vq)
    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : q,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : analytical,
            "vtk"           : True
        }
    return solver.solveTPFA(args)

def run_QFiveSpots(box_dimensions : tuple, order : int, mesh_args : tuple = None, verbose : bool = False) -> dict:
    Lx, Ly, Lz = box_dimensions
    mesh, mesh_params = None, None
    if mesh_args is None:
        meshfile = MeshGenerator().create_box(box_dimensions, order)
        mesh = helpers.load_mesh(meshfile)
    else:
        mesh, mesh_params = mesh_args
    name = "QFiveSpot"
    solver = TPFAsolver(verbose, verbose, name)

    v1, v2 = 100., 100.
    d1, d2 = 10000., 1.
    vq = 0.

    nvols = len(mesh.volumes)
    K = helpers.get_tensor(v1, nvols)
    xv, yv, zv = mesh.volumes.center[:][:, 0], mesh.volumes.center[:][:, 1], mesh.volumes.center[:][:, 2]
    xyz = np.vstack((xv, yv, zv)).T
    K_circle = helpers.get_tensor(v2, ())
    is_inside = np.linalg.norm(xyz - np.array([Lx/2, Ly/2, Lz/2]), axis = 1) < Lx / 4
    K[is_inside] = K_circle
    
    def neumann(x, y, z, faces, mesh):
        return np.full_like(x, 0.)
    
    # Baseado em área : 10% do canto inferior esquerdo e 10% do canto superior direito
    def dirichlet(x, y, z, faces, mesh):
        p1 = 0.1
        p2 = 0.1
        r1 = np.sqrt(Lx * Ly * p1 / np.pi)
        r2 = np.sqrt(Lx * Ly * p2 / np.pi)
        xy  = np.vstack((x, y)).T
        dist0 = np.linalg.norm(xy - np.array([0, 0]), axis = 1)
        dist1 = np.linalg.norm(xy - np.array([Lx, Ly]), axis = 1)

        v1   = np.where(dist0 < r1)[0]
        v2   = np.where(dist1 < r2)[0]

        dp   = np.full_like(x, np.nan)
        dp[v1] = d1
        dp[v2] = d2
        return dp
    
    
    def q(x, y, z, K):
        return np.full(nvols, vq)
    
    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : q,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : None,
            "vtk"           : False
        }
    return solver.solveTPFA(args)


def get_analytical_keys():
    return ["name", "p", "-div(K grad(p))", "dirichlet", "neumann"]

def run_analytical(box_dimensions : tuple, order : int, K : np.ndarray,  solution : dict, mesh_args : tuple = None, verbose : bool = True) -> dict:
    Lx, Ly, Lz = box_dimensions
    mesh, mesh_params = None, None
    if mesh_args is None:
        meshfile = MeshGenerator().create_box(box_dimensions, order)
        mesh = helpers.load_mesh(meshfile)
    else:
        mesh, mesh_params = mesh_args
    
    nvols = len(mesh.volumes)
    coords = mesh.volumes.center[:]
    xv, yv, zv = coords[:, 0], coords[:, 1], coords[:, 2]
    name       = solution["name"]
    analytical = solution["p"]
    q          = solution["-div(K grad(p))"]
    analytical_dirichlet = solution["dirichlet"]
    analytical_neumann   = solution["neumann"]

    
    def dirichlet_bc(x, y, z, faces, mesh):
        return analytical_dirichlet(x, y, z) 

    def neumann_bc(x, y, z, faces, mesh):
        return analytical_neumann(x, y, z)
    
    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : q,
            "neumann"       : neumann_bc,
            "dirichlet"     : dirichlet_bc,
            "analytical"    : analytical,
            "vtk"           : False
        }
    solver = TPFAsolver(verbose, verbose, name)
    return solver.solveTPFA(args)



def run_accurracy_tests(analytical_solutions : list, box_dimensions : tuple,  meshfiles : list = None) -> None:
    """
    Executa e plota testes de acurácia para as soluções analíticas passadas.
    """

    
    if meshfiles is None:                                  
        meshfiles = helpers.create_meshes(box_dimensions, num = 20, suffix = "acc")
    
    num_solutions = len(analytical_solutions)
    num_meshes = len(meshfiles)
    errors = {}
    verbose = False
    for i, meshfile in enumerate(meshfiles):
        mesh = helpers.load_mesh(meshfile)
        mesh_params = helpers.get_mesh_params(mesh)
        mesh_args = (mesh, mesh_params)
        nvols = len(mesh.volumes)
        K = helpers.get_tensor(1., nvols)
        for solution in analytical_solutions:

            helpers.verbose("Starting A-test [{}/{}] for {} volumes with solution {}.".format(i, num_meshes, nvols, solution["name"]), type_msg = "OUT")
            results = run_analytical(box_dimensions, 1, K,  solution, mesh_args, verbose = verbose)
            helpers.verbose("Total time: {:.2}s -> Error : {:.3}.".format(results["times"]["Total Time"], results["error"]), type_msg="INFO")
            sol_name = solution["name"]

            if sol_name not in errors.keys():
                errors[sol_name] = []
            
            errors[sol_name].append(np.array([nvols, results["error"]]))
    
    n = int(np.ceil(num_solutions / 2))
    m = int(np.ceil(num_solutions / n))

    fig, ax = plt.subplots(n, m, figsize = (11, 6))
    fig.suptitle("Comparação com Soluções Analíticas")
    c = ["black", "blue", "green", "red", "grey"]
    
    for i, sol_name in enumerate(errors.keys()):
        
        a, b = i // m, i % m
        ax[a][b].set_xscale('log')
        ax[a][b].set_yscale('log')
        errors[sol_name] = np.array(errors[sol_name])
        vols = errors[sol_name][:, 0]
        erro = errors[sol_name][:, 1]
        ratio = ax[a][b].get_data_ratio()
        tgalpha = ratio * ((np.log10(erro[0]) - np.log10(erro[-1])) /
                           (np.log10(vols[-1]) - np.log10(vols[0])))
        print("tgalpha: ", tgalpha)
        ax[a][b].plot(vols, erro, label = "I²rel", color = c[i % len(c)], marker = "p")

        scale_vols = (erro[0] * (vols[0] ** 2)) / vols ** 2
        if i > 1:
            ax[a][b].plot(vols, scale_vols, label = "O(n²)", color = 'purple', linestyle = "--")
        else:
            ax[a][b].set_ylim(1e-20, 1e-10)
        
        ax[a][b].set_title(sol_name)
        ax[a][b].set_xlabel("Número de Volumes")
        ax[a][b].set_ylabel("I²rel")
        ax[a][b].grid()
        ax[a][b].legend()
    
    plt.tight_layout()
    plt.savefig("./tests/accuracy_tests/accuracy.png")
    plt.clf()

def run_time_tests(solutions : list, box_dimensions : tuple, meshfiles : list = None):
    """
    Executa e plota testes de tempo de execução para casos específicos"
    """
    if meshfiles is None:                                  
        meshfiles = helpers.create_meshes(box_dimensions, num = 20, suffix = "time")
    
    num_meshes = len(meshfiles)
    num_solutions = len(solutions)
    times = [[] for i in range(num_solutions)]
    verbose = False
    num_tests = 10

    for j, meshfile in enumerate(meshfiles):
        mesh = helpers.load_mesh(meshfile)
        mesh_params = helpers.get_mesh_params(mesh)
        mesh_args = (mesh, mesh_params)
        nvols = len(mesh.volumes)
        K = helpers.get_tensor(1., nvols)
        for i, solution in enumerate(solutions):
            iter_times = []
            for k in range(num_tests):
                name = solution["name"] if type(solution) == dict else "QFiveSpot"
                if type(solution) == dict:
                    helpers.verbose("Starting T-test [{}/{} - {}/{}] for {} volumes for case {}.".format(k, num_tests, j, num_meshes, nvols, name), type_msg = "OUT")
                    results = run_analytical(box_dimensions, 1, K,  solution, mesh_args, verbose = verbose)
                    helpers.verbose("Total time: {:.2}s -> Error : {:.3}.".format(results["times"]["Total Time"], results["error"]), type_msg="INFO")
                else:
                    helpers.verbose("Starting T-test [{}/{} - {}/{}] for {} volumes for case {}.".format(k, num_tests, j, num_meshes, nvols, name), type_msg = "OUT")
                    results = solution(box_dimensions, 1,  mesh_args, verbose = verbose)
                    helpers.verbose("Total time: {:.2}s".format(results["times"]["Total Time"]), type_msg="INFO")
                iter_times.append([nvols,  results["times"]["Total Time"],
                                           results["times"]["Pre-Processing"],
                                           results["times"]["TPFA System Preparation"] + results["times"]["TPFA Boundary Conditions"],
                                           results["times"]["TPFA System Solving"]])
            times[i].append(np.array(iter_times).mean(axis = 0))
    
    n = int(np.ceil(num_solutions / 2))
    m = int(np.ceil(num_solutions / n))

    fig, ax = plt.subplots(2, 2, figsize = (11, 6))
    fig.suptitle("Tempo de Execução")

    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]
    for i, time_d in enumerate(times):
        
        name   = solutions[i]["name"] if type(solutions[i]) == dict else "QFiveSpot"  

        
        time_d = np.array(time_d, dtype = float)
        ax[0][0].title.set_text("Total")
        ax[0][0].plot(time_d[:, 0], time_d[:, 1], 
                      label = name, color = c[i % len(c)], marker = markers[0])
        
        ax[0][1].title.set_text("Pré-Processamento")
        ax[0][1].plot(time_d[:, 0], time_d[:, 2], 
                      label = name, color = c[i % len(c)], marker = markers[1])

        ax[1][0].title.set_text("Montar Sistema TPFA")
        ax[1][0].plot(time_d[:, 0], time_d[:, 3], 
                      label = name, color = c[i % len(c)], marker = markers[2])

        ax[1][1].title.set_text("Resolver Sistema")
        ax[1][1].plot(time_d[:, 0], time_d[:, 4],
                      label = name, color = c[i % len(c)], marker = markers[3])

    for i in range(2):
        for j in range(2):
            ax[i][j].grid()
            ax[i][j].legend(loc = "upper left")
            ax[i][j].set_xlabel("Número de Volumes")
            ax[i][j].set_ylabel("Tempo (s)")

    plt.tight_layout()
    plt.savefig("./tests/time_tests/tempo.png")
    plt.clf()

def run_memory_tests(solutions: list, box_dimensions : tuple, meshfiles : list = None):
    if meshfiles is None:                                  
        meshfiles = helpers.create_meshes(box_dimensions, num = 20, suffix = "mem")
    
    num_meshes = len(meshfiles)
    num_solutions = len(solutions)
    memory_cost = [[] for i in range(num_solutions)]
    verbose = False
    num_tests = 5
    for j, meshfile in enumerate(meshfiles):
        mesh = helpers.load_mesh(meshfile)
        mesh_params = helpers.get_mesh_params(mesh)
        mesh_args = (mesh, mesh_params)
        nvols = len(mesh.volumes)
        K = helpers.get_tensor(1., nvols)
        for i, solution in enumerate(solutions):
            iter_memory = []
            for k in range(num_tests):
                name = solution["name"] if type(solution) == dict else "QFiveSpot"
                if type(solution) == dict:
                    helpers.verbose("Starting M-test [{}/{} - {}/{}] for {} volumes for case {}.".format(k, num_tests, j, num_meshes, nvols, name), type_msg = "OUT")
                    tracemalloc.start()
                    results = run_analytical(box_dimensions, 1, K,  solution, mesh_args, verbose = verbose)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    helpers.verbose("Total time: {:.2}s -> Error : {:.3}.".format(results["times"]["Total Time"], results["error"]), type_msg="INFO")
                else:
                    helpers.verbose("Starting M-test [{}/{} - {}/{}] for {} volumes for case {}.".format(k, num_tests, j, num_meshes, nvols, name), type_msg = "OUT")
                    tracemalloc.start()
                    results = solution(box_dimensions, 1,  mesh_args, verbose = verbose)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    helpers.verbose("Total time: {:.2}s".format(results["times"]["Total Time"]), type_msg="INFO")
                cost = (peak - current) / 10**6
                iter_memory.append([nvols,  cost])
            memory_cost[i].append(np.array(iter_memory).mean(axis = 0).astype(float))
    
    n = int(np.ceil(num_solutions / 2))
    m = int(np.ceil(num_solutions / n))

    fig, ax = plt.subplots(1, 1, figsize = (11, 6))
    fig.suptitle("Tempo de Execução")

    c = ["black", "blue", "green", "red", "grey"]
    markers = ["p", "s", "o", "v", "D"]
    for i, mem_d in enumerate(memory_cost):
        mem_d = np.array(mem_d, dtype = float)
        name   = solutions[i]["name"] if type(solutions[i]) == dict else "QFiveSpot"  
        ax.plot(mem_d[:, 0], mem_d[:, 1], marker = markers[i % len(markers)], color = c[i % len(c)], label = name)

    
    ax.grid()
    ax.legend(loc = "upper left")
    ax.set_xlabel("Número de Volumes")
    ax.set_ylabel("Mémoria (MB)")

    plt.tight_layout()
    plt.savefig("./tests/memory_tests/memory.png")
    plt.clf()


def main():
    # meshlist = helpers.create_meshes((1, 1, 1), num = 20, suffix = "acc")
    meshlist = ["mesh/box_{}acc.msh".format(i) for i in range(10)]
    #meshlist = ["mesh/cube_hex.msh"]
    #print(get_analytical_keys())

    solution1 = {
        "name"              : "x + y + z",
        "p"                 : lambda x, y, z: x + y + z,
        "-div(K grad(p))"   : lambda x, y, z, K: np.zeros_like(x),
        "dirichlet"         : lambda x, y, z: x + y + z,
        "neumann"           : lambda x, y, z: np.zeros_like(x)
    }
    solution2 = {
        "name"              : "x^2 + y^2 + z^2",
        "p"                 : lambda x, y, z:       x**2 + y**2 + z**2,
        "-div(K grad(p))"   : lambda x, y, z, K:    -6 * np.ones_like(x),
        "dirichlet"         : lambda x, y, z:       x**2 + y**2 + z**2,
        "neumann"           : lambda x, y, z:       np.zeros_like(x)
    }
    solution3 = {
        "name"              : "sin(x) + cos(y) + exp(z)",
        "p"                 : lambda x, y, z:       np.sin(x) + np.cos(y) + np.exp(z),
        "-div(K grad(p))"   : lambda x, y, z, K:    -(-K[:,0,0] * np.sin(x) - K[:,1,1] * np.cos(y) + K[:,2,2] * np.exp(z)),
        "dirichlet"         : lambda x, y, z:       np.sin(x) + np.cos(y) + np.exp(z),
        "neumann"           : lambda x, y, z:       np.zeros_like(x)
    }
    solution4 = {
        "name"              : "(x + 1) * log(1 + x) + 1/(y + 1) + z^2",
        "p"                 : lambda x, y, z:       (x + 1) * np.log(1 + x) + 1/(y + 1) + z**2,
        "-div(K grad(p))"   : lambda x, y, z, K:    -(K[:,0,0] * (1/(x+1)) + K[:,1,1] * (2/((y+1)**3)) + K[:,2,2] * 2),
        "dirichlet"         : lambda x, y, z:       (x + 1) * np.log(1 + x) + 1/(y + 1) + z**2,
        "neumann"           : lambda x, y, z:       np.zeros_like(x)
    }
    solutions = [solution1, solution2, solution3, solution4]
    #run_accurracy_tests(solutions, (1, 1, 1), meshlist)
    #run_time_tests([solution1, solution2, run_QFiveSpots], (1, 1, 1), meshlist)
    #run_memory_tests([solution1, solution2, run_QFiveSpots], (1, 1, 1), meshlist)
    run_QFiveSpots((10,10,0.1), 10000, verbose=True)


if __name__ == '__main__':
    main()