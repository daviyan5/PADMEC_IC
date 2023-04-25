import numpy as np
import matplotlib.pyplot as plt
import helpers

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
    
    q = np.full(nvols, vq)
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
    for meshfile in meshfiles:
        mesh = helpers.load_mesh(meshfile)
        mesh_params = helpers.get_mesh_params(mesh)
        mesh_args = (mesh, mesh_params)
        nvols = len(mesh.volumes)
        K = helpers.get_tensor(1., nvols)
        for solution in analytical_solutions:

            helpers.verbose("Starting accuracy test for {} volumes with solution {}.".format(nvols, solution["name"]), type_msg = "OUT")
            results = run_analytical(box_dimensions, 1, K,  solution, mesh_args, verbose = False)
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
        if i != 0:
            ax[a][b].plot(vols, scale_vols, label = "O(n²)", color = 'purple', linestyle = "--")
        else:
            ax[a][b].set_ylim(1e-18, 1e-14)
        
        ax[a][b].set_title(sol_name)
        ax[a][b].set_xlabel("Número de Volumes")
        ax[a][b].set_ylabel("I²rel")
        ax[a][b].grid()
        ax[a][b].legend()
    
    plt.tight_layout()
    plt.savefig("./tests/accuracy_tests/accuracy.png")
    plt.clf()

def main():
    # meshlist = helpers.create_meshes((1, 1, 1), num = 20, suffix = "acc")
    meshlist = ["mesh/box_{}acc.msh".format(i) for i in range(5)]
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
        "p"                 : lambda x, y, z:       x**3 + y**3 + z**3,
        "-div(K grad(p))"   : lambda x, y, z, K:    -6 * (K[:,0,0] * x + K[:,1,1] * y + K[:,2,2] * z),
        "dirichlet"         : lambda x, y, z:       x**3 + y**3 + z**3,
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
    run_accurracy_tests(solutions, (1, 1, 1), meshlist)



if __name__ == '__main__':
    main()