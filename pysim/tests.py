import numpy as np
import matplotlib.pyplot as plt
import helpers
import tracemalloc

from solver import TPFAsolver
from mesh_generator import MeshGenerator
import plots

class Problem:
    def __init__(self):
        self.name = None
        self.handle = None
        self.meshfiles = None
        self.nvols_arr = None
        self.avg_error_arr = None
        self.avg_time_arr = None
        self.avg_memory_arr = None

    def init_problem(self, name, handle, meshfiles):
        self.name = name
        self.handle = handle
        self.meshfiles = meshfiles
        self.nvols_arr = [0] * len(meshfiles)
        self.avg_error_arr = [0] * len(meshfiles)
        basic = {}
        ks = [
              "Total Time", 
              "Pre-Processing", 
              "TPFA System Preparation", 
              "TPFA Boundary Conditions", 
              "TPFA System Solving", 
              "Post-Processing"
              ]
        for key in ks:
            basic[key] = 0
        self.avg_time_arr = [basic.copy() for _ in range(len(meshfiles))]
        self.avg_memory_arr = [0] * len(meshfiles)

    def add_to_problem(self, i, j, nvols, error, time, memory):
        if i >= len(self.meshfiles):
            raise IndexError('Index out of bounds')
        self.nvols_arr[i] = nvols
        self.avg_error_arr[i] = self.avg_error_arr[i] * (j-1) / j + error / j
        for key in time.keys():
            self.avg_time_arr[i][key] = self.avg_time_arr[i].get(key, 0) * (j-1) / j + time[key] / j
        self.avg_memory_arr[i] = self.avg_memory_arr[i] * (j-1) / j + memory / j

def linear_case(mesh, mesh_params, box_dimensions = (1, 1, 1), verbose = False):
    Lx, Ly, Lz = box_dimensions
    nvols = len(mesh_params[1])
    name = "x + y + z"
    K = helpers.get_tensor(1., nvols)

    def dirichlet(x, y, z, vols, mesh):
        return x + y + z
    def neumann(x, y, z, vols, mesh):
        return np.zeros_like(x)
    def source(x, y, z, K):
        return np.zeros_like(x)
    def analytical(x, y, z):
        return x + y + z

    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : source,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : analytical,
            "vtk"           : False
        }
    
    solver = TPFAsolver(verbose, verbose, name)
    solver.solveTPFA(args)
    return solver

def quadratic_case(mesh, mesh_params, box_dimensions = (1, 1, 1), verbose = False):
    Lx, Ly, Lz = box_dimensions
    nvols = len(mesh_params[1])
    name = "x^2 + y^2 + z^2"
    K = helpers.get_tensor(1., nvols)

    def dirichlet(x, y, z, vols, mesh):
        return x**2 + y**2 + z**2
    def neumann(x, y, z, vols, mesh):
        return np.zeros_like(x)
    def source(x, y, z, K):
        return -6 * np.ones_like(x)
    def analytical(x, y, z):
        return x**2 + y**2 + z**2

    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : source,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : analytical,
            "vtk"           : False
        }
    
    solver = TPFAsolver(verbose, verbose, name)
    solver.solveTPFA(args)
    return solver

def qfive_spot(mesh, mesh_params, box_dimensions = (1, 1, 1), verbose = False):
    Lx, Ly, Lz = box_dimensions
    nvols = len(mesh_params[1])
    name = "1/4 de Five-Spot"
    K = helpers.get_tensor(1., nvols)
    d1 = 10000.
    d2 = 1.
    k1 = 100.
    k2 = 100.
    vq = 0.
    
    def dirichlet(x, y, z, vols, mesh):
        p1 = 0.1
        p2 = 0.1
        l1 = Lx / 8
        c1 = Ly / 8
        l2 = Lx / 8
        c2 = Ly / 8
        dist0 = (x >= 0) & (x <= l1) & (y >= 0) & (y <= c1)
        dist1 = (x >= Lx - l2) & (x <= Lx) & (y >= Ly - c2) & (y <= Ly)
        v1 = np.where(dist0)
        v2 = np.where(dist1)

        d = np.full(len(x), np.nan)
        d[v1] = d1
        d[v2] = d2
        return d

    def neumann(x, y, z, vols, mesh):
        return np.zeros_like(x)
    
    def source(x, y, z, K):
        return np.ones_like(x) * vq
    
    def analytical(x, y, z):
        ref = np.load("vtks/qfive_spot_ref.npy", allow_pickle=True).item()
        xr = ref['x']
        yr = ref['y']
        pr = ref['p']

        n = int(np.sqrt(len(x)))

        dx = Lx / n
        dy = Ly / n

        i = np.floor(yr / dy).astype(int)
        j = np.floor(xr / dx).astype(int)

        idx = i * n + j
        a_p = np.zeros(n * n)
        freq = np.zeros(n * n)

        np.add.at(a_p, idx, pr)
        np.add.at(freq, idx, 1)

        m = freq[0]
        a_p /= m
        return a_p


    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : source,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : analytical,
            "vtk"           : False
        }
    
    solver = TPFAsolver(verbose, verbose, name)
    solver.solveTPFA(args)
    return solver

def extra_case1(mesh, mesh_params, box_dimensions = (1, 1, 1), verbose = False):
    Lx, Ly, Lz = box_dimensions
    nvols = len(mesh_params[1])
    name = "sin(x) + cos(y) + exp(z)"
    K = helpers.get_tensor(1., nvols)

    def dirichlet(x, y, z, vols, mesh):
        return np.sin(x) + np.cos(y) + np.exp(z)
    def neumann(x, y, z, vols, mesh):
        return np.zeros_like(x)
    def source(x, y, z, K):
        return -(-K[:,0,0] * np.sin(x) - K[:,1,1] * np.cos(y) + K[:,2,2] * np.exp(z))
    def analytical(x, y, z):
        return np.sin(x) + np.cos(y) + np.exp(z)

    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : source,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : analytical,
            "vtk"           : False
        }
    
    solver = TPFAsolver(verbose, verbose, name)
    solver.solveTPFA(args)
    return solver

def extra_case2(mesh, mesh_params, box_dimensions = (1, 1, 1), verbose = False):
    Lx, Ly, Lz = box_dimensions
    nvols = len(mesh_params[1])
    name = "(x + 1) * log(1 + x) + 1/(y + 1) + z^2"
    K = helpers.get_tensor(1., nvols)

    def dirichlet(x, y, z, vols, mesh):
        return (x + 1) * np.log(1 + x) + 1/(y + 1) + z**2
    def neumann(x, y, z, vols, mesh):
        return np.zeros_like(x)
    def source(x, y, z, K):
        return -(K[:,0,0] * (1/(x+1)) + K[:,1,1] * (2/((y+1)**3)) + K[:,2,2] * 2)
    def analytical(x, y, z):
        return (x + 1) * np.log(1 + x) + 1/(y + 1) + z**2

    args = {
            "meshfile"      : None,
            "mesh"          : mesh,
            "mesh_params"   : mesh_params,
            "dim"           : 3,
            "permeability"  : K,
            "source"        : source,
            "neumann"       : neumann,
            "dirichlet"     : dirichlet,
            "analytical"    : analytical,
            "vtk"           : False
        }
    
    solver = TPFAsolver(verbose, verbose, name)
    solver.solveTPFA(args)
    return solver


def tests(meshfiles, meshfiles_qfive, nrepeats = 5):
    p1, p2, p3, p4, p5 = Problem(), Problem(), Problem(), Problem(), Problem()
    p1.init_problem("linear_case",      linear_case,    meshfiles)
    p2.init_problem("quadratic_case",   quadratic_case, meshfiles)
    p3.init_problem("extra_case1",      extra_case1,    meshfiles)
    p4.init_problem("extra_case2",      extra_case2,    meshfiles)
    p5.init_problem("qfive_spot",       qfive_spot,     meshfiles_qfive)
    problems = [p1, p2, p3, p4, p5]
    verbose = False
    helpers.verbose("Loading meshes and mesh_params", "OUT")
    meshes_dict = {}
    for meshfile in meshfiles + meshfiles_qfive:
        helpers.verbose("Loading {}...".format(meshfile), "INFO")
        mesh = helpers.load_mesh(meshfile)
        mesh_params = helpers.get_mesh_params(mesh)
        meshes_dict[meshfile] = (mesh, mesh_params)
        helpers.verbose("Finished loading {} : {} volumes".format(meshfile, len(mesh.volumes)), "INFO")
    
    helpers.verbose("Starting tests", "OUT")
    for i in range(len(problems)):
        k = 0
        solver_name = ""
        for meshfile in problems[i].meshfiles:
            mesh = meshes_dict[meshfile][0]
            mesh_params = meshes_dict[meshfile][1]
            for j in range(1, nrepeats + 1):
                helpers.verbose(f"Starting [{j}/{nrepeats}] rep of [{k + 1}/{len(problems[i].meshfiles)}] in {problems[i].name}", "OUT")
                if problems[i].name != "qfive_spot":
                    solver = problems[i].handle(mesh, mesh_params, (1, 1, 1), verbose = verbose)
                else:
                    solver = problems[i].handle(mesh, mesh_params, (6, 4, 1), verbose = verbose)
                solver_name = solver.name
                helpers.verbose(f"Finished [{j}/{nrepeats}] reps of [{k + 1}/{len(problems[i].meshfiles)}] in {solver.name}", "INFO")
                problems[i].add_to_problem(k, j, solver.nvols, solver.get_error(), solver.times, solver.memory)
            k += 1
        
        pr = problems[i]
        name = solver_name
        meshfiles = np.array(pr.meshfiles)
        nvols = np.array(pr.nvols_arr)
        error = np.array(pr.avg_error_arr)
        times = np.array(pr.avg_time_arr)
        memory = np.array(pr.avg_memory_arr)
        d = {
            "name" : name,
            "meshfiles" : meshfiles,
            "nvols" : nvols,
            "error" : error,
            "times" : times,
            "memory" : memory
        }
        np.save(f"./results/{pr.name}.npy", d)


def main():
    meshfiles = [f"./mesh/box_{i}acc.msh" for i in range(15)]
    meshfiles_qfive = [f"./mesh/box_{i}.msh" for i in range(2, 8)]
    nrepeats = 5
    # meshfile = "./mesh/cube_hex.msh"
    # mesh = helpers.load_mesh(meshfile)
    # mesh_params = helpers.get_mesh_params(mesh)
    # linear_case(mesh, mesh_params, (1, 1, 1), verbose = True)
    #tests(meshfiles, meshfiles_qfive, nrepeats)
   
    case1 = np.load("./results/linear_case.npy", allow_pickle=True).item()
    case2 = np.load("./results/quadratic_case.npy", allow_pickle=True).item()
    case3 = np.load("./results/extra_case1.npy", allow_pickle=True).item()
    case4 = np.load("./results/extra_case2.npy", allow_pickle=True).item()
    case5 = np.load("./results/qfive_spot.npy", allow_pickle=True).item()
    cases = [case1, case2, case4, case5]
    plots.plot_accurracy(cases)
    plots.plot_times(cases)
    plots.plot_memory(cases)

if __name__ == '__main__':
    main()