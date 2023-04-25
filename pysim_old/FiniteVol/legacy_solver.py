import numpy as np
import time
import os
import sys

## Importing the mesh handler
current = os.path.dirname(os.path.realpath(__file__))   
parent = os.path.dirname(current)
sys.path.append(parent)
from Mesh.legacy_mesh_handler import Mesh
from Utils import tensor_generator as tg
sys.path.pop()



def get_random_tensor(a, b, size, n = 3, m = 3, only_diagonal = True):
    return tg.get_random_tensor(a, b, size, n, m, only_diagonal)

def simulate_tpfa(axis_attibutes, name, K = None, q = None, 
                fd = None, maskd = False, fn = None, maskn = False, 
                dense = False, plot_options = None, create_vtk = False):

    """
    Cria e simula uma malha.

    axis_attibutes: lista de tuplas "(n, dx)" onde n é o número de volumes no eixo e dx é o tamanho do volume no eixo
    name: nome da malha
    
    K: array de tensores de permeabilidade. shape = (nz, ny, nx, 3, 3)
    
    q: array de termos-fonte (nx * ny * nz). shape = (nz, ny, nx, 3, 3)
   
    fd: tupla (dirichlet_points, dirichlet_values) onde ->
        dirichlet_points: é um array de pontos (x, y, z) que farão parte da condição de contorno Dirichlet
        dirichlet_values: é um array de valores que serão atribuídos aos pontos de Dirichlet (em ordem de index)

    maskd: booleano. caso seja True, a tupla fd será interpretada como uma mascára. Ou seja:
        dirichelet_points: é um array de booleanos que indica quais 
                           volumes farão parte da condição de contorno Dirichlet. shape = (nx * ny * nz)
        dirichlet_values: é um array de valores que serão atribuídos aos volumes de Dirichlet. shape =  (nx * ny * nz)
    
    fn: tupla (neumann_points, neumann_values) onde ->
        neumann_points: é um array de pontos (x, y, z) que farão parte da condição de contorno Neumann
        neumann_values: é um array de valores que serão atribuídos aos pontos de Neumann (em ordem de index)

    maskn: booleano. caso seja True, a tupla fn será interpretada como uma mascára. Ou seja:
        neumann_points: é um array de booleanos que indica quais 
                        volumes farão parte da condição de contorno Neumann. shape = (nx * ny * nz)
        neumann_values: é um array de valores que serão atribuídos aos volumes de Neumann. shape =  (nx * ny * nz)
    
    dense: booleano. caso seja True, a matriz de transmissibilidade será armazenada como uma matriz densa. 
           Caso contrário, será armazenada como uma matriz esparsa.
    
    plot_options: tupla com as opções de plotagem (ver documentação da função plot_mesh).
    
    create_vtk: booleano. caso seja True, será criado um arquivo .vtk com os resultados da simulação.

    """          
    start_time = time.time()
    mesh = Mesh()
    mesh.assemble_mesh(axis_attibutes, name)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    nvols = mesh.nvols

    if K is None:
        K = get_random_tensor(0, 1, size = (nz, ny, nx))
    
    if q is None:
        q = np.zeros((nvols))
    
    if fd is None:
        dirichlet_points = np.full(fill_value = False, shape = (nvols), dtype=bool)
        dirichlet_points[0] = True
        dirichlet_values = np.zeros((nvols))
        dirichlet_values[0] = np.random.uniform(0, 1)
        fd = (dirichlet_points, dirichlet_values)
        maskd = True

    if fn is None:
        neumann_points = np.full(fill_value = False, shape = (nvols), dtype=bool)
        neumann_points[-1] = True
        neumann_values = np.zeros((nvols))
        neumann_values[-1] = np.random.uniform(0, 1)
        fn = (neumann_points, neumann_values)
        maskn = True
    
    def simulate(q):
        mesh.assemble_faces_transmissibilities(K)
        mesh.assemble_tpfa_matrix()
        q = mesh.set_boundary_conditions("dirichlet", fd, q, maskd)
        q = mesh.set_boundary_conditions("neumann", fn, q, maskn)
        mesh.solve_tpfa(q, dense)

    simulate(q)
    print("Time to simulate {} elements in mesh {}: \t {} s\n\n".format(mesh.nvols, mesh.name, round(time.time() - start_time, 5)))
    
    if plot_options is not None:
        mesh.plot(plot_options)
    
    if create_vtk:
        mesh.create_vtk()
    
    legacy = "legacy"
    np.save(legacy + "volumes_trans", mesh.volumes_trans)
    np.save(legacy + "faces_trans", mesh.faces_trans)
    np.save(legacy + "A", mesh.A.todense())
    np.save(legacy + "p", mesh.p)
    np.save(legacy + "dirichlet_points", mesh.dirichlet_points)
    np.save(legacy + "dirichlet_values", mesh.dirichlet_values)
    np.save(legacy + "neumann_points", mesh.neumann_points)
    np.save(legacy + "neumann_values", mesh.neumann_values)
    np.save(legacy + "faces_adjacents", mesh.faces_adjacents)
    return mesh


def main():
    np.set_printoptions(suppress=True)

    show_coordinates = False
    show_volumes = True
    show_faces = False
    show_adjacents = False
    show_transmissibilities = False
    print_matrices = False
    show_solution = True

    options = (show_coordinates, show_volumes, show_faces, show_adjacents, show_transmissibilities, print_matrices, show_solution)
    


if __name__ == "__main__":
    main()