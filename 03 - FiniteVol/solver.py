import numpy as np
import time
from mesh import Mesh
from memory_profiler import profile


def get_random_tensor(a, b, size):
    """
    Retorna um array de tamanho size cujos elementos são tensores aleatórios diagonais 3 x 3, com valores entre a e b

    """
    diags = np.random.uniform(a, b, size = size + (3,))
    return np.apply_along_axis(np.diag, 3, diags)


def create_mesh(axis_attibutes, name, K = None, q = None, 
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
        
    return mesh

@profile
def exemplo1():
    # Example 1
    Lx, Ly = 0.6, 1
    nx, ny, nz = 500, 500, 1
    dx, dy, dz = Lx/nx, Ly/ny, 0.1

    k_time = time.time()
    K1 = get_random_tensor(52, 52, size = (nz, ny, nx))
    print("Time to create K1: \t\t {} s".format(round(time.time() - k_time, 5)))

    border_time = time.time()
    all_indices = np.arange(nx * ny * nz)
    left_border = all_indices[all_indices % nx == 0]
    up_border = all_indices[nx * (ny - 1):]
    right_border = all_indices[all_indices % nx == nx - 1]
    down_border = all_indices[0: nx]
    print("Time to create borders: \t {} s".format(round(time.time() - border_time, 5)))

    d_time = time.time()
    fd = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fd[up_border] = fd[down_border] = fd[right_border] = True
    fd_values = np.zeros((nx * ny * nz))
    fd_values[up_border] ,fd_values[down_border] ,fd_values[right_border] = 0, 100, 0
    fd1 = (fd, fd_values)
    print("Time to create dirichlet: \t {} s".format(round(time.time() - d_time, 5)))


    n_time = time.time()
    fn = np.full(fill_value = False, shape = (nx * ny * nz), dtype=bool)
    fn[left_border] = True
    fn_values = np.zeros((nx * ny * nz))
    fn_values[left_border] = 0
    fn1 = (fn, fn_values)
    print("Time to create neumann: \t {} s".format(round(time.time() - n_time, 5)))

    print("-----------------------------------------------------------------------")
    mesh1 = create_mesh([(nx, dx), (ny, dy), (nz, dz)], "Exemplo 1", 
                        fd = fd1, maskd = True, K = K1, fn = fn1, maskn = True, create_vtk=True)
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
    
    exemplo1()
    
    



def compare(mesh3d):
    # Compare face transmissibilities with IMPRESS TFPA matrix saved in faces_trans.npy
    A_tpfa = np.load("Atpfa.npy")
    print(np.allclose(mesh3d.A.todense(), A_tpfa))
    print(mesh3d.A.todense()[0:3, 0:3])
    print(A_tpfa[0:3, 0:3])

if __name__ == "__main__":
    main()