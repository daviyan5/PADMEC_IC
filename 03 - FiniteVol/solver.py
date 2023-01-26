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


def create_mesh(axis_attibutes, name, K = None, fd = None, fn = None, dense = False, plot_options = None, create_vtk = False):
    start_time = time.time()
    mesh = Mesh()
    mesh.assemble_mesh(axis_attibutes, name)
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
    nvols = mesh.nvols
    if K is None:
        K = get_random_tensor(1e-12, 1e-1, size = (nz, ny, nx))
        #index = np.arange(mesh.nx)
        #K[:, index, index] = np.array([[1/1000, 0, 0], [0, 1/1000, 0], [0, 0, 1/1000]])
    
    maskd = False
    if fd is None:
        i, j, k = np.meshgrid([0, nx-1], [0, ny-1], [0, nz-1], indexing='ij')
        indices = ((i + j*nx + k*nx*ny).flatten())

        dirichlet_points = np.full(fill_value = False, shape = (nvols), dtype=bool)
        dirichlet_points[indices] = True
        dirichlet_values = np.zeros((nvols))
        dirichlet_values[indices] = np.random.uniform(0, 1, size = (len(indices)))
        fd = (dirichlet_points, dirichlet_values)
        maskd = True

    maskn = False
    if fn is None:
        indices = np.random.randint(0, nvols, size = (nvols//10))

        neumann_points = np.full(fill_value = False, shape = (nvols), dtype=bool)
        neumann_points[indices] = True
        neumann_values = np.zeros((nvols))
        neumann_values[indices] = np.random.uniform(0, 1, size = (len(indices)))
        fn = (neumann_points, neumann_values)
        maskn = True
    
    @profile
    def simulate():
        mesh.assemble_faces_transmissibilities(K)
        mesh.assemble_tpfa_matrix()
        #mesh.set_boundary_conditions("dirichlet", fd, maskd)
        mesh.set_boundary_conditions("neumann", fn, maskn)
        mesh.solve_tpfa(dense)

    simulate()
    print("Time to simulate {} elements in mesh {}: \t {} s\n\n".format(mesh.nvols, mesh.name, round(time.time() - start_time, 5)))
    
    if plot_options is not None:
        mesh.plot(plot_options)
    
    if create_vtk:
        mesh.create_vtk()
        
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
    #options = None
    #mesh1d = create_mesh([(20, 0.1)], "1D", plot_options=options, create_vtk=True)
    mesh2d = create_mesh([(300, 1), (300, 1)], "2D", plot_options=options, create_vtk=True)
    mesh3d = create_mesh([(50, 1), (50, 1), (50, 1)], "3D", plot_options=options, create_vtk=True)

if __name__ == "__main__":
    main()