import numpy as np
import time
from mesh import Mesh

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

    if K is None:
        K = get_random_tensor(1, 1, size = (mesh.nz, mesh.ny, mesh.nx))
        index = np.arange(mesh.nx)
        K[:, index, index] = np.array([[1/1000, 0, 0], [0, 1/1000, 0], [0, 0, 1/1000]])
    
    if fd is None:
        dirichlet_points = np.array([[mesh.dx / 2, mesh.dy / 2, mesh.dz / 2]])
        dirichlet_values = np.array([1])
        fd = (dirichlet_points, dirichlet_values)

    if fn is None:
        neumann_points = np.array([[(mesh.nx - 1/2) * mesh.dx, (mesh.ny - 1/2) * mesh.dy, (mesh.nz - 1/2) * mesh.dz]])
        neumann_values = np.array([1])
        fn = (neumann_points, neumann_values)
    
    mesh.assemble_faces_transmissibilities(K)
    mesh.assemble_tpfa_matrix()
    mesh.set_boundary_conditions("dirichlet", fd)
    mesh.set_boundary_conditions("neumann", fn)

    mesh.solve_tpfa(dense)
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
    mesh2d = create_mesh([(3, 1), (3, 1)], "2D", plot_options=options, create_vtk=True, dense=False)
    mesh3d = create_mesh([(40, 1), (40, 1), (40, 1)], "3D", plot_options=options, create_vtk=True)

if __name__ == "__main__":
    main()