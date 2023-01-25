import numpy as np
import time
from mesh import Mesh

def get_random_tensor(a, b, size):
    """
    Retorna um array de tamanho size cujos elementos são tensores aleatórios diagonais 3 x 3, com valores entre a e b

    """
    diags = np.random.uniform(a, b, size = size + (3,))
    return np.apply_along_axis(np.diag, 3, diags)

def create_mesh(axis_attibutes, name, K = None, fd = None, fn = None, q = None, dense = False, plot_options = None, create_vtk = False):
    start_time = time.time()
    mesh = Mesh()
    mesh.assemble_mesh(axis_attibutes, name)

    if K is None:
        K = get_random_tensor(1, 10, size = (mesh.nz, mesh.ny, mesh.nx))
    
    if q is None:
        q = np.zeros(mesh._get_num_volumes())

    if fd is None:
        fd = lambda x, y, z: 1

    if fn is None:
        fn = lambda x, y, z: 0
    
    mesh.assemble_faces_transmissibilities(K)
    mesh.assemble_tpfa_matrix(dense)
    mesh.set_boundary_conditions("dirichlet", q, fd)
    mesh.set_boundary_conditions("neumann", q, fn)

    mesh.solve_tpfa(q)
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
    show_A = False
    show_solution = True

    options = (show_coordinates, show_volumes, show_faces, show_adjacents, show_transmissibilities, show_A, show_solution)
    options = None
    mesh1d = create_mesh([(20, 0.1)], "1D", plot_options=options, create_vtk=True)
    mesh2d = create_mesh([(200, 0.1), (200, 0.3)], "2D", plot_options=options, create_vtk=True)
    mesh3d = create_mesh([(30, 0.1), (30, 0.3), (30, 0.5)], "3D", plot_options=options, create_vtk=True)

if __name__ == "__main__":
    main()