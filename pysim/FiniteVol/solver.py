import numpy as np
import numba as nb
import time
import os
import sys
import vtk
import warnings


from vtk.util import numpy_support
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning
from pypardiso import spsolve as pd_spsolve
## Importing the mesh handler
current = os.path.dirname(os.path.realpath(__file__))   
parent = os.path.dirname(current)
sys.path.append(parent)
from Mesh.mesh_handler import Mesh
from Utils import tensor_generator as tg
sys.path.pop()


print_times = True

def get_random_tensor(a, b, size, n = 3, m = 3):
    return tg.get_random_tensor(a, b, size, n, m)

def simulate_tpfa(axis_attibutes, name, K = None, q = None, 
                fd = None, maskd = False, fn = None, maskn = False, 
                dense = False, check = False, create_vtk = False):

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
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    start_time = time.time()
    mesh = Mesh()
    mesh.assemble_mesh(axis_attibutes, print_times, name)
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
    
    
    solver = Solver(mesh)
    solver.assemble_faces_transmissibilities(K)
    solver.assemble_tpfa_matrix()
    q = solver.set_boundary_conditions("dirichlet", fd, q, maskd)
    q = solver.set_boundary_conditions("neumann", fn, q, maskn)
    solver.solve_tpfa(q, dense, check)

    if print_times: 
        print("Time to simulate {} elements in mesh {}: \t {} s\n\n".format(mesh.nvols, mesh.name, round(time.time() - start_time, 5)))

    
    if create_vtk:
        solver.create_vtk()
    
    #solver.save_all()
    return mesh, solver


class Solver:
    def __init__(self, mesh):
        self.mesh = mesh
        self.volumes_trans_tensor = None
        self.volumes_trans = None
        self.faces_trans = None

        self.A = None
        self.p = None

        self.dirichlet_points = None
        self.dirichlet_values = None
        self.neumann_points = None
        self.neumann_values = None
    
    def save_all(self):
        np.save("volumes_trans", self.volumes_trans)
        np.save("faces_trans", self.faces_trans)
        np.save("A", self.A.todense())
        np.save("p", self.p)
        np.save("dirichlet_points", self.dirichlet_points)
        np.save("dirichlet_values", self.dirichlet_values)
        np.save("neumann_points", self.neumann_points)
        np.save("neumann_values", self.neumann_values)
        np.save("faces_adjacents", self.mesh.faces_adjacents)

    def assemble_faces_transmissibilities(self, K):
        """
        Monta as transmissibilidades das faces da malha
        """
        start_time = time.time()

        self.volumes_trans_tensor = np.flip(K, 1)
        self.volumes_trans = (np.trace(self.volumes_trans_tensor, axis1=3, axis2=4)/3).flatten()
    

        Kh = 2 / (1/K[:, :, 1:, 0, 0] + 1/K[:, :, :-1, 0, 0])
        Kh = np.insert(Kh,  0, K[:, :, 0, 0, 0], axis = 2)
        Kh = np.insert(Kh, Kh.shape[2], K[:, :, -1, 0, 0], axis = 2)
        faces_trans_h = self.mesh.Sh * np.flip(Kh, 1).flatten() / self.mesh.dx / 2

        faces_trans_l = np.empty((0))
        if(self.mesh.dimension > 1):
            Kl = 2 / (1/K[:, 1:, :, 1, 1] + 1/K[:, :-1, :, 1, 1])
            Kl = np.insert(Kl,  0, K[:, 0, :, 1, 1], axis = 1)
            Kl = np.insert(Kl, Kl.shape[1], K[:, -1, :, 1, 1], axis = 1)
            faces_trans_l = self.mesh.Sl * np.flip(Kl, 1).flatten() / self.mesh.dy / 2

        faces_trans_w = np.empty((0))
        if(self.mesh.dimension > 2):
            Kw = 2 / (1/K[1:, :, :, 2, 2] + 1/K[:-1, :, :, 2, 2])
            Kw = np.insert(Kw,  0, K[0, :, :, 2, 2], axis = 0)
            Kw = np.insert(Kw, Kw.shape[0], K[-1, :, :, 2, 2], axis = 0)
            faces_trans_w = self.mesh.Sw * np.flip(Kw, 1).flatten() / self.mesh.dz / 2
        
        self.faces_trans = np.hstack((faces_trans_h, faces_trans_l, faces_trans_w))
        self.faces_trans = np.hstack((-self.faces_trans, -self.faces_trans))

        if print_times: 
            print("Time to assemble faces T in mesh {}: \t\t".format(self.mesh.name), round(time.time() - start_time, 5), "s")
    
    def assemble_tpfa_matrix(self):
        """
        Monta a matriz de transmissibilidade TPFA
        """
        start_time = time.time()

        row_index_p = self.mesh.faces.adjacents[:, 0]
        col_index_p = self.mesh.faces.adjacents[:, 1]
        
        row_index = np.hstack((row_index_p, col_index_p))
        col_index = np.hstack((col_index_p, row_index_p))

        #equal_idx = np.where(row_index == col_index)[0]
        #equal_idx = row_index[equal_idx]
        #self.faces_trans[equal_idx] = 1.
        
        assert len(row_index) == len(col_index)
        assert len(row_index) == 2 * self.mesh.nfaces
        assert len(row_index) == len(self.faces_trans)


        creation_time = time.time()
        self.A = csr_matrix((self.faces_trans, (row_index, col_index)), shape=(self.mesh.nvols, self.mesh.nvols))
        if print_times: 
            print("Time to create TPFA matrix in mesh {}: \t\t".format(self.mesh.name), round(time.time() - creation_time, 5), "s")

    
        A = self.A.tolil()
        self.A.setdiag(0)
        self.A.setdiag(-self.A.sum(axis=1)) 
        A = self.A.tocsr()
        
        if print_times: 
            print("Time to assemble TPFA matrix in mesh {}: \t\t".format(self.mesh.name), round(time.time() - start_time, 5), "s")
    
    def set_boundary_conditions(self, bc, f, q, mask = False):
        """
        Define as condições de contorno

        Parametros
        ----------
        bc: str
            Tipo de condição de contorno
        f: tuple
            Tupla com as coordenadas e valores das condições de contorno
        q: np.array
            Vetor de fontes
        mask: bool
            Se True, as condições de contorno serão aplicadas apenas nos volumes que possuem a mesma coordenada que as condições de contorno
        ----------
        """
        if(f[0].shape[0] == 0):
            print("No boundary conditions were set for mesh {}".format(self.mesh.name))
            return q
        start_time = time.time()
        if bc == "dirichlet":
            self.dirichlet_points = f[0]
            self.dirichlet_values = f[1]
            
            indexes = None
            if not mask:
                # Get indexes of boundary volumes points (x,y,z) -> index
                indexes = self.mesh.volumes._get_index_from_coords(coords = (f[0][:, 0], f[0][:, 1], f[0][:, 2]))
                indexes = np.where(self.mesh.volumes.boundary[indexes] == True)[0]
            else:
                indexes = np.where(np.logical_and(self.dirichlet_points, self.mesh.volumes.boundary))[0]
            for i in indexes:
                self.A.data[self.A.indptr[i] : self.A.indptr[i + 1]] = 0.
            self.A[indexes, indexes] = 1.
            
            q[indexes] = self.dirichlet_values
        elif bc == "neumann":
            self.neumann_points = f[0]
            self.neumann_values = f[1]

            indexes = None
            if not mask:
                # Get indexes of boundary volumes points (x,y,z) -> index
                indexes = self.mesh.volumes._get_index_from_coords(coords = (f[0][:, 0], f[0][:, 1], f[0][:, 2]))
                indexes = np.where(self.mesh.volumes.boundary[indexes] == True)[0]
            else:
                indexes = np.where(np.logical_and(self.neumann_points, self.mesh.volumes.boundary))[0]
            q[indexes] += self.neumann_values

        if print_times: 
            print("Time to set {} bc's in mesh {}: \t\t".format(bc, self.mesh.name), round(time.time() - start_time, 5), "s")
        return q

    def solve_tpfa(self, q, dense = False, check = False):
        """
        Resolve o sistema linear da malha
        """
        start_time = time.time()
        # Check if there's no intersection between dirichlet and neumann points
        if dense:
            self.A = self.A.todense()
            self.p = np.linalg.solve(self.A, q)
        else:
            self.A.eliminate_zeros()
            q = csr_matrix(q).T
            start_time = time.time()
            self.p = pd_spsolve(self.A, q)
        
        if check:
            check_time = time.time()
            assert np.allclose(self.A.dot(self.p), q)
            print("CHECK ({}): TPFA system solved correctly in mesh {}".format(round(time.time() - check_time,5), self.mesh.name))
        #print(np.around(self.A.todense(), 3))
        #print(np.around(q, 3))
        #print(np.around(self.p, 3))
        if print_times: 
            print("Time to solve TPFA system in mesh {}: \t\t".format(self.mesh.name), round(time.time() - start_time, 5), "s")
    
    def create_vtk(self):
        # Create the rectilinear grid
        grid = vtk.vtkImageData()
        grid.SetDimensions(self.mesh.nx + 1, self.mesh.ny + 1, self.mesh.nz + 1)
        #index = np.arange(self.mesh.nvols)
        #i, j, k = index % self.mesh.nx, (index // self.mesh.nx) % self.mesh.ny, index // (self.mesh.nx * self.mesh.ny)
        #x, y, z = (i + 1/2) * self.mesh.dx, (j + 1/2) * self.mesh.dy, (k + 1/2) * self.mesh.dz
        grid.SetSpacing(self.mesh.dx, self.mesh.dy, self.mesh.dz)
        # Add the permeability array as a cell data array
        permeability_array = numpy_support.numpy_to_vtk(self.volumes_trans, deep=True)
        permeability_array.SetName("Permeability")
        grid.GetCellData().AddArray(permeability_array)

        # Add the pressure array as a point data array
        
        if(self.p.max() - self.p.min() <= 1e-6 or self.p.max() > 1e6 or self.p.min() < -1e6):
            p = self.p
            if(p.max() == p.min()):
                norm = np.ones_like(p)
            else:
                norm = (p - p.min()) / (p.max() - p.min())
            pressure_array = numpy_support.numpy_to_vtk(norm, deep=True)
            pressure_array.SetName("Pressure normalized")
            grid.GetCellData().AddArray(pressure_array)
        else:
            pressure_array = numpy_support.numpy_to_vtk(self.p, deep=True)
            pressure_array.SetName("Pressure")
            grid.GetCellData().AddArray(pressure_array)

        # Write the rectilinear grid to a .vtk filea
        writer = vtk.vtkDataSetWriter()
        writer.SetFileName("{}_mesh.vtk".format(self.mesh.name))
        writer.SetInputData(grid)
        writer.Write() 

if __name__ == "__main__":
    main()