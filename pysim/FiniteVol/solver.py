import numpy as np
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


verbose = True

def get_random_tensor(a, b, size, n = 3, m = 3):
    return tg.get_random_tensor(a, b, size, n, m)

def get_tensor(a, size, n = 3, m = 3):
    return tg.get_tensor(a, size, n, m)

class Solver:
    def __init__(self):
        self.volumes_trans_tensor = None
        self.volumes_trans = None
        self.faces_trans = None

        self.A = None
        self.b = None
        self.p = None
        self.A_d = None
        self.b_d = None
        self.b_n = None
    
    def create_mesh(self, nx, ny, nz, dx, dy, dz, name = "mesh"):
        self.mesh = Mesh()
        axis = [[nx, dx], [ny, dy], [nz, dz]]
        self.mesh.assemble_mesh(axis, verbose, name = name)
        return self.mesh
        
    def solve(self, mesh, K, q, fd, fn, create_vtk = False, check = False, an_sol = None):
        
        self.mesh = mesh
        self._assemble_faces_transmissibilities(K)
        self._assemble_tpfa_matrix(q)
        self._set_boundary_conditions("dirichlet", fd)
        self._set_boundary_conditions("neumann", fn)
        self._solve_tpfa(check=check)
        if create_vtk:
            self._create_vtk(an_sol = an_sol)
        return self.p


    def _assemble_faces_transmissibilities(self, K):
        """
        Monta as transmissibilidades das faces da malha
        """
        start_time = time.time()

        self.volumes_trans_tensor = np.flip(K, 1)
        self.volumes_trans = (np.trace(self.volumes_trans_tensor, axis1=3, axis2=4)/3).flatten()
    

        # Kh = 2 / (1/K[:, :, 1:, 0, 0] + 1/K[:, :, :-1, 0, 0])
        # Kh = np.insert(Kh,  0, K[:, :, 0, 0, 0], axis = 2)
        # Kh = np.insert(Kh, Kh.shape[2], K[:, :, -1, 0, 0], axis = 2)
        # faces_trans_h = self.mesh.Sh * np.flip(Kh, 1).flatten() / (self.mesh.dx / 2) 
# 
        # faces_trans_l = np.empty((0))
        # Kl = 2 / (1/K[:, 1:, :, 1, 1] + 1/K[:, :-1, :, 1, 1])
        # Kl = np.insert(Kl,  0, K[:, 0, :, 1, 1], axis = 1)
        # Kl = np.insert(Kl, Kl.shape[1], K[:, -1, :, 1, 1], axis = 1)
        # faces_trans_l = self.mesh.Sl * np.flip(Kl, 1).flatten() / (self.mesh.dy / 2) 
# 
        # faces_trans_w = np.empty((0))
        # Kw = 2 / (1/K[1:, :, :, 2, 2] + 1/K[:-1, :, :, 2, 2])
        # Kw = np.insert(Kw,  0, K[0, :, :, 2, 2], axis = 0)
        # Kw = np.insert(Kw, Kw.shape[0], K[-1, :, :, 2, 2], axis = 0)
        # faces_trans_w = self.mesh.Sw * np.flip(Kw, 1).flatten() / (self.mesh.dz / 2) 

        volumes_pairs = self.mesh.faces.adjacents[:]
        
        K = np.flip(K, 1)
        K = K.flatten().reshape(self.mesh.nvols, 3, 3)
        
        KL = K[volumes_pairs[:, 0]]
        KR = K[volumes_pairs[:, 1]]

        KnL = np.einsum("ij,ikj->ik", self.mesh.faces.normal, KL) 
        KnR = np.einsum("ij,ikj->ik", self.mesh.faces.normal, KR) 

        KnL = np.einsum("ij,ij->i", self.mesh.faces.normal, KnL) 
        KnR = np.einsum("ij,ij->i", self.mesh.faces.normal, KnR) 
        
        hl = self.mesh.volumes.volume / self.mesh.faces.areas 
        hr = self.mesh.volumes.volume / self.mesh.faces.areas
        self.faces_trans = 2 * self.mesh.faces.areas * (KnL * KnR) / ((KnL * hl) + (KnR * hr))
        
        self.faces_trans = np.hstack((-self.faces_trans, -self.faces_trans))
        
        self.mesh.times["Montar Transmissibilidade das Faces"] = time.time() - start_time
        if verbose: 
            print("Time to assemble faces T: \t\t", round(self.mesh.times["Montar Transmissibilidade das Faces"], 5), "s")
    
    def _assemble_tpfa_matrix(self, q):
        """
        Monta a matriz de transmissibilidade TPFA
        """
        start_time = time.time()

        row_index_p = self.mesh.faces.adjacents[:, 0]
        col_index_p = self.mesh.faces.adjacents[:, 1]
        
        row_index = np.hstack((row_index_p, col_index_p))
        col_index = np.hstack((col_index_p, row_index_p))

       
        assert len(row_index) == len(col_index)
        assert len(row_index) == 2 * self.mesh.nfaces
        assert len(row_index) == len(self.faces_trans)


        time_to_assemble = time.time()
        self.A = csr_matrix((self.faces_trans, (row_index, col_index)), shape=(self.mesh.nvols, self.mesh.nvols))
        self.b = np.array(q, dtype=float)

        self.A = self.A.tolil()
        self.A.setdiag(0)
        self.A.setdiag(-self.A.sum(axis=1)) 
        self.A = self.A.tocsr()
        
        self.mesh.times["Montar Matriz A do TPFA"] = time.time() - start_time
        if verbose: 
            print("Time to assemble TPFA matrix: \t\t", round(self.mesh.times["Montar Matriz A do TPFA"], 5), "s")
    
    def _set_boundary_conditions(self, bc, f):
        """
        Define as condições de contorno

        Parametros
        ----------
        bc: str
            Tipo de condição de contorno
        f: tuple
            lambda f(x,y,z): valor da condição de contorno em cada ponto. f deve retornar um array de valores v
            cujo tamanho é igual ao número de pontos de contorno. x, y e z são arrays de coordenadas dos pontos de contorno 1 : 1.
        ----------
        """
        
        start_time = time.time()
        if bc == "dirichlet":
            x, y, z = self.mesh.faces._get_coords_from_index(self.mesh.faces.boundary)
            index = self.mesh.faces.boundary
            d_values = f(x, y, z)
            d_nodes = index[d_values != None]
            d_values = d_values[d_values != None].astype(float)
            volumes = self.mesh.faces.adjacents[d_nodes][:, 0]

            self.d_values = d_values
            self.d_nodes = d_nodes
            
            #self.A[volumes, volumes] -= self.faces_trans[d_nodes]
            #self.b[volumes] -= d_values * self.faces_trans[d_nodes]

            self.A_d = np.zeros(self.mesh.nvols)
            np.add.at(self.A_d, volumes, -self.faces_trans[d_nodes])
            row = np.arange(self.mesh.nvols)
            col = np.arange(self.mesh.nvols)
            self.A_d = csr_matrix((self.A_d, (row, col)), shape=(self.mesh.nvols, self.mesh.nvols))

            self.b_d = np.zeros(self.mesh.nvols)
            np.add.at(self.b_d, volumes, -d_values * self.faces_trans[d_nodes])

        elif bc == "neumann":
            x, y, z = self.mesh.faces._get_coords_from_index(self.mesh.faces.boundary)
            n_values = f(x, y, z)
            if n_values is None:
                return
            index = self.mesh.faces.boundary
            n_nodes  = index[n_values != None]
            n_values = n_values[n_values != None].astype(float)

            self.n_values = n_values
            self.n_nodes = n_nodes
            self.b_n = np.zeros(self.mesh.nvols)
            volumes = self.mesh.faces.adjacents[n_nodes][:, 0]
            #self.b[volumes] += n_values * self.mesh.volume
            np.add.at(self.b_n, volumes, n_values * self.mesh.volume)
        
        self.mesh.times["Condição de contorno - {}".format((bc))] = time.time() - start_time
        if verbose: 
            print("Time to set {} bc's: \t\t".format(bc), round(self.mesh.times["Condição de contorno - {}".format((bc))], 5), "s")
        

    def _solve_tpfa(self, dense = False, check = False):
        """
        Resolve o sistema linear da malha
        """
        start_time = time.time()
        # Check if there's no intersection between dirichlet and neumann points
        if dense:
            self.A = self.A.todense()
            self.p = np.linalg.solve(self.A, self.b)
        else:
            self.A.eliminate_zeros()
            start_time = time.time()
            self.p = pd_spsolve(self.A + self.A_d, self.b + self.b_d + self.b_n)
        np.set_printoptions(suppress=True)
        print((self.A).todense())
        print(self.p)
        print(self.b)
        if check:
            check_time = time.time()
            assert np.allclose((self.A + self.A_d).dot(self.p), self.b + self.b_d + self.b_n)
            print("CHECK ({}): TPFA system solved correctly".format(round(time.time() - check_time,5)))

        
        self.mesh.times["Resolver o Sistema TPFA"] = time.time() - start_time
        if verbose: 
            print("Time to solve TPFA system: \t\t", round(self.mesh.times["Resolver o Sistema TPFA"],5), "s")
    
    def _create_vtk(self, name = None, an_sol = None):
        if name == None:
            name = self.mesh.name
        vtk_time = time.time()
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
            pressure_array.SetName("Numerical Pressure normalized")
            grid.GetCellData().AddArray(pressure_array)
        else:
            pressure_array = numpy_support.numpy_to_vtk(self.p, deep=True)
            pressure_array.SetName("Numerical Pressure")
            grid.GetCellData().AddArray(pressure_array)

        # Add coordinates as point data
        x, y, z = self.mesh.volumes.x, self.mesh.volumes.y, self.mesh.volumes.z
        x_array = numpy_support.numpy_to_vtk(x, deep=True)
        x_array.SetName("X")
        grid.GetCellData().AddArray(x_array)
        y_array = numpy_support.numpy_to_vtk(y, deep=True)
        y_array.SetName("Y")
        grid.GetCellData().AddArray(y_array)
        z_array = numpy_support.numpy_to_vtk(z, deep=True)
        z_array.SetName("Z")
        grid.GetCellData().AddArray(z_array)
        
        # Add analytical solution as point data + error
        if an_sol is not None:
            an_sol = an_sol(x, y, z)
            an_sol_array = numpy_support.numpy_to_vtk(an_sol, deep=True)
            an_sol_array.SetName("Analytical Pressure")
            grid.GetCellData().AddArray(an_sol_array)
            error = np.abs(self.p - an_sol)
            error_array = numpy_support.numpy_to_vtk(error, deep=True)
            error_array.SetName("Error")
            grid.GetCellData().AddArray(error_array)
        

        # Write the rectilinear grid to a .vtk filea
        writer = vtk.vtkDataSetWriter()
        writer.SetFileName("./tmp/{}_mesh.vtk".format(name))
        writer.SetInputData(grid)
        writer.Write()
        self.mesh.times["Criar arquivo VTK"] = time.time() - vtk_time
        if verbose: 
            print("Time to create vtk file: \t\t", round(self.mesh.times["Criar arquivo VTK"], 5), "s")

if __name__ == "__main__":
    main()