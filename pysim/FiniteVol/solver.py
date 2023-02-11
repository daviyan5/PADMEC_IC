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
        faces_trans_h = self.mesh.Sh * np.flip(Kh, 1).flatten() / (self.mesh.dx)

        faces_trans_l = np.empty((0))
        if(self.mesh.dimension > 1):
            Kl = 2 / (1/K[:, 1:, :, 1, 1] + 1/K[:, :-1, :, 1, 1])
            Kl = np.insert(Kl,  0, K[:, 0, :, 1, 1], axis = 1)
            Kl = np.insert(Kl, Kl.shape[1], K[:, -1, :, 1, 1], axis = 1)
            faces_trans_l = self.mesh.Sl * np.flip(Kl, 1).flatten() / (self.mesh.dy)

        faces_trans_w = np.empty((0))
        if(self.mesh.dimension > 2):
            Kw = 2 / (1/K[1:, :, :, 2, 2] + 1/K[:-1, :, :, 2, 2])
            Kw = np.insert(Kw,  0, K[0, :, :, 2, 2], axis = 0)
            Kw = np.insert(Kw, Kw.shape[0], K[-1, :, :, 2, 2], axis = 0)
            faces_trans_w = self.mesh.Sw * np.flip(Kw, 1).flatten() / (self.mesh.dz)
        
        self.faces_trans = np.hstack((faces_trans_h, faces_trans_l, faces_trans_w))
        self.faces_trans = np.hstack((-self.faces_trans, -self.faces_trans))

        mesh.times["assemble_faces_transmissibilities"] = round(time.time() - start_time, 5)
        if verbose: 
            print("Time to assemble faces T: \t\t", mesh.times["assemble_faces_transmissibilities"], "s")
    
    def assemble_tpfa_matrix(self):
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


        self.A = csr_matrix((self.faces_trans, (row_index, col_index)), shape=(self.mesh.nvols, self.mesh.nvols))

        A = self.A.tolil()
        self.A.setdiag(0)
        self.A.setdiag(-self.A.sum(axis=1)) 
        A = self.A.tocsr()
        
        mesh.times["assemble_tpfa_matrix"] = round(time.time() - start_time, 5)
        if verbose: 
            print("Time to assemble TPFA matrix: \t\t", mesh.times["assemble_tpfa_matrix"], "s")
    
    def set_boundary_conditions(self, bc, f, q):
        """
        Define as condições de contorno

        Parametros
        ----------
        bc: str
            Tipo de condição de contorno
        f: tuple
            lambda f(x,y,z): valor
        q: np.array
            Vetor de fontes
        ----------
        """
        
        start_time = time.time()
        if bc == "dirichlet":
            pass
        elif bc == "neumann":
           pass
        
        mesh.times["set_boundary_conditions"] = round(time.time() - start_time, 5)
        if verbose: 
            print("Time to set {} bc's: \t\t".format(bc), mesh.times["set_boundary_conditions"], "s")
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
        np.set_printoptions(suppress=True)
        if check:
            check_time = time.time()
            assert np.allclose(self.A.dot(self.p), q.todense().T[0])
            print("CHECK ({}): TPFA system solved correctly".format(round(time.time() - check_time,5)))
        
        mesh.times["solve_tpfa"] = round(time.time() - start_time, 5)
        if verbose: 
            print("Time to solve TPFA system: \t\t", mesh.times["solve_tpfa"], "s")
    
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

        

        # Write the rectilinear grid to a .vtk filea
        writer = vtk.vtkDataSetWriter()
        writer.SetFileName("./tmp/{}_mesh.vtk".format(self.mesh.name))
        writer.SetInputData(grid)
        writer.Write() 

if __name__ == "__main__":
    main()