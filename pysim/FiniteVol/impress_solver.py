import numpy as np
import time
import os
import sys
import warnings

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning
#from pypardiso import spsolve as pd_spsolve
from preprocessor.meshHandle.finescaleMesh import FineScaleMesh


def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__

verbose = True


def get_random_tensor(a, b, size, n = 3, m = 3):
    """
    Retorna um array de tamanho size cujos elementos são tensores aleatórios n x m, com valores entre a e b
    Se only_diagonal for True, os elementos fora da diagonal serão iguais a zero, e o tensor será n x n

    """
    if type(size) == int:
        size = (size,)
    return np.random.uniform(a, b, size = size + (n, m))

def get_tensor(a, size, n = 3, m = 3, only_diagonal = True):
    """
    Retorna um array de tamanho size cujos elementos são tensores n x m, com valores iguais a a
    Se only_diagonal for True, os elementos fora da diagonal serão iguais a zero, e o tensor será n x n
    
    """
    if type(size) == int:
        size = (size,)
    return np.full(size + (n, m), a)

class Solver:
    def __init__(self):
        self.volumes_trans_tensor = None
        self.volumes_trans = None
        self.faces_trans = None
        self.internal_faces = None
        self.vols_pairs = None
        self.mesh = None

        self.times = {}
        self.A = None
        self.b = None
        self.p = None
    
    
    def create_mesh(self, meshfile):
        
        block_print()
        self.mesh = FineScaleMesh(meshfile)
        enable_print()
        self.nvols = len(self.mesh.volumes)
        self.areas = np.array(self.mesh.faces.area[:])
        self.volumes = np.array(self.mesh.volumes.volume[:])

        return self.mesh
    
    def solve(self, mesh, K, q, fd, fn, create_vtk = False, check = False, an_sol = None):
        
        self.mesh = mesh
        self.internal_faces = mesh.faces.internal[:]
        self.boundary_faces = mesh.faces.boundary[:]
        self.vols_pairs = mesh.faces.bridge_adjacencies(self.internal_faces, 2, 3)

        
        K = np.flip(K, axis = 1)
        K = np.reshape(K, (self.nvols, 9))
        self.mesh.permeability[:] = K

        self._assemble_normal_vectors()
        self._assemble_faces_transmissibilities()
        self._assemble_tpfa_matrix(q)
        self._set_boundary_conditions("dirichlet", fd)
        self._set_boundary_conditions("neumann", fn)
        self._solve_tpfa(check=check)

        if an_sol:
            coords = self.mesh.volumes.center[:]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            self.mesh.an_sol[:] = an_sol(x, y, z)
        if create_vtk:
            self._create_vtk()
        return self.p

    def _assemble_normal_vectors(self):
        """
        Monta os vetores normais das faces internas da malha
        """
        start_time = time.time()

        faces_nodes = self.mesh.faces.connectivities[self.internal_faces]

        i = self.mesh.nodes.coords[faces_nodes[:, 0]]
        j = self.mesh.nodes.coords[faces_nodes[:, 1]]
        k = self.mesh.nodes.coords[faces_nodes[:, 2]]

        n_vols_pairs = len(self.vols_pairs)

        volumes_centers = self.mesh.volumes.center[self.vols_pairs.flatten()].reshape((n_vols_pairs, 2, 3))
        faces_centers = self.mesh.faces.center[self.internal_faces]

       

        N = np.cross(i - j, k - j) # Vetor Normal a Face 
        self.N_norm = N / np.linalg.norm(N, axis = 1).reshape((len(N), 1))

        L = self.mesh.volumes.center[self.vols_pairs[:, 0]]
        R = self.mesh.volumes.center[self.vols_pairs[:, 1]]

        vL = faces_centers - L
        vR = faces_centers - R

        orientation = np.sign(np.einsum("ij,ij->i", N, vL))
        N[orientation < 0] = -N[orientation < 0]
        self.h_L = np.abs(np.einsum("ij,ij->i", self.N_norm, vL)) # Row-wise dot product
        self.h_R = np.abs(np.einsum("ij,ij->i", self.N_norm, vR))

        self.times["Montar Vetores Normais"] = time.time() - start_time
        if verbose: 
            print("Time to assemble normal vectors: \t", round(self.times["Montar Vetores Normais"], 5), "s")

    def _assemble_faces_transmissibilities(self):
        """
        Monta as transmissibilidades das faces da malha
        """
        start_time = time.time()
        
        n_vols_pairs = len(self.vols_pairs)

        KL = self.mesh.permeability[self.vols_pairs[:, 0]].reshape((n_vols_pairs, 3, 3))
        KR = self.mesh.permeability[self.vols_pairs[:, 1]].reshape((n_vols_pairs, 3, 3))

        KnL = np.einsum("ij,ikj->ik", self.N_norm, KL) # Normal Components
        KnR = np.einsum("ij,ikj->ik", self.N_norm, KR) 

        KnL = np.einsum("ij,ij->i", self.N_norm, KnL) 
        KnR = np.einsum("ij,ij->i", self.N_norm, KnR) 

        Keq = (KnL * KnR) / ((KnL * self.h_R) + (KnR * self.h_L))
        self.faces_trans = Keq * self.areas[self.internal_faces]

        self.times["Montar Transmissibilidade das Faces"] = time.time() - start_time
        if verbose: 
            print("Time to assemble faces T: \t\t", round(self.times["Montar Transmissibilidade das Faces"], 5), "s")
    
    def _assemble_tpfa_matrix(self, q):
        """
        Monta a matriz de transmissibilidade TPFA
        """
        start_time = time.time()

        row_index_p = self.vols_pairs[:, 0]
        col_index_p = self.vols_pairs[:, 1]
        
        row_index = np.hstack((row_index_p, col_index_p))
        col_index = np.hstack((col_index_p, row_index_p))
        data = np.hstack((-self.faces_trans, -self.faces_trans))

        self.A = csr_matrix((data, (row_index, col_index)), 
                             shape=(self.nvols, self.nvols))
        self.b = np.array(q, dtype=float)

        self.A = self.A.tolil()
        self.A.setdiag(-self.A.sum(axis=1)) 
        self.A = self.A.tocsr()

        self.times["Montar Matriz A do TPFA"] = time.time() - start_time
        if verbose: 
            print("Time to assemble TPFA matrix: \t\t", round(self.times["Montar Matriz A do TPFA"], 5), "s")
    
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
            coords = self.mesh.faces.center[self.boundary_faces]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

            d_values = f(x, y, z)
            d_nodes = self.boundary_faces[d_values != None]
            d_values = d_values[d_values != None].astype('float')
            d_volumes = self.mesh.faces.bridge_adjacencies(d_nodes, 2, 3).astype('int').flatten()
            d_vertex = self.mesh.faces.connectivities(d_nodes)
            
            self.d_values = d_values
            self.d_nodes = d_nodes

            C = self.mesh.volumes.center[d_volumes]

            i = self.mesh.nodes.coords[d_vertex[:, 0]]
            j = self.mesh.nodes.coords[d_vertex[:, 1]]
            k = self.mesh.nodes.coords[d_vertex[:, 2]]

            faces_centers = self.mesh.faces.center[d_nodes]
            vC = faces_centers - C

            N = np.abs(np.cross(i - j, k - j))
            N_norm = N / np.linalg.norm(N, axis = 1).reshape((len(N), 1))
            Area = self.areas[d_nodes]
            orientation = np.sign(np.einsum("ij,ij->i", N, vC))
            N[orientation < 0] = -N[orientation < 0]
            
            h = np.abs(np.einsum("ij,ij->i", N_norm, vC))

            K = self.mesh.permeability[d_volumes].reshape((len(d_volumes), 3, 3))
            K = np.einsum("ij,ikj->ik", N_norm, K)
            K = np.einsum("ij,ij->i", N_norm, K)
            
            self.A_d = np.zeros(shape = (self.nvols))
            
            self.A_d[d_volumes] += (K * Area) / h 

            row_index = np.arange(self.nvols)
            col_index = np.arange(self.nvols)
            
            self.A_d = csr_matrix((self.A_d, (row_index, col_index)), 
                                   shape=(self.nvols, self.nvols))
            
            self.b_d = np.zeros(shape = (self.nvols))

            np.add.at(self.b_d, d_volumes, (d_values * (K * Area) / h ))

        elif bc == "neumann":
            coords = self.mesh.faces.center[self.boundary_faces]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            
            n_values = f(x, y, z)
            self.mesh.neumann = n_values
            n_nodes = self.boundary_faces[n_values != None]
            n_values = n_values[n_values != None].astype('float')
            
            n_volumes = self.mesh.faces.bridge_adjacencies(n_nodes, 2, 3).astype('int').flatten()
            self.n_values = n_values
            self.n_nodes = n_nodes
            self.b_n = np.zeros(shape = (self.nvols))
            np.add.at(self.b_n, n_volumes, (n_values * self.volumes[n_volumes]))

        self.times["Condição de contorno - {}".format((bc))] = time.time() - start_time
        if verbose: 
            print("Time to set {} bc's: \t\t".format(bc), round(self.times["Condição de contorno - {}".format((bc))], 5), "s")
        

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
            self.A_d.eliminate_zeros()
            start_time = time.time()
            self.p = spsolve(self.A + self.A_d, self.b + self.b_d + self.b_n)
        np.set_printoptions(suppress=True)
        #print(self.A.todense())
        #print(self.p)
        #print(self.b)
        if check:
            check_time = time.time()
            assert np.allclose((self.A + self.A_d).dot(self.p), self.b + self.b_d + self.b_n)
            print("CHECK ({}): TPFA system solved correctly".format(round(time.time() - check_time,5)))

        
        self.times["Resolver o Sistema TPFA"] = time.time() - start_time
        if verbose: 
            print("Time to solve TPFA system: \t\t", round(self.times["Resolver o Sistema TPFA"],5), "s")
    
    def _create_vtk(self):
        vtk_time = time.time()
        meshset = self.mesh.core.mb.create_meshset()
        coords = self.mesh.volumes.center[:]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        self.mesh.nu_sol[:] = self.p
        self.mesh.x, self.mesh.y, self.mesh.z = x, y, z
        self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_volumes)
        #self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_faces)
        self.mesh.core.mb.write_file("./mesh_impress.vtk", [meshset])
        self.times["Criar arquivo VTK"] = time.time() - vtk_time
        if verbose: 
            print("Time to create vtk file: \t\t", round(self.times["Criar arquivo VTK"], 5), "s")
