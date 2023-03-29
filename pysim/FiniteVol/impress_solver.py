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
from memory_profiler import profile

def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout.close()
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
        self.permeability = None
        
    def create_mesh(self, meshfile):
        if type(meshfile) == str:
            block_print()
            self.mesh = FineScaleMesh(meshfile)
            self.nvols = len(self.mesh.volumes)
            enable_print()
            self.areas = np.array(self.mesh.faces.area[:])
            faces_by_volume = self.mesh.volumes.bridge_adjacencies(self.mesh.volumes.all, 3, 2)
            nfaces = len(faces_by_volume[0])
            if nfaces == 6:
                faces_flat = faces_by_volume.flatten()
                area_by_face = self.areas[faces_flat].reshape((self.nvols, nfaces))
                self.volumes = np.prod(area_by_face, axis = 1) ** (1 / 4)
                #assert np.allclose(self.volumes, np.array(self.mesh.volumes.volume[:]))
                
            else:
                self.volumes = np.array(self.mesh.volumes.volume[:])
        else:
            block_print()
            self.mesh = FineScaleMesh(meshfile[0])
            self.nvols = len(self.mesh.volumes)
            enable_print()
            self.areas = np.array(meshfile[1])
            self.volumes = np.array(meshfile[2])

        
        return self.mesh
    
    def solve(self, mesh, K, q, fd, fn, create_vtk = False, check = False, an_sol = None, name = "IMPRESS"):
        self.name = name
        start_time = time.time()
        self.mesh = mesh
        self.internal_faces = mesh.faces.internal[:]
        self.boundary_faces = mesh.faces.boundary[:]
        self.boundary_without_conditions = self.boundary_faces
        self.vols_pairs = mesh.faces.bridge_adjacencies(self.internal_faces, 2, 3)
        
    
        K = np.flip(K, axis = 0)
        K = np.reshape(K, (self.nvols, 9))
        
        self.permeability = K
        for i in range(self.nvols):
            self.mesh.permeability[i] = K[i]

        self._assemble_normal_vectors()
        self._assemble_faces_transmissibilities()
        self._assemble_tpfa_matrix(q, check)

        self.A_d = np.zeros(shape = (self.nvols))
        self.b_d = np.zeros(shape = (self.nvols))
        self.b_n = np.zeros(shape = (self.nvols))

        self._set_boundary_conditions("dirichlet", fd)
        self._set_boundary_conditions("neumann", fn)
        self.times["Montar Sistema TPFA"] = time.time() - start_time
        if verbose: 
            print("Time to assemble TPFA matrix: \t\t", round(self.times["Montar Sistema TPFA"], 5), "s")
    
        self._solve_tpfa(check=check)
        if an_sol and create_vtk:
            coords = self.mesh.volumes.center[:]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            
            self.mesh.an_sol[:] = an_sol(x, y, z)
            self.mesh.error[:] = ((an_sol(x, y, z) - self.p) ** 2  * self.volumes) / (an_sol(x, y, z) * self.volumes) 
            self._create_vtk()
        elif create_vtk:
            self._create_vtk()
        elif an_sol:
            self.an_sol = an_sol
        return self.p

    def _get_normal(self, faces_index, volumes_pairs = None):
        faces_nodes = self.mesh.faces.connectivities[faces_index]

        i = self.mesh.nodes.coords[faces_nodes[:, 0]]
        j = self.mesh.nodes.coords[faces_nodes[:, 1]]
        k = self.mesh.nodes.coords[faces_nodes[:, 2]]

        if volumes_pairs is None:
            volumes_pairs = self.mesh.faces.bridge_adjacencies(faces_index, 2, 3)
        
        n_vols_pairs = volumes_pairs.shape[0] 
        n_vols = volumes_pairs.shape[1] if len(volumes_pairs.shape) > 1 else 1

        volumes_centers = self.mesh.volumes.center[volumes_pairs.flatten()].reshape((n_vols_pairs, n_vols, 3))
        faces_centers = self.mesh.faces.center[faces_index]

        L = volumes_centers[:, 0]

        vL = faces_centers - L

        if n_vols > 1:   
            R = volumes_centers[:, 1]
            vR = faces_centers - R
        else:
            vR = None

        #N = np.cross(i - j, k - j) # Vetor Normal a Face 
        #N_test = np.sign(np.einsum("ij,ij->i", vL, N))
        #i[N_test < 0], k[N_test < 0] = k[N_test < 0], i[N_test < 0]
        N = np.cross(i - j, k - j) # Vetor Normal a Face 
        orientation = np.sign(np.einsum("ij,ij->i", N, vL))
        N[orientation < 0] = -N[orientation < 0]

        return N, vL, vR
    
    def _assemble_normal_vectors(self):
        """
        Monta os vetores normais das faces internas da malha
        """
        start_time = time.time()

        N, vL, vR = self._get_normal(self.internal_faces, self.vols_pairs)
        self.N_norm = N / np.linalg.norm(N, axis = 1).reshape((len(N), 1))

       
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

        KL = self.permeability[self.vols_pairs[:, 0]].reshape((n_vols_pairs, 3, 3))
        KR = self.permeability[self.vols_pairs[:, 1]].reshape((n_vols_pairs, 3, 3))

        KnL = np.einsum("ij,ikj->ik", self.N_norm, KL) # Normal Components
        KnR = np.einsum("ij,ikj->ik", self.N_norm, KR) 

        KnL = np.einsum("ij,ij->i", self.N_norm, KnL) 
        KnR = np.einsum("ij,ij->i", self.N_norm, KnR) 

        Keq = (KnL * KnR) / ((KnL * self.h_R) + (KnR * self.h_L))
        self.faces_trans = Keq * self.areas[self.internal_faces]

        self.times["Montar Transmissibilidade das Faces"] = time.time() - start_time
        if verbose: 
            print("Time to assemble faces T: \t\t", round(self.times["Montar Transmissibilidade das Faces"], 5), "s")
    
    def _assemble_tpfa_matrix(self, q, check):
        """
        Monta a matriz de transmissibilidade TPFA
        """
        

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
        if check:
            assert np.allclose(self.A.sum(axis = 1), np.zeros(shape = (self.nvols, 1)))

        
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
            
            mask = np.isin(d_nodes, self.boundary_without_conditions)
            d_nodes = d_nodes[mask == True]
            d_values = d_values[mask == True]
            self.boundary_without_conditions = np.setdiff1d(self.boundary_without_conditions, d_nodes)

            d_volumes = self.mesh.faces.bridge_adjacencies(d_nodes, 2, 3).astype('int').flatten()
            
            self.d_values = d_values
            self.d_nodes = d_nodes

            N, vL, vR = self._get_normal(d_nodes, d_volumes)
            N_norm = N / np.linalg.norm(N, axis = 1).reshape((len(N), 1))

            Area = self.areas[d_nodes]
            
            h = np.abs(np.einsum("ij,ij->i", N_norm, vL))

            K = self.permeability[d_volumes].reshape((len(d_volumes), 3, 3))
            K = np.einsum("ij,ikj->ik", N_norm, K)
            K = np.einsum("ij,ij->i", N_norm, K)
            
            
            np.add.at(self.A_d, d_volumes, (K * Area) / h)

            row_index = np.arange(self.nvols)
            col_index = np.arange(self.nvols)
            
            self.A_d = csr_matrix((self.A_d, (row_index, col_index)), 
                                   shape=(self.nvols, self.nvols))
            
            np.add.at(self.b_d, d_volumes, d_values * (K * Area) / h)
            self.d_N_norm = N_norm
            

        elif bc == "neumann":
            coords = self.mesh.faces.center[self.boundary_faces]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            
            n_values = f(x, y, z)
            n_nodes = self.boundary_faces[n_values != None]
            n_values = n_values[n_values != None].astype('float')
            
            mask = np.isin(n_nodes, self.boundary_without_conditions)
            n_nodes = n_nodes[mask == True]
            n_values = n_values[mask == True]
            self.boundary_without_conditions = np.setdiff1d(self.boundary_without_conditions, n_nodes)

            if len(n_nodes) > 1:
                n_volumes = self.mesh.faces.bridge_adjacencies(n_nodes, 2, 3).astype('int').flatten()
                self.n_values = n_values
                self.n_nodes = n_nodes
                np.add.at(self.b_n, n_volumes, n_values * self.volumes[n_volumes])
                #self.b_n[n_volumes] += (n_values * self.volumes[n_volumes])

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
            #self.A_d.eliminate_zeros()
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
        self.mesh.flux[:] = self.b + self.b_d + self.b_n
        self.mesh.x, self.mesh.y, self.mesh.z = x, y, z
        self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_volumes)
        #self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_faces)
        self.mesh.core.mb.write_file("./mesh_{}.vtk".format(self.name), [meshset])
        self.times["Criar arquivo VTK"] = time.time() - vtk_time
        if verbose: 
            print("Time to create vtk file: \t\t", round(self.times["Criar arquivo VTK"], 5), "s")
