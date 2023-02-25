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
        mesh = FineScaleMesh(meshfile)
        return mesh
        
    def solve(self, mesh, K, q, fd, fn, create_vtk = False, check = False, an_sol = None):
        
        self.mesh = mesh
        self.internal_faces = mesh.faces.internal[:]
        self.vols_pairs = mesh.faces.bridge_adjacencies(self.mesh.faces.all, 2, 3)
        b_vols = self.vols_pairs[mesh.faces.boundary]
        b_vols = [np.array([x[0], x[0]]) for x in b_vols]
        self.vols_pairs[mesh.faces.boundary] = b_vols
        self.vols_pairs = np.array([np.array(x) for x in self.vols_pairs])


        self.nvols = len(mesh.volumes)
        K = np.flip(K, axis = 1)
        K = np.reshape(K, (self.nvols, 9))
        self.mesh.permeability[:] = K

        self._assemble_normal_vectors()
        self._assemble_faces_transmissibilities()
        self._assemble_tpfa_matrix(q)
        self._set_boundary_conditions("dirichlet", fd)
        self._set_boundary_conditions("neumann", fn)
        self._solve_tpfa(check=check)
        if create_vtk:
            self._create_vtk(an_sol = an_sol)
        return self.p

    def _assemble_normal_vectors(self):
        """
        Monta os vetores normais das faces internas da malha
        """
        start_time = time.time()

        faces_nodes = self.mesh.faces.bridge_adjacencies(self.mesh.faces.all, 0, 0)
        x_idx = faces_nodes[:, 0]
        y_idx = faces_nodes[:, 1]
        z_idx = faces_nodes[:, 2]

        x = self.mesh.nodes.coords[x_idx]
        y = self.mesh.nodes.coords[y_idx]
        z = self.mesh.nodes.coords[z_idx]

        n_vols_pairs = len(self.vols_pairs)
        volumes_centers_flat = self.mesh.volumes.center[self.vols_pairs.flatten()]

        volumes_centers = volumes_centers_flat.reshape((n_vols_pairs, 2, 3))

        yL = y - volumes_centers[:, 0]

        self.Ns = 0.5 * np.cross(x - y, z - y)
        self.Ns_norm = np.linalg.norm(self.Ns, axis=1)

        N_sign = np.sign(np.einsum("ij,ij->i", yL, self.Ns))
        (self.vols_pairs[N_sign < 0, 0],
         self.vols_pairs[N_sign < 0, 1]) = (self.vols_pairs[N_sign < 0, 1],
                                               self.vols_pairs[N_sign < 0, 0])

        L = self.mesh.volumes.center[self.vols_pairs[:, 0]]
        R = self.mesh.volumes.center[self.vols_pairs[:, 1]]


        yL = y - L
        yR = y - R

        self.h_L = np.abs(np.einsum("ij,ij->i", self.Ns, yL) / self.Ns_norm)
        self.h_R = np.abs(np.einsum("ij,ij->i", self.Ns, yR) / self.Ns_norm)

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

        KnL_pre = np.einsum("ij,ikj->ik", self.Ns, KL)
        KnR_pre = np.einsum("ij,ikj->ik", self.Ns, KR)

        KnL = np.einsum("ij,ij->i", KnL_pre, self.Ns) / self.Ns_norm ** 2
        KnR = np.einsum("ij,ij->i", KnR_pre, self.Ns) / self.Ns_norm ** 2

        Kn_L = KnL[:]
        Kn_R = KnR[:]

        Kn_prod = Kn_L * Kn_R
        Keq = Kn_prod / ((Kn_L * self.h_R) + (Kn_R * self.h_L))
        self.faces_trans = Keq * self.Ns_norm

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
        self.A.setdiag(0)
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
            coords = self.mesh.faces.center[self.mesh.faces.boundary]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            boundary_faces = self.mesh.faces.boundary
            d_values = f(x, y, z)
            self.mesh.dirichlet = d_values
            d_nodes = boundary_faces[d_values != None]
            d_values = d_values[d_values != None].astype(float)
            d_volumes = self.mesh.faces.bridge_adjacencies(d_nodes, 2, 3).astype('int')
            self.A = self.A.tolil()
            self.A[d_volumes, d_volumes] += self.faces_trans[d_nodes].reshape(len(d_nodes), 1)
            self.b[d_volumes] += (d_values * self.faces_trans[d_nodes]).reshape(len(d_nodes),1)
            #self.A[d_volumes] = 0.
            #self.A[d_volumes, d_volumes] = 1.
            #self.b[d_volumes] = d_values.reshape(len(d_values), 1)
            self.A = self.A.tocsr()

        elif bc == "neumann":
            coords = self.mesh.faces.center[self.mesh.faces.boundary]
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            boundary_faces = self.mesh.faces.boundary
            n_values = f(x, y, z)
            self.mesh.neumann = n_values
            n_nodes = boundary_faces[n_values != None]
            n_values = n_values[n_values != None].astype(float)
            n_volumes = self.mesh.faces.bridge_adjacencies(n_nodes, 2, 3).flatten()
           
            self.b[n_volumes] += n_values * self.mesh.volumes.volume[n_volumes]
        
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
            start_time = time.time()
            self.p = spsolve(self.A, self.b)
        np.set_printoptions(suppress=True)
        #print(self.A.todense())
        #print(self.p)
        #print(self.b)
        if check:
            check_time = time.time()
            assert np.allclose(self.A.dot(self.p), self.b)
            print("CHECK ({}): TPFA system solved correctly".format(round(time.time() - check_time,5)))

        
        self.times["Resolver o Sistema TPFA"] = time.time() - start_time
        if verbose: 
            print("Time to solve TPFA system: \t\t", round(self.times["Resolver o Sistema TPFA"],5), "s")
    
    def _create_vtk(self, an_sol = None):
        vtk_time = time.time()
        meshset = self.mesh.core.mb.create_meshset()
        coords = self.mesh.volumes.center[:]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        self.mesh.an_sol[:] = an_sol(x, y, z)
        self.mesh.nu_sol[:] = self.p
        self.mesh.x, self.mesh.y, self.mesh.z = x, y, z
        self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_volumes)
        #self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_faces)
        self.mesh.core.mb.write_file("./mesh_impress.vtk", [meshset])
        self.times["Criar arquivo VTK"] = time.time() - vtk_time
        if verbose: 
            print("Time to create vtk file: \t\t", round(self.times["Criar arquivo VTK"], 5), "s")
