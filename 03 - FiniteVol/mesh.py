import numpy as np
import cupy as cp
import vtk
import matplotlib.pyplot as plt


from vtk.util import numpy_support
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve

from scipy.sparse import csr_matrix, lil_matrix, dia_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning

from pypardiso import spsolve as pd_spsolve

import time
import warnings
USE_GPU = False

class Node:
   def __init__(self, type_node, internal, index, i = 0, j = 0, k = 0, x = 0, y = 0, z = 0):
    """
    Inicializa um nó da malha

    Atributos:
    ----------
    x, y, z: float
        Coordenadas do nó
    type_node: string
        Tipo do nó (volume ou hface, lface ou wface)
    internal : bool
        Indica se o nó é interno ou não
    index: int
        Índice do nó na malha (Esquerda para direita, de cima para baixo)
        Depende do tipo de nó 
    adjacents: list
        Lista de nós adjacentes e suas distâncias
        Se o nó for do tipo volume, serão nós de face
        Se o nó for do tipo face, serão nós de volume
    coeff: list
        Lista de coeficientes de cada nó
        Se o nó for do tipo volume, será uma lista de no máximo 5 coeficientes (cima, direita, centro, esquerda, baixo)
        Se o nó for do tipo face, será um valor único
    """
    self.x, self.y, self.z = x, y, z
    self.type_node = type_node
    self.internal = internal
    self.index = index
    self.i, self.j, self.k = i, j, k

    
    def __repr__(self):
        return f"Node({self.coords}, {self.type_node}, {self.internal}, {self.index})"

    def __str__(self):
        return f"Node({self.coords}, {self.type_node}, {self.internal}, {self.index})"

class Mesh:
    def __init__(self):
        """
        Inicializa uma malha de linhas (1D), quadrados (2D) ou cubos (3D)

        Atributos:
        ----------
        volumes : np.array
            Lista de nós do tipo volume
        faces : np.array
            Lista de nós do tipo face

        """
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
        self.name = None

        self.axis_attibutes = None
        self.dimension = None
        self.faces_adjacents = None

        self.nx, self.ny, self.nz = 1, 1, 1
        self.dx, self.dy, self.dz = 0, 0, 0

        self.nhfaces, self.nlfaces, self.nwfaces = 0, 0, 0
        self.nvols, self.nfaces = 0, 0

        self.volumes_trans = None
        self.faces_trans = None

        self.A = None
        self.p = None
        self.q = None

        self.dirichlet_points = None
        self.dirichlet_values = None
        self.neumann_points = None
        self.neumann_values = None

        self.internal_volumes = None
        self.boundary_volumes = None
    def assemble_mesh(self, axis_attibutes, name = "Mesh " + str(np.random.randint(0, 1000))):
        """
        Monta a malha de acordo com os atributos passados

        Parâmetros:
        -----------
        axis_attibutes : list of tuples
            Lista de tuplas com os atributos de cada eixo
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: [(nx, dx)]
                     [(nx, dx), (ny, dy)]
                     [(nx, dx), (ny, dy), (nz, dz)] 

        """
        start_time = time.time()
        self.name = name
        #print("Creating mesh {} with {} elements".format(self.name, nvols)")
        self.axis_attibutes = axis_attibutes
        self.dimension = len(axis_attibutes)
        if(self.dimension > 3):
            raise Exception("Número de eixos inválido")

        self.nx = axis_attibutes[0][0]
        self.dx = axis_attibutes[0][1]

        self.ny = axis_attibutes[1][0] if self.dimension > 1 else 1
        self.dy = axis_attibutes[1][1] if self.dimension > 1 else 1

        self.nz = axis_attibutes[2][0] if self.dimension > 2 else 1
        self.dz = axis_attibutes[2][1] if self.dimension > 2 else 1
        
        self.nvols = self._get_num_volumes()

        self.volumes_trans = np.empty((self.nvols), dtype=float)
        

        self.nfaces = self._get_num_faces()

        self.nhfaces = (self.nx + 1) * self.ny * self.nz
        self.nlfaces = self.nx * (self.ny + 1) * self.nz
        self.nwfaces = self.nx * self.ny * (self.nz + 1)

        self.faces_trans = np.empty((self.nfaces), dtype=float)
        self.faces_adjacents = np.empty((self.nfaces, 2), dtype = int)
        
        
        start_quick_time = time.time()
        self._assemble_adjacents()
        print("Time to assemble adjs in mesh {}: \t\t".format(self.name), round(time.time() - start_quick_time, 5), "s")

        assert self.faces_adjacents.shape == (self.nfaces, 2)

        print("Time to assemble mesh {}: \t\t\t".format(self.name), round(time.time() - start_time, 5), "s")

    def assemble_faces_transmissibilities(self, K):
        """
        Monta as transmissibilidades das faces da malha
        """
        start_time = time.time()

        self.volumes_trans = np.flip(K, 1)
        self.volumes_trans = (np.trace(self.volumes_trans, axis1=3, axis2=4)/3).flatten()
    

        Kh = 2 / (1/K[:, :, 1:, 0, 0] + 1/K[:, :, :-1, 0, 0])
        Kh = np.insert(Kh,  0, K[:, :, 0, 0, 0], axis = 2)
        Kh = np.insert(Kh, Kh.shape[2], K[:, :, -1, 0, 0], axis = 2)
        faces_trans_h = np.flip(Kh, 1).flatten() / self.dx #/ 2

        faces_trans_l = np.empty((0))
        if(self.dimension > 1):
            Kl = 2 / (1/K[:, 1:, :, 1, 1] + 1/K[:, :-1, :, 1, 1])
            Kl = np.insert(Kl,  0, K[:, 0, :, 1, 1], axis = 1)
            Kl = np.insert(Kl, Kl.shape[1], K[:, -1, :, 1, 1], axis = 1)
            faces_trans_l = np.flip(Kl, 1).flatten() / self.dy #/ 2

        faces_trans_w = np.empty((0))
        if(self.dimension > 2):
            Kw = 2 / (1/K[1:, :, :, 2, 2] + 1/K[:-1, :, :, 2, 2])
            Kw = np.insert(Kw,  0, K[0, :, :, 2, 2], axis = 0)
            Kw = np.insert(Kw, Kw.shape[0], K[-1, :, :, 2, 2], axis = 0)
            faces_trans_w = np.flip(Kw, 1).flatten() / self.dz #/ 2
        
        self.faces_trans = np.hstack((faces_trans_h, faces_trans_l, faces_trans_w))
        self.faces_trans = np.hstack((-self.faces_trans, -self.faces_trans))

        print("Time to assemble faces T in mesh {}: \t\t".format(self.name), round(time.time() - start_time, 5), "s")
        
    def assemble_tpfa_matrix(self):
        """
        Monta a matriz de transmissibilidade TPFA
        """
        start_time = time.time()

        row_index_p = self.faces_adjacents[:, 0]
        col_index_p = self.faces_adjacents[:, 1]
        
        row_index = np.hstack((row_index_p, col_index_p))
        col_index = np.hstack((col_index_p, row_index_p))

        
        assert len(row_index) == len(col_index)
        assert len(row_index) == 2 * self.nfaces
        assert len(row_index) == len(self.faces_trans)

        self.A = csr_matrix((self.faces_trans, (row_index, col_index)), shape=(self.nvols, self.nvols))
        self.A.setdiag(0)
        self.A.setdiag(-self.A.sum(axis=1))
        print("Time to assemble TPFA matrix in mesh {}: \t".format(self.name), round(time.time() - start_time, 5), "s")

    def plot(self, options = None):
        """
        Plota a malha (1D, 2D ou 3D)
        """
        if self.dimension == 1 or self.dimension == 2:
            self._plot_2d(options)
        elif self.dimension == 3:
            self._plot_3d(options)
        else:
            raise Exception("Número de eixos inválido")

    def create_vtk(self):
        # Create the rectilinear grid
        grid = vtk.vtkImageData()
        grid.SetDimensions(self.nx + 1, self.ny + 1, self.nz + 1)
        #index = np.arange(self.nvols)
        #i, j, k = index % self.nx, (index // self.nx) % self.ny, index // (self.nx * self.ny)
        #x, y, z = (i + 1/2) * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz
        grid.SetSpacing(self.dx, self.dy, self.dz)
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
        writer.SetFileName("{}_mesh.vtk".format(self.name))
        writer.SetInputData(grid)
        writer.Write() 
    
    def set_boundary_conditions(self, bc, f, mask = False):
        """
        Define as condições de contorno
        """
        if(f[0].shape[0] == 0):
            return
        start_time = time.time()
        # Get x, y and z and indexes of boundary volumes nodes
        self.q = np.zeros(self.nvols) if self.q is None else self.q

        if self.internal_volumes is None or self.boundary_volumes is None:
            index = np.arange(self.nvols)
            i, j, k = index % self.nx, (index // self.nx) % self.ny, index // (self.nx * self.ny)
            self.internal_volumes = self._is_internal_node(i, j, k, "volume")
            self.boundary_volumes = np.logical_not(self.internal_volumes)
        

        if bc == "dirichlet":
            self.dirichlet_points = f[0]
            self.dirichlet_values = f[1]
            
            if not mask:
                # Get indexes of boundary volumes points (x,y,z) -> index
                indexes = self._get_vol_index_from_coords(coords = (f[0][:, 0], f[0][:, 1], f[0][:, 2]))
                indexes = np.where(self.boundary_volumes[indexes] == True)[0]
            else:
                indexes = np.where(np.logical_and(self.dirichlet_points, self.boundary_volumes))[0]
                self.dirichlet_points = self._get_vol_coords_from_index(indexes)
                self.dirichlet_values = self.dirichlet_values[indexes]
            for i in indexes:
                self.A.data[self.A.indptr[i] : self.A.indptr[i + 1]] = 0.
            self.A[indexes, indexes] = 1.
            
            self.A.eliminate_zeros()
            self.q[indexes] = self.dirichlet_values

        elif bc == "neumann":
            self.neumann_points = f[0]
            self.neumann_values = f[1]

            if not mask:
                # Get indexes of boundary volumes points (x,y,z) -> index
                indexes = self._get_vol_index_from_coords(coords = (f[0][:, 0], f[0][:, 1], f[0][:, 2]))
                indexes = np.where(self.boundary_volumes[indexes] == True)[0]
            else:
                indexes = np.where(np.logical_and(self.neumann_points, self.boundary_volumes))[0]
                self.neumann_values = self.neumann_values[indexes]

            self.q[indexes] += self.neumann_values

        print("Time to set {} bc's in mesh {}: \t\t".format(bc, self.name), round(time.time() - start_time, 5), "s")
        
    def solve_tpfa(self, dense = False):
        """
        Resolve o sistema linear da malha
        """
        start_time = time.time()
        # Check if there's no intersection between dirichlet and neumann points
        if dense:
            self.A = self.A.todense()
            self.p = np.linalg.solve(self.A, self.q)
        else:
            if USE_GPU:
                self.p = cp_spsolve(cp_csr_matrix(self.A), cp.array(self.q))
                self.p = cp.asnumpy(self.p)
            else:
                start_time = time.time()
                self.p = pd_spsolve(self.A, self.q)
        #print(np.around(self.A.todense(), 3))
        #print(np.around(self.q, 3))
        #print(np.around(self.p, 3))
        print("Time to solve TPFA system in mesh {}: \t\t".format(self.name), round(time.time() - start_time, 5), "s")

    def _plot_2d(self, options):
        """
        Plota a malha 2D com o índice de cada nó
        """
        (show_coordinates, show_volumes, 
         show_faces, show_adjacents, 
         show_transmissibilities, print_matrices,
         show_solution) = options if options else (False, True, True, False, False, False, False)

        off = 0.1 * (self.dy) * (0.06 if self.dimension == 1 else 2)
        fig, ax = plt.subplots()

        if show_volumes:
            index = np.arange(self.nvols)
            x = (index % self.nx + 1/2) * self.dx
            y = ((index // self.nx) % self.ny + 1/2) * self.dy
            i = index % self.nx
            j = (index // self.nx) % self.ny
            internal = self.internal_volumes
            p = self.p
            
            if not show_solution:
                plt.scatter(x, y, c="b" if internal else "y")
            else:
                if(p.max() == p.min()):
                    norm = np.ones_like(p)
                else:
                    norm = (p - p.min()) / (p.max() - p.min())
                color = norm.reshape(self.nx, self.ny)
                plt.scatter(x, y, c = color, cmap = plt.get_cmap('jet'))  
                plt.colorbar()

            if show_coordinates:
                plt.annotate("i: " + str(i), (x, y + off))
                plt.annotate("j: " + str(j), (x, y + off/2))
            
        if show_faces:
            for face in self.faces():
                plt.scatter(face.x, face.y, c="b" if face.internal else "y")
                plt.annotate(face.index, (face.x, face.y))
                if show_coordinates:
                    plt.annotate("i: " + str(face.i), (face.x, face.y + off))
                    plt.annotate("j: " + str(face.j), (face.x, face.y + off/2))
                if show_transmissibilities:
                    plt.annotate("T: " + str(np.around(self.faces_trans[face.index],4)), (face.x, face.y - off/2))

        if show_adjacents:
            for face_idx, face_adj in enumerate(self.faces_adjacents): 
                for vol_idx in face_adj:
                    face = self._get_face(face_idx)
                    vol = self._get_volume(vol_idx)
                    plt.plot([face.x, vol.x], 
                             [face.y, vol.y], c="b")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Malha {}".format(self.name))
        ax.grid()
        plt.show()

        if print_matrices:
            print("Matriz de transmissibilidade:\n\n", np.around(self.A,4), "\n")
            print("Pressão:\n\n", np.around(self.p,4), "\n")

    def _plot_3d(self, options):
        """
        Plota a malha 3D com o índice de cada nó
        """
        (show_coordinates, show_volumes, 
         show_faces, show_adjacents, 
         show_transmissibilities, print_matrices,
         show_solution) = options if options else (False, True, True, False, False, False, False)
        
        show_boundary = True if show_volumes else True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        off = 0.1 * (self.dy) * 2
        
        if show_volumes:
            index = np.arange(self.nvols)
            i, j, k = index % self.nx, (index // self.nx) % self.ny, index // (self.nx * self.ny)
            x, y, z = (i + 1/2) * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz
            internal = self.internal_volumes
            p = self.p
            nx, ny, nz = self.nx, self.ny, self.nz
            if not show_boundary and self.nx > 2 and self.ny > 2 and self.nz > 2:
                internal_idx = np.where(internal == True)[0]
                x, y, z = x[internal_idx], y[internal_idx], z[internal_idx]
                i, j, k = i[internal_idx], j[internal_idx], k[internal_idx]
                p = p[internal_idx]
                nx, ny, nz = nx - 2, ny - 2, nz - 2
            if not show_solution:
                ax.scatter(x, y, z, c="r" if internal else "g")
            else:
                if(p.max() == p.min()):
                    norm = np.ones_like(p)
                else:
                    norm = (p - p.min()) / (p.max() - p.min())
                color = norm.reshape(nx, ny, nz)
                ax.scatter(x, y, z, c=color, cmap = "jet")
                fig.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=ax)
            if show_coordinates:
                ax.text(x, y, z + off, "i: " + str(i))
                ax.text(x, y, z + 2 * off / 2, "j: " + str(j))
                ax.text(x, y, z + off / 3, "k: " + str(k))
            if show_transmissibilities:
                ax.text(x, y, z - off/2, "T: " +  str(np.around(self.volumes_trans[index], 4)))
            
        if show_faces:
            for face in self.faces():
                ax.scatter(face.x, face.y, face.z, c="b" if face.internal else "y")
                ax.text(face.x, face.y, face.z, face.index)
                if show_coordinates:
                    ax.text(face.x, face.y, face.z + off, "i: " + str(face.i))
                    ax.text(face.x, face.y, face.z + 2 * off / 2, "j: " + str(face.j))
                    ax.text(face.x, face.y, face.z + off / 3, "k: " + str(face.k))
                if show_transmissibilities:
                    ax.text(face.x, face.y, face.z - off/2, "T: " +  str(np.around(self.faces_trans[face.index], 4)))
        
        if show_adjacents:
            for face_idx, face_adj in enumerate(self.faces_adjacents): 
                for vol_idx in face_adj:
                    face = self._get_face(face_idx)
                    vol = self._get_volume(vol_idx)
                    ax.plot([face.x, vol.x], 
                            [face.y, vol.y], 
                            [face.z, vol.z], c="b")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Malha {}".format(self.name))
        ax.grid()
        plt.show()

        if print_matrices:
            print("Matriz de transmissibilidade:\n\n", np.around(self.A,4), "\n")
            print("Pressão:\n\n", np.around(self.p,4), "\n")
            
    def volumes(self):
        """
        Retorna os nós do tipo volume
        """
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    index = i + j * self.nx + k * self.nx * self.ny
                    x = (i + 1/2) * self.dx
                    y = (j + 1/2) * self.dy
                    z = (k + 1/2) * self.dz
                    vol = Node("volume", self._is_internal_node(i, j, k, "volume"), index, i, j, k, x, y, z)
                    yield vol

    def faces(self):
        """
        Monta os nós do tipo face
        """
        for face in self.hfaces():
            yield face
        if self.dimension >= 2:
            for face in self.lfaces():
                yield face
        if self.dimension == 3:
           for face in self.wfaces():
                yield face
        
    def hfaces(self):
        """
        Monta os nós do tipo face de altura
        """
        for i in range(self.nx + 1):
            for j in range(self.ny):
                for k in range(self.nz):
                    index = i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny)
                    x = i * self.dx
                    y = (j + 1/2) * self.dy
                    z = (k + 1/2) * self.dz
                    face = Node("hface", self._is_internal_node(i, j, k, "hface"), index, i, j, k, x, y, z)
                    yield face
        
    def lfaces(self):
        """
        Monta os nós do tipo face de comprimento
        """
        for i in range(self.nx):
            for j in range(self.ny + 1):
                for k in range(self.nz):
                    index = i + j * self.nx + k * self.nx * (self.ny + 1) + self.nhfaces
                    x = (i + 1/2) * self.dx
                    y = j * self.dy
                    z = (k + 1/2) * self.dz
                    face = Node("lface", self._is_internal_node(i, j, k, "lface"), index, i, j, k, x, y, z)
                    yield face
    
    def wfaces(self):
        """
        Monta os nós do tipo face de largura
        """
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz + 1):
                    index = i + j * self.nx + k * self.nx * self.ny + self.nhfaces + self.nlfaces
                    x = (i + 1/2) * self.dx
                    y = (j + 1/2) * self.dy
                    z = k * self.dz
                    face = Node("wface", self._is_internal_node(i, j, k, "wface"), index, i, j, k, x, y, z)
                    yield face
    
    def _assemble_adjacents(self):
        """
        Monta a matriz de adjacência de forma rápida
        """
        hfaces = np.arange(self.nhfaces)
        hfaces_coords = self._get_faces_coords_from_index(hfaces, "hface")
        hfaces_coords = hfaces_coords.T

        assert hfaces_coords.shape == (self.nhfaces, 3)
        assert self.faces_adjacents.shape == (self.nfaces, 2)

        self.faces_adjacents[:self.nhfaces] = self._get_adjacents_to_face(hfaces_coords, "hface")

        if self.dimension >= 2:
            lfaces = np.arange(self.nhfaces, self.nhfaces + self.nlfaces)
            lfaces_coords = self._get_faces_coords_from_index(lfaces, "lface")
            lfaces_coords = lfaces_coords.T
            assert lfaces_coords.shape == (self.nlfaces, 3)

            self.faces_adjacents[self.nhfaces:self.nhfaces + self.nlfaces] = self._get_adjacents_to_face(lfaces_coords, "lface")

            if self.dimension == 3:
                wfaces = np.arange(self.nhfaces + self.nlfaces, self.nhfaces + self.nlfaces + self.nwfaces)
                wfaces_coords = self._get_faces_coords_from_index(wfaces, "wface")
                wfaces_coords = wfaces_coords.T
                assert wfaces_coords.shape == (self.nwfaces, 3)
                
                self.faces_adjacents[self.nhfaces + self.nlfaces:] = self._get_adjacents_to_face(wfaces_coords, "wface")

    def _get_adjacents_to_face(self, face_coords, face_type):
        """
        Retorna os índices dos volumes adjacentes a uma hface
        """
        
        v1x, v1y, v1z = face_coords[:,0], face_coords[:,1], face_coords[:,2]
        v2x, v2y, v2z = (np.array(v1x, copy = True), 
                         np.array(v1y, copy = True), 
                         np.array(v1z, copy = True))

        if face_type == "hface":
            v1x += self.dx/2
            v2x -= self.dx/2
        elif face_type == "lface":
            v1y += self.dy/2
            v2y -= self.dy/2
        elif face_type == "wface":
            v1z += self.dz/2
            v2z -= self.dz/2

        np.clip(v1x, self.dx/2, (self.nx - 1/2) * self.dx, out = v1x)
        np.clip(v1y, self.dy/2, (self.ny - 1/2) * self.dy, out = v1y)
        np.clip(v1z, self.dz/2, (self.nz - 1/2) * self.dz, out = v1z)
        
        np.clip(v2x, self.dx/2, (self.nx - 1/2) * self.dx, out = v2x)
        np.clip(v2y, self.dy/2, (self.ny - 1/2) * self.dy, out = v2y)
        np.clip(v2z, self.dz/2, (self.nz - 1/2) * self.dz, out = v2z)

        v1 = self._get_vol_index_from_coords((v1x, v1y, v1z))
        v2 = self._get_vol_index_from_coords((v2x, v2y, v2z))
        return np.array([v1, v2]).T

    def _is_valid_volume_coords(self, coords):
        """
        Verifica se as coordenadas estão dentro do domínio dos volumes
        """
        x, y, z = coords
        return ((x >= 0 and x <= self.nx * self.dx - 1/2) and 
                (y >= 0 and y <= self.ny * self.dy - 1/2) and 
                (z >= 0 and z <= self.nz * self.dz - 1/2))
    
    def _get_faces_coords_from_index(self, index, face_type):
        """
        Retorna as coordenadas das faces a partir do índice
        """
        if face_type == "hface":
            i = index % (self.nx + 1)
            j = (index // (self.nx + 1)) % self.ny
            k = (index // (self.nx + 1)) // self.ny
            return np.array([i * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz])
           
        elif face_type == "lface":
            index -= self.nhfaces
            i = index % self.nx
            j = (index // self.nx) % (self.ny + 1)
            k = (index // self.nx // (self.ny + 1))
            return np.array([(i + 1/2) * self.dx, j * self.dy, (k + 1/2) * self.dz])
       
        elif face_type == "wface":
            index -= (self.nhfaces + self.nlfaces)
            i = index % self.nx
            j = (index // self.nx) % self.ny
            k = (index // self.nx // self.ny)
            return np.array([(i + 1/2) * self.dx, (j + 1/2) * self.dy, k * self.dz])

    def _get_vol_coords_from_index(self, index):
        """
        Retorna as coordenadas dos volumes a partir do índice
        """
        i = index % self.nx
        j = (index // self.nx) % self.ny
        k = (index // self.nx // self.ny)
        return np.array([(i + 1/2) * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz])

    def _get_vol_index_from_coords(self, coords):
        """
        Retorna o índice do volume a partir das coordenadas
        """
        i = coords[0] // self.dx
        j = coords[1] // self.dy
        k = coords[2] // self.dz
        return  (i + j * self.nx + k * self.nx * self.ny).astype(int)
        
    def _get_volume(self, info):
        """
        Retorna o volume que contém o ponto (x, y, z) ou o índice index
        """
        if type(info) == tuple or type(info) == np.ndarray:
            x, y, z = info
            index = None
        elif type(info) == int or type(info) == np.int64:
            index = info
            x, y, z = None, None, None
        else:
            raise TypeError("info must be a tuple or an int")
        if(index != None):
            i = index % self.nx
            j = int(index / self.nx) % self.ny
            k = int(index / self.nx / self.ny)
            volume = Node("volume", self._is_internal_node(i, j, k, "volume"), index, 
                          i, j, k, (i + 1/2) * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz)
            return volume
        else:
            i = int(x / self.dx - 1/2)
            j = int(y / self.dy - 1/2)
            k = int(z / self.dz - 1/2)
            index = i + j * self.nx + k * self.nx * self.ny

            if(i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz or index >= self.nvols):
                return None
            else:
                volume = Node("volume", self._is_internal_node(i, j, k, "volume"), index, 
                              i, j, k, (i + 1/2) * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz)
                return volume
    
    def _get_face(self, info):
        """
        Retorna a face que contém o ponto (x, y, z) ou o índice index
        """
        if type(info) == tuple:
            x, y, z = info
            index = None
        elif type(info) == int:
            index = info
            x, y, z = None, None, None
        else:
            raise TypeError("info must be a tuple or an int")
        if(index != None):
            if index < self.nhfaces:
                i = index % (self.nx + 1)
                j = int(index / (self.nx + 1)) % self.ny
                k = int(index / (self.nx + 1) / self.ny)
                face = Node("hface", self._is_internal_node(i, j, k, "hface"), index, 
                            i, j, k, i * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz)
                return face
            elif index < self.nhfaces + self.nlfaces:
                index -= self.nhfaces
                i = index % self.nx
                j = int(index / self.nx) % (self.ny + 1)
                k = int(index / self.nx / (self.ny + 1))
                index += self.nhfaces
                face = Node("lface", self._is_internal_node(i, j, k, "vface"), index, 
                            i, j, k, (i + 1/2) * self.dx, j * self.dy, (k + 1/2) * self.dz)
                return face
            elif index < self.nhfaces + self.nlfaces + self.nwfaces:
                index -= (self.nhfaces + self.nlfaces)
                i = index % self.nx
                j = int(index / self.nx) % self.ny
                k = int(index / self.nx / self.ny)
                index += (self.nhfaces + self.nlfaces)
                face = Node("wface", self._is_internal_node(i, j, k, "wface"), index, 
                            i, j, k, (i + 1/2) * self.dx, (j + 1/2) * self.dy, k * self.dz)
                return face
        
        else:
            if x % self.dx == 0:
                i = int(x / self.dx)
                j = int(y / self.dy - 1/2)
                k = int(z / self.dz - 1/2)
                index = i + j * (self.nx + 1) + k * (self.nx + 1) * self.ny
                face = Node("hface", self._is_internal_node(i, j, k, "hface"), index, 
                            i, j, k, i * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz)
                return face
            elif y % self.dy == 0:
                i = int(x / self.dx - 1/2)
                j = int(y / self.dy)
                k = int(z / self.dz - 1/2)
                index = i + j * self.nx + k * self.nx * (self.ny + 1)
                face = Node("lface", self._is_internal_node(i, j, k, "vface"), index, 
                            i, j, k, (i + 1/2) * self.dx, j * self.dy, (k + 1/2) * self.dz)
                return face
            elif z % self.dz == 0:
                i = int(x / self.dx - 1/2)
                j = int(y / self.dy - 1/2)
                k = int(z / self.dz)
                index = i + j * self.nx + k * self.nx * self.ny
                face = Node("wface", self._is_internal_node(i, j, k, "wface"), index, 
                            i, j, k, (i + 1/2) * self.dx, (j + 1/2) * self.dy, k * self.dz)
                return face
            else:
                return None

    def _is_internal_node(self, i, j, k, node_type):
        """
        Retorna se o nó é interno ou não
        """
        if(node_type == "volume"):
            if(self.dimension == 1):
                return np.logical_and(i != 0, i != (self.nx - 1))
            elif(self.dimension == 2):
                return np.logical_and(np.logical_and(i != 0, i != (self.nx - 1)), 
                                      np.logical_and(j != 0, j != (self.ny - 1)))
            elif(self.dimension == 3):
                return np.logical_and(np.logical_and(
                                      np.logical_and(i != 0, i != (self.nx - 1)), 
                                      np.logical_and(j != 0, j != (self.ny - 1))), 
                                      np.logical_and(k != 0, k != (self.nz - 1)))
        if(node_type == "hface"):
            return np.logical_and(i != 0, i != self.nx)
        if(node_type == "lface"):
            return np.logical_and(j != 0, j != self.ny)
        if(node_type == "wface"):
            return np.logical_and(k != 0, k != self.nz)

    def _get_num_volumes(self):
        """
        Retorna o número de volumes da malha
        """
        if(self.dimension == 1):
            return self.nx
        elif(self.dimension == 2):
            return self.nx * self.ny
        elif(self.dimension == 3):
            return self.nx * self.ny * self.nz
        
    def _get_num_faces(self):
        """
        Retorna o número de faces da malha
        """

        # N de faces ->
        # 1D: nfaces = nx + 1
        # 2D: nfaces = nx * ny + nx * (ny + 1) + ny * (nx + 1)
        # 3D: nfaces = nx * ny * nz + nx * ny * (nz + 1) + nx * (ny + 1) * nz + (nx + 1) * ny * nz

        if(self.dimension == 1):
            return self.nx + 1
        elif(self.dimension == 2):
            return self.nx * (self.ny + 1) + self.ny * (self.nx + 1)
        elif(self.dimension == 3):
            return self.nx * self.ny * (self.nz + 1) + self.nx * (self.ny + 1) * self.nz + (self.nx + 1) * self.ny * self.nz