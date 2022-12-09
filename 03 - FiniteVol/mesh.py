import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

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
        self.volumes = None
        self.faces = None
        self.axis_attibutes = None
        self.dimension = None
        self.volumes_adjacents = None
        self.faces_adjacents = None
        self.nx, self.ny, self.nz = 1, 1, 1
        self.dx, self.dy, self.dz = 0, 0, 0
        self.nhfaces, self.nlfaces, self.nwfaces = 0, 0, 0
        self.nvols, self.nfaces = 0, 0
        self.volumes_trans = None
        self.faces_trans = None
    
    def assemble_mesh(self, axis_attibutes):
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
        self.volumes = np.empty((self.nvols), dtype=Node)
        self.volumes_trans = np.empty((self.nvols), dtype=float)
        self.volumes_adjacents = np.empty((self.nvols, 0), dtype = Node)
        

        self.nfaces = self._get_num_faces()
        self.faces = np.empty((self.nfaces), dtype=Node)
        self.faces_trans = np.empty((self.nfaces), dtype=float)
        self.faces_adjacents = np.empty((self.nfaces, 2), dtype = int)
        

        start_vols_time = time.time()
        self._assemble_volumes()
        print("Time required to assemble volumes: ", round(time.time() - start_vols_time, 5), "s")

        start_faces_time = time.time()
        self._assemble_faces()
        print("Time required to assemble faces: ", round(time.time() - start_faces_time, 5), "s")
        
        start_adjs_time = time.time()
        self._assemble_adjacents()
        print("Time required to assemble adjs: ", round(time.time() - start_adjs_time, 5), "s")

        print("Time required to assemble mesh: ", round(time.time() - start_time, 5), "s")
    

    def assemble_faces_transmissibilities(self, K):
        """
        Monta as transmissibilidades das faces da malha
        """
        start_time = time.time()

        self.volumes_trans = np.flip(K, 1).flatten()

        Kh = 2 / (1/K[:, :, 1:] + 1/K[:, :, :-1])
        Kh = np.insert(Kh,  0, K[:, :, 0], axis = 2)
        Kh = np.insert(Kh, self.nx, K[:, :, -1], axis = 2)
        
        Kl = 2 / (1/K[:, 1:, :] + 1/K[:, :-1, :])
        Kl = np.insert(Kl,  0, K[:, 0, :], axis = 1)
        Kl = np.insert(Kl, self.ny, K[:, -1, :], axis = 1)
        
        Kw = 2 / (1/K[1:, :, :] + 1/K[:-1, :, :])
        Kw = np.insert(Kw,  0, K[0, :, :], axis = 0)
        Kw = np.insert(Kw, self.nz, K[-1, :, :], axis = 0)
    
        faces_trans_h = np.flip(Kh, 1).flatten() / self.dx
        faces_trans_l = np.flip(Kl, 1).flatten() / self.dy if self.dimension > 1 else np.empty((0))
        faces_trans_w = np.flip(Kw, 1).flatten() / self.dz if self.dimension > 2 else np.empty((0))

        self.faces_trans = np.hstack((faces_trans_h, faces_trans_l, faces_trans_w))
        self.faces_trans = np.hstack((-self.faces_trans, -self.faces_trans))

        print("Time required to assemble faces transmissibilities: ", round(time.time() - start_time, 5), "s")
        
    def assemble_tpfa_matrix(self):
        """
        Monta a matriz de transmissibilidade TPFA
        """
        start_time = time.time()

        row_index_p = self.faces_adjacents[:, 0]
        col_index_p = self.faces_adjacents[:, 1]
        row_index = np.hstack((row_index_p, col_index_p))
        col_index = np.hstack((col_index_p, row_index_p))

        A = csr_matrix((self.faces_trans, (row_index, col_index)), shape=(self.nvols, self.nvols))
        A.setdiag(0)
        A.setdiag(-A.sum(axis=1))

        print("Time required to assemble TPFA matrix: ", round(time.time() - start_time, 5), "s")
        return A

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
    
    def _plot_2d(self, options):
        """
        Plota a malha 2D com o índice de cada nó
        """
        (show_coordinates, show_volumes, 
         show_faces, show_adjacents, 
         show_transmissibilities, show_A) = options if options else (False, True, True, False, False, False)

        off = 0.1 * (self.dy) * (0.06 if self.dimension == 1 else 2)
        fig, ax = plt.subplots()
        if show_volumes:
            for volume in self.volumes:
                ax.scatter(volume.x, volume.y, c="r" if volume.internal else "g")
                ax.annotate(volume.index, (volume.x, volume.y))
                if show_coordinates:
                    ax.annotate("i: " + str(volume.i), (volume.x, volume.y + off))
                    ax.annotate("j: " + str(volume.j), (volume.x, volume.y + off/2))
                if show_transmissibilities:
                    ax.annotate("T: " + str(np.around(self.volumes_trans[volume.index],4)), (volume.x, volume.y - off/2))

        if show_faces:
            for face in self.faces:
                ax.scatter(face.x, face.y, c="b" if face.internal else "y")
                ax.annotate(face.index, (face.x, face.y))
                if show_coordinates:
                    ax.annotate("i: " + str(face.i), (face.x, face.y + off))
                    ax.annotate("j: " + str(face.j), (face.x, face.y + off/2))
                if show_transmissibilities:
                    ax.annotate("T: " + str(np.around(self.faces_trans[face.index],4)), (face.x, face.y - off/2))

        if show_adjacents:
            for face_idx, face_adj in enumerate(self.faces_adjacents): 
                for vol_idx in face_adj:
                    ax.plot([self.faces[face_idx].x, self.volumes[vol_idx].x], [self.faces[face_idx].y, self.volumes[vol_idx].y], c="b")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Malha {}D".format(self.dimension))
        ax.grid()
        plt.show()

        if show_A:
            print("Matriz de transmissibilidade:\n\n", np.around(self.assemble_tpfa_matrix().todense(),4), "\n")

    def _plot_3d(self, options):
        """
        Plota a malha 3D com o índice de cada nó
        """
        (show_coordinates, show_volumes, 
         show_faces, show_adjacents, 
         show_transmissibilities, show_A) = options if options else (False, True, True, False, False, False)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        off = 0.1 * (self.dy) * 2
        
        if show_volumes:
            for volume in self.volumes:
                ax.scatter(volume.x, volume.y, volume.z, c="r" if volume.internal else "g")
                ax.text(volume.x, volume.y, volume.z, volume.index)
                if show_coordinates:
                    ax.text(volume.x, volume.y, volume.z + off, "i: " + str(volume.i))
                    ax.text(volume.x, volume.y, volume.z + 2 * off / 3 , "j: " + str(volume.j))
                    ax.text(volume.x, volume.y, volume.z + off / 3, "k: " + str(volume.k))
                if show_transmissibilities:
                    ax.text(volume.x, volume.y, volume.z - off/2, "T: " + str(np.around(self.volumes_trans[volume.index], 4)))
        
        if show_faces:
            for face in self.faces:
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
                    ax.plot([self.faces[face_idx].x, self.volumes[vol_idx].x], [self.faces[face_idx].y, self.volumes[vol_idx].y], [self.faces[face_idx].z, self.volumes[vol_idx].z], c="b")
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Malha {}D".format(self.dimension))
        ax.grid()
        plt.show()

        if show_A:
            print("Matriz de transmissibilidade:\n\n", np.around(self.assemble_tpfa_matrix().todense(),4), "\n")
            
    def _assemble_volumes(self):
        """
        Monta os nós do tipo volume
        """
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    index = i + j * self.nx + k * self.nx * self.ny
                    x = (i + 1/2) * self.dx
                    y = (j + 1/2) * self.dy
                    z = (k + 1/2) * self.dz
                    self.volumes[index] = Node("volume", self._is_internal_node(i, j, k, "volume"), index, i, j, k, x, y, z)

    def _assemble_faces(self):
        """
        Monta os nós do tipo face
        """
        self._assemble_hfaces()
        if self.dimension >= 2:
            self._assemble_lfaces()
        if self.dimension == 3:
           self._assemble_wfaces()
        
    def _assemble_hfaces(self):
        """
        Monta os nós do tipo face de altura
        """
        self.nhfaces = (self.nx + 1) * self.ny * self.nz
        for i in range(self.nx + 1):
            for j in range(self.ny):
                for k in range(self.nz):
                    index = i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny)
                    x = i * self.dx
                    y = (j + 1/2) * self.dy
                    z = (k + 1/2) * self.dz
                    self.faces[index] = Node("hface", self._is_internal_node(i, j, k, "hface"), index, i, j, k, x, y, z)
        
    def _assemble_lfaces(self):
        """
        Monta os nós do tipo face de comprimento
        """
        self.nlfaces = self.nx * (self.ny + 1) * self.nz
        for i in range(self.nx):
            for j in range(self.ny + 1):
                for k in range(self.nz):
                    index = i + j * self.nx + k * self.nx * (self.ny + 1) + self.nhfaces
                    x = (i + 1/2) * self.dx
                    y = j * self.dy
                    z = (k + 1/2) * self.dz
                    self.faces[index] = Node("lface", self._is_internal_node(i, j, k, "lface"), index, i, j, k, x, y, z)
    
    def _assemble_wfaces(self):
        """
        Monta os nós do tipo face de largura
        """
        self.nwfaces = self.nx * self.ny * (self.nz + 1)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz + 1):
                    index = i + j * self.nx + k * self.nx * self.ny + self.nhfaces + self.nlfaces
                    x = (i + 1/2) * self.dx
                    y = (j + 1/2) * self.dy
                    z = k * self.dz
                    self.faces[index] = Node("wface", self._is_internal_node(i, j, k, "wface"), index, i, j, k, x, y, z)
    
    def _assemble_adjacents(self):
        """
        Monta a lista de adjacência de cada nó
        """
        for face in self.faces:
            if face.type_node == "hface":
                v1x, v1y, v1z = face.x + self.dx/2, face.y, face.z
                v2x, v2y, v2z = face.x - self.dx/2, face.y, face.z
                v1 = self._get_volume(v1x, v1y, v1z)
                v2 = self._get_volume(v2x, v2y, v2z)
                self.faces_adjacents[face.index] = np.array([v1 if v1 != None else v2, v2 if v2 != None else v1])
                if v1:
                    self.volumes_adjacents[v1] = np.append(self.volumes_adjacents[v1],face.index)
                if v2:
                    self.volumes_adjacents[v2] = np.append(self.volumes_adjacents[v2],face.index)
            elif face.type_node == "lface":
                v1x, v1y, v1z = face.x, face.y + self.dy/2, face.z
                v2x, v2y, v2z = face.x, face.y - self.dy/2, face.z
                v1 = self._get_volume(v1x, v1y, v1z)
                v2 = self._get_volume(v2x, v2y, v2z)
                self.faces_adjacents[face.index] = np.array([v1 if v1 != None else v2, v2 if v2 != None else v1])
                if v1:
                    self.volumes_adjacents[v1] = np.append(self.volumes_adjacents[v1],face.index)
                if v2:
                    self.volumes_adjacents[v2] = np.append(self.volumes_adjacents[v2],face.index)
            elif face.type_node == "wface":
                v1x, v1y, v1z = face.x, face.y, face.z + self.dz/2
                v2x, v2y, v2z = face.x, face.y, face.z - self.dz/2
                v1 = self._get_volume(v1x, v1y, v1z)
                v2 = self._get_volume(v2x, v2y, v2z)
                self.faces_adjacents[face.index] = np.array([v1 if v1 != None else v2, v2 if v2 != None else v1])
                if v1:
                    self.volumes_adjacents[v1] = np.append(self.volumes_adjacents[v1],face.index)
                if v2:
                    self.volumes_adjacents[v2] = np.append(self.volumes_adjacents[v2],face.index)
            
    def _get_volume(self, x, y, z):
        """
        Retorna o volume que contém o ponto (x, y, z)
        """
        i = int(x / self.dx)
        j = int(y / self.dy)
        k = int(z / self.dz)
        index = i + j * self.nx + k * self.nx * self.ny
        if(i < 0 or i >= self.nx or j < 0 or j >= self.ny or k < 0 or k >= self.nz or index >= len(self.volumes)):
            return None
        return index
    
    def _is_internal_node(self, i, j, k, node_type):
        """
        Retorna se o nó é interno ou não
        """
        if(node_type == "volume"):
            if(self.dimension == 1):
                return i != 0 and i != self.nx - 1
            elif(self.dimension == 2):
                return (i != 0 and i != self.nx - 1) and (j != 0 and j != self.ny - 1)
            elif(self.dimension == 3):
                return (i != 0 and i != self.nx - 1) and (j != 0 and j != self.ny - 1) and (k != 0 and k != self.nz - 1)
        if(node_type == "hface"):
            return i != 0 and i != self.nx
        if(node_type == "lface"):
            return j != 0 and j != self.ny
        if(node_type == "wface"):
            return k != 0 and k != self.nz

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


def main():
    np.set_printoptions(suppress=True)
    (nx, dx) = (1000, 1)
    (ny, dy) = (1000, 1)
    (nz, dz) = (1000, 1)

    mesh1d = Mesh()
    mesh1d.assemble_mesh([(nx, dx)])
    
    mesh2d = Mesh()
    mesh2d.assemble_mesh([(nx, dx), (ny, dy)])
    
    mesh3d = Mesh()
    mesh3d.assemble_mesh([(nx, dx), (ny, dy), (nz, dz)])

    K1d = np.array([[[1. for i in range (mesh1d.nx)]]])
    K1d[0,0,2:] = 1000

    K2d = np.array([[[1. for i in range(mesh2d.nx)] for j in range(mesh2d.ny)]])
    np.fill_diagonal(K2d[0], 1/1000)

    K3d = np.array([[[1for i in range(mesh3d.nx)] for j in range(mesh3d.ny)] for k in range(mesh3d.nz)])
    
    mesh1d.assemble_faces_transmissibilities(K1d)
    A1d = mesh1d.assemble_tpfa_matrix()

    mesh2d.assemble_faces_transmissibilities(K2d)
    A2d = mesh2d.assemble_tpfa_matrix()

    mesh3d.assemble_faces_transmissibilities(K3d)
    A3d = mesh3d.assemble_tpfa_matrix()

    options = (show_coordinates, show_volumes, 
               show_faces, show_adjacents, 
               show_transmissibilities, show_A) = (False, True, True, True, False, False)
    
    #mesh1d.plot(options)
    #mesh2d.plot(options)
    #mesh3d.plot(options)
    
    

if __name__ == "__main__":
    main()