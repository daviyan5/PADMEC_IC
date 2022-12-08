import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

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
        self.volumes_adjacents = np.empty((self.nvols, 0), dtype = Node)

        self.nfaces = self._get_num_faces()
        self.faces = np.empty((self.nfaces), dtype=Node)
        self.faces_adjacents = np.empty((self.nfaces, 0))

        self._assemble_volumes()
        self._assemble_faces()
        self._assemble_adjacents()

    

    def assemble_face_transmissibilities(self, K):
        """
        Monta as transmissibilidades das faces da malha
        """
        #TODO
    
    def plot(self, show_coordinates = False, show_volumes = False, show_faces = False, show_adjacents = False):
        """
        Plota a malha (1D, 2D ou 3D)
        """
        if self.dimension == 1 or self.dimension == 2:
            self._plot_2d(show_coordinates, show_volumes, show_faces, show_adjacents)
        elif self.dimension == 3:
            self._plot_3d(show_coordinates, show_volumes, show_faces, show_adjacents)
        else:
            raise Exception("Número de eixos inválido")
    
    def _plot_2d(self, show_coordinates, show_volumes, show_faces, show_adjacents):
        """
        Plota a malha 2D com o índice de cada nó
        """
        off = 0.1 * (self.dy) * (0.06 if self.dimension == 1 else 2)
        fig, ax = plt.subplots()
        if show_volumes:
            for volume in self.volumes:
                ax.scatter(volume.x, volume.y, c="r" if volume.internal else "g")
                ax.annotate(volume.index, (volume.x, volume.y))
                if show_coordinates:
                    ax.annotate("i: " + str(volume.i), (volume.x, volume.y + off))
                    ax.annotate("j: " + str(volume.j), (volume.x, volume.y + off/2))

        if show_faces:
            for face in self.faces:
                ax.scatter(face.x, face.y, c="b" if face.internal else "y")
                ax.annotate(face.index, (face.x, face.y))
                if show_coordinates:
                    ax.annotate("i: " + str(face.i), (face.x, face.y + off))
                    ax.annotate("j: " + str(face.j), (face.x, face.y + off/2))

        if show_adjacents:
            for face in self.faces_adjacents:
                for adj in face:
                    ax.plot([face.x, self.volumes[adj].x], [face.y, self.volumes[adj].y], c="b")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Malha {}D".format(self.dimension))
        ax.grid()
        plt.show()

    def _plot_3d(self, show_coordinates, show_volumes, show_faces, show_adjacents):
        """
        Plota a malha 3D com o índice de cada nó
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if show_volumes:
            for volume in self.volumes:
                ax.scatter(volume.x, volume.y, volume.z, c="r" if volume.internal else "g")
                ax.text(volume.x, volume.y, volume.z, volume.index)
                if show_coordinates:
                    ax.text(volume.x, volume.y, volume.z + 0.1, "i: " + str(volume.i))
                    ax.text(volume.x, volume.y, volume.z + 0.2, "j: " + str(volume.j))
                    ax.text(volume.x, volume.y, volume.z + 0.3, "k: " + str(volume.k))
        if show_faces:
            for face in self.faces:
                ax.scatter(face.x, face.y, face.z, c="b" if face.internal else "y")
                ax.text(face.x, face.y, face.z, face.index)
                if show_coordinates:
                    ax.text(face.x, face.y, face.z + 0.1, "i: " + str(face.i))
                    ax.text(face.x, face.y, face.z + 0.2, "j: " + str(face.j))
                    ax.text(face.x, face.y, face.z + 0.3, "k: " + str(face.k))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Malha {}D".format(self.dimension))
        ax.grid()
        plt.show()
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
        print("Assemblagem das faces concluída")
        
    def _assemble_hfaces(self):
        """
        Monta os nós do tipo face de altura
        """
        self.nhfaces = (self.nx + 1) * self.ny * self.nz
        for i in range(self.nx + 1):
            for j in range(self.ny):
                for k in range(self.nz):
                    index = i + j * (self.nx + 1) + k * (self.nx + 1) * (self.ny)
                    print("Face de altura: " + str(index))
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
                    print("Face de comprimento: " + str(index))
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
        Monta os nós adjacentes de cada nó
        """
        #TODO

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
    (nx, dx) = (2, 0.1)
    (ny, dy) = (2, 0.1)
    (nz, dz) = (2, 0.1)
    mesh1d = Mesh()
    mesh1d.assemble_mesh([(nx, dx)])
    mesh1d.plot(False, True, True)
    mesh2d = Mesh()
    mesh2d.assemble_mesh([(nx, dx), (ny, dy)])
    mesh2d.plot(False, True, False)
    mesh3d = Mesh()
    mesh3d.assemble_mesh([(nx, dx), (ny, dy), (nz, dz)])
    mesh3d.plot(False, False, True)

if __name__ == "__main__":
    main()