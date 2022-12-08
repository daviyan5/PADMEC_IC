import numpy as np
import matplotlib.pyplot as plt

class Node:
   def __init__(self, type_node, internal, index, i = 0, j = 0, k = 0, x = 0, y = 0, z = 0):
    """
    Inicializa um nó da malha

    Atributos:
    ----------
    x, y, z: float
        Coordenadas do nó
    type_node: string
        Tipo do nó (volume ou face)
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
    self.adjacents = []
    self.coeff = []
    
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
        if len(axis_attibutes) == 1:
            self.dimension = 1
            self.assemble_1d_mesh(axis_attibutes[0])
        elif len(axis_attibutes) == 2:
            self.dimension = 2
            self.assemble_2d_mesh(axis_attibutes[0], axis_attibutes[1])
        elif len(axis_attibutes) == 3:
            self.dimension = 3
            self.assemble_3d_mesh(axis_attibutes[0], axis_attibutes[1], axis_attibutes[2])
        else:
            raise Exception("Número de eixos inválido")
    
    def assemble_1d_mesh(self, axis_attibutes):
        """
        Monta a malha 1D de acordo com os atributos passados

        Parâmetros:
        -----------
        axis_attibutes : tuple
            Tupla com os atributos do eixo
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: (nx, dx)

        """
        nx, dx = axis_attibutes
        self.volumes = np.array([None] * nx, dtype=object)
        # Montando os volumes
        for i in range(nx):
            volume = Node("volume", not (i == 0 or i == nx - 1), i, i, x = dx * (i + 1/2))
            self.volumes[volume.index] = (volume)
        
        self.faces = np.array([None] * (nx + 1), dtype=object)
        # Montando as faces
        for i in range(nx + 1):
            face = Node("face", not (i == 0 or i == nx), i, i, x = dx * i)

            # Conectando aos volumes
            if i != 0:
                face.adjacents.append((self.volumes[i - 1].index, dx))
                self.volumes[i - 1].adjacents.append((face.index, dx))
            if i != nx:
                face.adjacents.append((self.volumes[i].index, dx))
                self.volumes[i].adjacents.append((face.index, dx))
            
            self.faces[face.index] = face
            
    
    def assemble_2d_mesh(self, axis_attibutes_x, axis_attibutes_y):
        """
        Monta a malha 2D de acordo com os atributos passados

        Parâmetros:
        -----------
        axis_attibutes_x : tuple
            Tupla com os atributos do eixo x
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: (nx, dx)
        axis_attibutes_y : tuple
            Tupla com os atributos do eixo y
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: (ny, dy)

        """
        nx, dx = axis_attibutes_x
        ny, dy = axis_attibutes_y
        self.volumes = np.array([None] * (nx * ny), dtype=object)
        # Montando os volumes
        for i in range(nx):
            for j in range(ny):
                volume = Node("volume", not (i == 0 or i == nx - 1 or j == 0 or j == ny - 1), 
                              i + j * nx, i, j, x = dx * (i + 1/2), y = dy * (j + 1/2))
                self.volumes[volume.index] = volume
        
        # Montando as faces de cima pra baixo da esquerda pra direita
        # Número de faces = nx * (ny + 1) + (nx + 1) * ny
        self.faces = np.array([None] * (nx * (ny + 1) + (nx + 1) * ny), dtype=object)
        for i in range(nx):
            for j in range(ny + 1):
                face = Node("face", not (i == 0 or i == nx - 1 or j == 0 or j == ny), 
                            i + j * nx, i, j, x = dx * (i + 1/2), y = dy * j)
                # Conectando aos volumes
                if j > 0:
                    face.adjacents.append((self.volumes[i + (j - 1) * nx].index, dy))
                    self.volumes[i + (j - 1) * nx].adjacents.append((face.index, dy))
                if j < ny:
                    face.adjacents.append((self.volumes[i + j * nx].index, dy))
                    self.volumes[i + j * nx].adjacents.append((face.index, dy))
                
                
                self.faces[face.index] = face
        
        for i in range(nx + 1):
            for j in range(ny):
                face = Node("face", not (i == 0 or i == nx or j == 0 or j == ny - 1), 
                            nx * (ny + 1) + i + j * (nx + 1), i, j, x = dx * i, y = dy * (j + 1/2))

                # Conectando aos volumes
                if i > 0:
                    face.adjacents.append((self.volumes[i - 1 + j * nx].index, dx))
                    self.volumes[i - 1 + j * nx].adjacents.append((face.index, dx))
                if i < nx:
                    face.adjacents.append((self.volumes[i + j * nx].index, dx))
                    self.volumes[i + j * nx].adjacents.append((face.index, dx))
                self.faces[face.index] = face

    def assemble_3d_mesh(self, axis_attibutes_x, axis_attibutes_y, axis_attibutes_z):
        """
        Monta a malha 3D de acordo com os atributos passados

        Parâmetros:
        -----------
        axis_attibutes_x : tuple
            Tupla com os atributos do eixo x
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: (nx, dx)
        axis_attibutes_y : tuple
            Tupla com os atributos do eixo y
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: (ny, dy)
        axis_attibutes_z : tuple
            Tupla com os atributos do eixo z
            Cada tupla contém o número de nós de volumes no eixo (n) e o espaçamento no eixo (d)
            Exemplo: (nz, dz)

        """
        nx, dx = axis_attibutes_x
        ny, dy = axis_attibutes_y
        nz, dz = axis_attibutes_z
        self.volumes = np.array([None] * (nx * ny * nz), dtype=object)
        # Montando os volumes
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    volume = Node("volume", not (i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1), 
                                  i + j * nx + k * nx * ny, i, j, k, x = dx * (i + 1/2), y = dy * (j + 1/2), z = dz * (k + 1/2))
                    self.volumes[volume.index] = volume
        
        # Montando as faces de cima pra baixo da esquerda pra direita
        # Número de faces = nx * (ny + 1) + (nx + 1) * ny
        self.faces = np.array([None] * (nx * (ny + 1) + (nx + 1) * ny), dtype=object)
        #TODO

    def assemble_face_transmissibilities(self, K):
        """
        Monta as transmissibilidades das faces da malha
        """
        #TODO
    
    def plot(self, show_coordinates = False):
        """
        Plota a malha (1D ou 2D)
        """
        off = 0.12
        fig, ax = plt.subplots()
        for volume in self.volumes:
            ax.scatter(volume.x, volume.y, c="r")
            ax.annotate(volume.index, (volume.x, volume.y))
            if show_coordinates:
                ax.annotate("i: " + str(volume.i), (volume.x + off, volume.y + off))
                ax.annotate("j: " + str(volume.j), (volume.x + off, volume.y))

        
        for face in self.faces:
            ax.scatter(face.x, face.y, c="b")
            ax.annotate(face.index, (face.x, face.y))
            if show_coordinates:
                ax.annotate("i: " + str(face.i), (face.x + off, face.y + off))
                ax.annotate("j: " + str(face.j), (face.x + off, face.y))

        # Plot adjacenties
        for volume in self.volumes:
            for adjacent in volume.adjacents:
                ax.plot([volume.x, self.faces[adjacent[0]].x], [volume.y, self.faces[adjacent[0]].y], c="k")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Malha {}D".format(self.dimension))
        ax.grid()
        plt.show()



def main():
    (nx, dx) = (5, 0.1)
    (ny, dy) = (3, 0.1)
    (nz, dz) = (2, 2)
    mesh1d = Mesh()
    mesh1d.assemble_mesh([(nx, dx)])
    mesh1d.plot()
    mesh2d = Mesh()
    mesh2d.assemble_mesh([(nx, dx), (ny, dy)])
    mesh2d.plot()

if __name__ == "__main__":
    main()