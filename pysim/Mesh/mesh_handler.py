import numpy as np
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
        
        self.name = None

        self.axis_attibutes = None
        self.dimension = None
        self.faces_adjacents = None

        self.nx, self.ny, self.nz = 1, 1, 1
        self.dx, self.dy, self.dz = 0, 0, 0

        self.nhfaces, self.nlfaces, self.nwfaces = 0, 0, 0
        self.Sh, self.Sl, self.Sw = 0, 0, 0
        self.nvols, self.nfaces = 0, 0

        self.internal_volumes = None
        self.internal_faces = None
        self.boundary_volumes = None
        self.boundary_faces = None

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
        self.nfaces = self._get_num_faces()

        self.Sh = self.dy * self.dz
        self.Sl = self.dx * self.dz
        self.Sw = self.dx * self.dy
        self.nhfaces = (self.nx + 1) * self.ny * self.nz
        self.nlfaces = self.nx * (self.ny + 1) * self.nz
        self.nwfaces = self.nx * self.ny * (self.nz + 1)

        self.faces_adjacents = np.empty((self.nfaces, 2), dtype = int)
        
    
        adj_time = time.time()
        self._assemble_adjacents()
        print("Time to assemble adjs in mesh {}: \t\t".format(self.name), round(time.time() - adj_time, 5), "s")

        assert self.faces_adjacents.shape == (self.nfaces, 2)

        print("Time to assemble mesh {}: \t\t\t".format(self.name), round(time.time() - start_time, 5), "s")
    
    def _assemble_adjacents(self):
        """
        Monta a matriz de adjacência de forma rápida
        """
        index = np.arange(self.nvols)
        i, j, k = index % self.nx, (index // self.nx) % self.ny, index // (self.nx * self.ny)
        self.internal_volumes = self._is_internal_node(i, j, k, "volume")
        self.boundary_volumes = np.logical_not(self.internal_volumes)

        
        index = np.arange(self.nhfaces)
        i, j, k = index % (self.nx + 1), (index // (self.nx + 1)) % self.ny, index // ((self.nx + 1) * self.ny)
        self.internal_faces = self._is_internal_node(i, j, k, "hface")
        if self.dimension >= 2:
            index = np.arange(self.nlfaces)
            i, j, k = index % self.nx, (index // self.nx) % (self.ny + 1), index // (self.nx * (self.ny + 1))
            np.append(self.internal_faces, self._is_internal_node(i, j, k, "lface"))
        if self.dimension >= 3:
            index = np.arange(self.nwfaces)
            i, j, k = index % self.nx, (index // self.nx) % self.ny, index // (self.nx * self.ny)
            np.append(self.internal_faces, self._is_internal_node(i, j, k, "wface"))
        
        self.boundary_faces = np.logical_not(self.internal_faces)



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
        Retorna os índices dos volumes adjacentes a uma face
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
        return (np.all(x >= self.dx/2 & x <= (self.nx - 1/2) * self.dx) and
                np.all(y >= self.dy/2 & y <= (self.ny - 1/2) * self.dy) and
                np.all(z >= self.dz/2 & z <= (self.nz - 1/2) * self.dz))
    
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

    def _get_faces_index_from_coords(self, coords, face_type):
        """
        Retorna o índice da face a partir das coordenadas
        """
        if face_type == "hface":
            i = coords[0] // self.dx
            j = coords[1] // self.dy
            k = coords[2] // self.dz
            return (i + j * (self.nx + 1) + k * (self.nx + 1) * self.ny).astype(int)
        
        elif face_type == "lface":
            i = coords[0] // self.dx
            j = coords[1] // self.dy
            k = coords[2] // self.dz 
            return (i + j * self.nx + k * self.nx * (self.ny + 1) + self.nhfaces).astype(int)
        
        elif face_type == "wface":
            i = coords[0] // self.dx 
            j = coords[1] // self.dy 
            k = coords[2] // self.dz
            return (i + j * self.nx + k * self.nx * self.ny + self.nhfaces + self.nlfaces).astype(int)

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
    
    def _volumes(self):
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

    def _faces(self):
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
        
    def _hfaces(self):
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
        
    def _lfaces(self):
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
    
    def _wfaces(self):
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