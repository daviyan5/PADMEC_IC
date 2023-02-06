import numpy as np
import time


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

        self.nx, self.ny, self.nz = 1, 1, 1
        self.dx, self.dy, self.dz = 0, 0, 0

        self.nhfaces, self.nlfaces, self.nwfaces = 0, 0, 0
        self.Sh, self.Sl, self.Sw = 0, 0, 0
        self.nvols, self.nfaces = 0, 0

        self.volumes = None
        self.faces = None

    def assemble_mesh(self, axis_attibutes, print_times = True, name = "Mesh " + str(np.random.randint(0, 1000))):
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

        volumes_time = time.time()
        self.volumes = Volumes(self)
        if print_times: 
            print("Time to assemble volumes in mesh {}: \t\t".format(self.name), round(time.time() - volumes_time, 5), "s")
        
        faces_time = time.time()
        self.faces = Faces(self)
        if print_times: 
            print("Time to assemble faces in mesh {}: \t\t".format(self.name), round(time.time() - faces_time, 5), "s")

        if print_times: 
            print("Time to assemble mesh {}: \t\t\t".format(self.name), round(time.time() - start_time, 5), "s")

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
    
class Volumes:
    def __init__(self, mesh):
        self.nx, self.ny, self.nz = mesh.nx, mesh.ny, mesh.nz
        self.dx, self.dy, self.dz = mesh.dx, mesh.dy, mesh.dz
        self.nvols = mesh.nx * mesh.ny * mesh.nz

        index = np.arange(self.nvols)
        self.i, self.j, self.k = index % self.nx, (index // self.nx) % self.ny, (index // self.nx // self.ny)
        self.x, self.y, self.z = (self.i + 1/2) * self.dx, (self.j + 1/2) * self.dy, (self.k + 1/2) * self.dz
        self.internal = mesh._is_internal_node(self.i, self.j, self.k, "volume")
        self.boundary = np.logical_not(self.internal)

        self.adjacentes = None
    
    def _get_index_from_coords(self, coords):
        """
        Retorna o índice do volume a partir das coordenadas
        """
        i = coords[0] // self.dx
        j = coords[1] // self.dy 
        k = coords[2] // self.dz 
        return  (i + j * self.nx + k * self.nx * self.ny).astype(int)
    
    def _get_coords_from_index(self, index):
        """
        Retorna as coordenadas dos volumes a partir do índice
        """
        i = index % self.nx
        j = (index // self.nx) % self.ny
        k = (index // self.nx // self.ny)
        return np.array([(i + 1/2) * self.dx, (j + 1/2) * self.dy, (k + 1/2) * self.dz])

    def _is_valid_coords(self, coords):
        """
        Verifica se as coordenadas estão dentro do domínio dos volumes
        """
        x, y, z = coords
        return (np.all(x >= self.dx/2 & x <= (self.nx - 1/2) * self.dx) and
                np.all(y >= self.dy/2 & y <= (self.ny - 1/2) * self.dy) and
                np.all(z >= self.dz/2 & z <= (self.nz - 1/2) * self.dz))

class Faces:
    def __init__(self, mesh):

        
        self.nx, self.ny, self.nz = mesh.nx, mesh.ny, mesh.nz
        self.dx, self.dy, self.dz = mesh.dx, mesh.dy, mesh.dz
        self.nhfaces = (self.nx + 1) * self.ny * self.nz
        self.nlfaces = self.nx * (self.ny + 1) * self.nz
        self.nwfaces = self.nx * self.ny * (self.nz + 1)
        self.nfaces = self.nhfaces + self.nlfaces + self.nwfaces

        index = np.arange(self.nhfaces)
        self.i, self.j, self.k = index % (self.nx + 1), (index // (self.nx + 1)) % self.ny, (index // (self.nx + 1)) // self.ny
        self.x, self.y, self.z = self.i * self.dx, (self.j + 1/2) * self.dy, (self.k + 1/2) * self.dz
        self.internal = mesh._is_internal_node(self.i, self.j, self.k, "hface")
        
        if mesh.dimension > 1:
            index = np.arange(self.nlfaces)
            self.i = np.append(self.i, index % self.nx)
            self.j = np.append(self.j, (index // self.nx) % (self.ny + 1))
            self.k = np.append(self.k, (index // self.nx // (self.ny + 1)))
            self.x = np.append(self.x, (self.i[-self.nlfaces:] + 1/2) * self.dx)
            self.y = np.append(self.y, self.j[-self.nlfaces:] * self.dy)
            self.z = np.append(self.z, (self.k[-self.nlfaces:] + 1/2) * self.dz)
            self.internal = np.append(self.internal, mesh._is_internal_node(self.i, self.j, self.k, "lface"))
            if mesh.dimension > 2:
                index = np.arange(self.nwfaces)
                self.i = np.append(self.i, index % self.nx)
                self.j = np.append(self.j, (index // self.nx) % self.ny)
                self.k = np.append(self.k, (index // self.nx // self.ny))
                self.x = np.append(self.x, (self.i[-self.nwfaces:] + 1/2) * self.dx)
                self.y = np.append(self.y, (self.j[-self.nwfaces:] + 1/2) * self.dy)
                self.z = np.append(self.z, self.k[-self.nwfaces:] * self.dz)
                self.internal = np.append(self.internal, mesh._is_internal_node(self.i, self.j, self.k, "wface"))

        self.boundary = np.logical_not(self.internal)

        self.adjacents = None
        self._assemble_adjacents(mesh)
    
    def _get_coords_from_index(self, index):
        """
        Retorna as coordenadas das faces a partir do índice
        """
        return np.array([self.x[index], self.y[index], self.z[index]])

    def _get_index_from_coords(self, coords, face_type):
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
    
    def _assemble_adjacents(self, mesh):
        """
        Monta a matriz de adjacência de forma rápida
        """
        self.adjacents = np.zeros((self.nfaces, 2), dtype=int)
        hfaces = np.arange(self.nhfaces)
        hfaces_coords = self._get_coords_from_index(hfaces)
        hfaces_coords = hfaces_coords.T

        self.adjacents[:self.nhfaces] = self._get_adjacents_to_face(mesh, hfaces_coords, "hface")

        if mesh.dimension >= 2:
            lfaces = np.arange(self.nhfaces, self.nhfaces + self.nlfaces)
            lfaces_coords = self._get_coords_from_index(lfaces)
            lfaces_coords = lfaces_coords.T
            assert lfaces_coords.shape == (self.nlfaces, 3)

            self.adjacents[self.nhfaces:self.nhfaces + self.nlfaces] = self._get_adjacents_to_face(mesh, lfaces_coords, "lface")

            if mesh.dimension == 3:
                wfaces = np.arange(self.nhfaces + self.nlfaces, self.nhfaces + self.nlfaces + self.nwfaces)
                wfaces_coords = self._get_coords_from_index(wfaces)
                wfaces_coords = wfaces_coords.T
                assert wfaces_coords.shape == (self.nwfaces, 3)
                
                self.adjacents[self.nhfaces + self.nlfaces:] = self._get_adjacents_to_face(mesh, wfaces_coords, "wface")

    def _get_adjacents_to_face(self, mesh, face_coords, face_type):
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

        v1 = mesh.volumes._get_index_from_coords((v1x, v1y, v1z))
        v2 = mesh.volumes._get_index_from_coords((v2x, v2y, v2z))
        return np.array([v1, v2]).T
    
