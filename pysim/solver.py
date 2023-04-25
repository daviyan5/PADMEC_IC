import numpy as np
import scipy as sp
import os
import sys
import time

import helpers


counter = 0
# Subindo o caminho para o diretório pai, para importar o módulo
sys.path.append('../')

from preprocessor.meshHandle.finescaleMesh import FineScaleMesh                     # type: ignore 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class TPFAsolver:
    def __init__(self, verbose = True, check : bool = True, name: str = "MESH") -> None:

        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(verbose, bool) or not isinstance(check, bool):
            raise TypeError("\t verbose and check must be boolean")
        if not isinstance(name, str):
            raise TypeError("\t name must be a string")
        # ===========================================================================================================#
        
        self.verbose        = verbose                                               # Flag para verbose mode    (bool)
        self.check          = check                                                 # Flag para check mode      (bool)
        np.set_printoptions(precision = 4)
        global counter
        if name == "MESH":
            name += str(counter)
        
        counter += 1
        self.name           = name                                                  # Nome da malha            (str)

        # Utilidades-----------------------------------------------------------------------------------------------#
        self.times                  = {}                                            # Dicionário para os tempos (dict)
        self.mesh                   = None                                          # Malha                     (FineScaleMesh)                           
        self.dim                    = None                                          # Dimensão da malha         (int)

        self.faces_areas            = None                                          # Área das faces            (np.ndarray)
        self.volumes                = None                                          # Volume dos volumes        (np.ndarray)

        self.nvols                  = None                                          # Número de volumes         (int)
        self.nfaces                 = None                                          # Número de faces           (int)

        self.nvols_pairs            = None                                          # Número de pares de    
                                                                                    # volumes                   (int)

        self.vols_pairs             = None                                          # Pares de volumes      
                                                                                    # array de dimensões 
                                                                                    # {nfaces, 2}               (np.ndarray)
        self.permeability         = None                                          # Permeabilidades           (np.ndarray)
        # IMPRESS - Handles carregados do arquivo de malha -------------------------------------------------------#
        self.internal_faces         = None                                          # Índice das faces  
                                                                                    # internas                  (np.ndarray)

        self.boundary_faces         = None                                          # Índice das faces
                                                                                    # de contorno               (np.ndarray)

        self.faces_center           = None                                          # Centro das faces          (np.ndarray)
        self.volumes_center         = None                                          # Centro dos volumes        (np.ndarray)        
        self.faces_connectivities   = None                                          # Conectividade das faces   (np.ndarray)  
        self.nodes_coords           = None                                          # Coordenadas dos nós       (np.ndarray)

        # Utilidades para o TPFA ----------------------------------------------------------------------------------#
        self.N                      = None                                          # Array {nfaces, 3}     
                                                                                    # com os vetores unitários
                                                                                    # normais às faces 
                                                                                    # internas por
                                                                                    # par de volumes.           (np.ndarray)

        self.hL, self.hR            = None, None                                    # Arrays {nfaces}     
                                                                                    # com as distâncias dos
                                                                                    # centros dos volumes
                                                                                    # até a face interna.       (np.ndarray)

        self.faces_trans            = None                                          # Array {nfaces, 3}
                                                                                    # com a transmissibilidade
                                                                                    # das faces internas.       (np.ndarray)

        self.At_TPFA                = None                                          # Matriz esparsa
                                                                                    # de transmissibilidade
                                                                                    # das faces internas.       (csr_matrix)

        self.bt_TPFA                = None                                          # Lado direito do sistema   (np.ndarray)
        self.p_TPFA                 = None                                          # Pressão nos volumes       (np.ndarray)

        self.irel                   = None                                          # Erro relativo por volume  (np.ndarray)
        self.numerical_p            = None                                          # Solução numérica          (np.ndarray)
        self.analytical_p           = None                                          # Solução analítica         (np.ndarray)

        # Condicoes de contorno -----------------------------------------------------------------------------------#
        self.faces_with_bc          = None                                          # Índice das faces com
                                                                                    # condições de contorno     (np.ndarray)

        self.Ad_TPFA                = None                                          # Contribuição da CC 
                                                                                    # de Dirichlet na matriz    (csr_matrix)

        self.bd_TPFA                = None                                          # Contribuição da CC
                                                                                    # de Dirichlet no lado
                                                                                    # direito do sistema        (np.ndarray)

        self.d_faces                = None                                          # Índice das faces com  
                                                                                    # condições de contorno
                                                                                    # de Dirichlet              (np.ndarray)

        self.d_values               = None                                          # Valores da CC de
                                                                                    # Dirichlet                 (np.ndarray)

        self.d_volumes              = None                                          # Volumes com condições
                                                                                    # de contorno de Dirichlet  (np.ndarray)

        self.bn_TPFA                = None                                          # Contribuição da CC
                                                                                    # de Neumann no lado
                                                                                    # direito do sistema        (np.ndarray)

        self.n_faces                = None                                          # Índice das faces com
                                                                                    # condições de contorno
                                                                                    # de Neumann                (np.ndarray)

        self.n_values               = None                                          # Valores da CC de
                                                                                    # Neumann                   (np.ndarray)

        self.n_volume               = None                                          # Volumes com condições
                                                                                    # de contorno de Neumann    (np.ndarray)
        
         
    def __del__(self) -> None:
        pass

    def reset(self, verbose : bool = True, check : bool = True, name: str = "MESH") -> None:

        self = TPFAsolver(verbose, check, name)

    def set_name(self, name : str) -> None:
        """
        Define o nome da malha
        """
        self.name = name

    def get_args_expected_keys() -> list:
        """
        Retorna a lista de chaves esperadas para o dicionário de argumentos
        """
        return ["meshfile", "mesh", "mesh_params", "dim", "permeability", "source", "neumann", "dirichlet", "analytical", "vtk"]
    
    def get_results_keys() -> list:
        """
        Retorna a lista de chaves referentes aos resultados
        """
        return ["mesh_params", "nvols, nfaces", "mesh", "numerical_p", "analytical_p", "irel", "error", "vtk_filename", "times"]
    
    def get_times_keys() -> list:
        """
        Retorna a lista de chaves referentes aos tempos
        """
        return ["Total Time", "Pre-Processing", "TPFA System Preparation", "TPFA Boundary Conditions", "TPFA System Solving", "Post-Processing"]
    def solveTPFA(self, args: dict) -> dict:
        """
        Resolve o TPFA
        """

        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(args, dict):
            raise TypeError("\t args must be a dictionary")
        # ===========================================================================================================#

        full_step_name = "Total Time"
        self.times[full_step_name] = time.time()

        # ------ VERBOSE ------ #
        if self.verbose:
            helpers.reset_verbose()
            helpers.verbose("== Aplying TPFA scheme {} on mesh {}...".format(counter, self.name), "OUT")
        # ---------------------- #

        # Pré-processamento
        step_name = "Pre-Processing"
        if self.verbose:
            print("{}...".format(step_name), end ="\r")
        self.times[step_name] = time.time()
        self.mesh_params    = self.pre_process(args["meshfile"], 
                                               args["mesh_params"],
                                               args["mesh"], 
                                               args["dim"])
        
        self.vols_pairs     = self.__get_volumes_pairs(self.mesh, self.internal_faces)
        self.nvols_pairs    = self.vols_pairs.shape[0]
        self.permeability   = args["permeability"]

        self.times[step_name] = self.__set_time(self.times[step_name], step_name)

        # ------ VERBOSE ------ #
        if self.verbose:
            self.__verbose(step_name)
        # ---------------------- #
        
        # Prepara o sistema TPFA
        step_name = "TPFA System Preparation"
        if self.verbose:
            print("{}...".format(step_name), end ="\r")
        self.times[step_name] = time.time()

        self.__assemble_faces_transmissibilities()
        self.__assemble_TPFA_matrix(args["source"])
        self.times[step_name] = self.__set_time(self.times[step_name], step_name)

        # ------ VERBOSE ------ #
        if self.verbose:
            self.__verbose(step_name)
        # ---------------------- #

        step_name = "TPFA Boundary Conditions"
        if self.verbose:
            print("{}...".format(step_name), end ="\r")
        self.times[step_name] = time.time()

        self.__set_dirichlet_boundary_conditions(args["dirichlet"])
        self.__set_neumann_boundary_conditions(args["neumann"])
        self.times[step_name] = self.__set_time(self.times[step_name], step_name)

        # ------ VERBOSE ------ #
        if self.verbose:
           self.__verbose(step_name)
        # ---------------------- #
            
        # Resolve o sistema TPFA
        step_name = "TPFA System Solving"
        if self.verbose:
            print("{}...".format(step_name), end ="\r")
        self.times[step_name] = time.time()
        self.__solve_TPFA_system()
        self.times[step_name] = self.__set_time(self.times[step_name], step_name)
        # ------ VERBOSE ------ #
        if self.verbose:
            self.__verbose(step_name)
        
        # ---------------------- #

        # Pós-processamento
        step_name = "Post-Processing"
        if self.verbose:
            print("{}...".format(step_name), end ="\r")
        self.times[step_name] = time.time()
        x, y, z                 = self.volumes_center[:, 0], self.volumes_center[:, 1], self.volumes_center[:, 2]
        self.numerical_p        = self.p_TPFA
        self.analytical_p       = args["analytical"](x, y, z) if args["analytical"] else None
        self.irel               = np.sqrt((((self.analytical_p - self.p_TPFA) ** 2  * self.volumes) / 
                                            ((self.analytical_p ** 2) * self.volumes))) if args["analytical"] else None
        
        if args["vtk"]:
            self.__set_mesh_variables()
            self.vtk_filename   = self.create_vtk()
        else:
            self.vtk_filename   = "null"
        self.times[step_name]   = self.__set_time(self.times[step_name], step_name)
        
        # ------ VERBOSE ------ #
        if self.verbose:
            self.__verbose(step_name)
        # ---------------------- #

        self.times[full_step_name]   = self.__set_time(self.times[full_step_name], full_step_name)
        results = {
            "mesh_params"   :  self.mesh_params,
            "nvols, nfaces" : (self.nvols, self.nfaces),
            "mesh"          :  self.mesh,
            "numerical_p"   :  self.numerical_p,
            "analytical_p"  :  self.analytical_p,
            "irel"          :  self.irel,
            "error"         :  self.get_error(),
            "vtk_filename"  :  self.vtk_filename,
            "times"         :  self.times
        }
        if self.verbose:
            self.__verbose(full_step_name)
        return results

    def pre_process(self, meshfile: str = None, mesh_params : tuple = None, mesh : FineScaleMesh = None, dim : int = 3) -> tuple:
        """
        Realiza o pré-processamento da malha: Calcula área e volume dos elementos e 
        retorna uma tupla (A, V) para reutilização, se necessário
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(meshfile, str) and meshfile is not None:
            raise TypeError("\t meshfile must be a string!")
        if not isinstance(mesh_params, tuple) and mesh_params is not None:
            raise TypeError("\t mesh_params must be a tuple!")
        if not isinstance(mesh, FineScaleMesh) and mesh is not None:
            raise TypeError("\t mesh must be a FineScaleMesh!")
        if not isinstance(dim, int):
            raise TypeError("\t dim must be an integer!")
        if dim not in [2, 3]:
            raise ValueError("\t dim must be 2 or 3!")
        if mesh is None and meshfile is None:
            raise ValueError("\t mesh and meshfile cannot be None at the same time!")
        # ===========================================================================================================#

        self.dim = dim
        if mesh is not None and meshfile is not None:
            helpers.block_print()
            self.mesh = FineScaleMesh(meshfile, self.dim)
            helpers.enable_print()
        elif mesh is not None:
            self.mesh = mesh
        
        if not mesh_params:
            self.faces_areas = TPFAsolver.get_areas(self.mesh, self.mesh.faces.all)
            self.volumes = TPFAsolver.get_volumes(self.mesh, self.mesh.volumes.all, self.faces_areas)
            mesh_params = self.faces_areas, self.volumes
        else:
            self.faces_areas, self.volumes = mesh_params

        self.nvols = len(self.mesh.volumes)
        self.nfaces = len(self.mesh.faces)

        self.internal_faces         = self.mesh.faces.internal[:]
        self.boundary_faces         = self.mesh.faces.boundary[:]
        self.faces_center           = self.mesh.faces.center[:]
        self.volumes_center         = self.mesh.volumes.center[:]
        self.faces_connectivities   = self.mesh.faces.connectivities[:]
        self.nodes_coords           = self.mesh.nodes.coords[:]

        return mesh_params

    def get_error(self) -> float:
        """
        Calcula o erro relativo entre a solução analítica e a solução numérica
        """
        if self.analytical_p is not None and self.numerical_p is not None:
            return np.sqrt(((np.sum((self.analytical_p - self.p_TPFA)** 2 * self.volumes) ) / 
                            (np.sum(self.analytical_p ** 2  * self.volumes))))
        else:
            return None
    
    @staticmethod    
    def get_areas(mesh : FineScaleMesh, faces_index : np.ndarray) -> np.ndarray:
        """
        Calcula a área de cada face
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(mesh, FineScaleMesh):
            raise TypeError("\t mesh must be a FineScaleMesh!")
        if not isinstance(faces_index, np.ndarray):
            raise TypeError("\t faces_index must be a numpy.ndarray!")
        if len(faces_index.shape) != 1:
            raise ValueError("\t faces_index must be a 1D numpy.ndarray!")
        # ===========================================================================================================#

        faces_nodes = mesh.faces.connectivities[faces_index]
        i = mesh.nodes.coords[faces_nodes[:, 0]]
        j = mesh.nodes.coords[faces_nodes[:, 1]]
        k = mesh.nodes.coords[faces_nodes[:, 2]]
        return np.linalg.norm(np.cross(i - j, k - j), axis = 1) 
    
    @staticmethod
    def get_volumes(mesh : FineScaleMesh, vols_index : np.ndarray, faces_areas : np.ndarray) -> np.ndarray:
        """
        Calcula o volume de cada volume, assumindo que o volume é um hexaedro
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(mesh, FineScaleMesh):
            raise TypeError("\t mesh must be a FineScaleMesh!")
        if not isinstance(vols_index, np.ndarray):
            raise TypeError("\t vols_index must be a numpy.ndarray!")
        if len(vols_index.shape) != 1:
            raise ValueError("\t vols_index must be a 1D numpy.ndarray!")
        if not isinstance(faces_areas, np.ndarray):
            raise TypeError("\t faces_areas must be a numpy.ndarray!")
        if len(faces_areas.shape) != 1:
            raise ValueError("\t faces_areas must be a 1D numpy.ndarray!")
        # ===========================================================================================================#

        nvols           = len(vols_index)
        faces_by_volume = mesh.volumes.bridge_adjacencies(mesh.volumes.all, 3, 2)
        nfaces          = len(faces_by_volume[0])
        area_by_face    = faces_areas[faces_by_volume.flatten()].reshape(nvols, nfaces)

        return np.prod(area_by_face, axis = 1) ** (1 / 4)
    
    def create_vtk(self) -> None:

        meshset = self.mesh.core.mb.create_meshset()
        self.mesh.core.mb.add_entities(meshset, self.mesh.core.all_volumes)
        vtk_filename = "mesh_{}.vtk".format(self.name)
        write_filename = os.path.join("vtks", vtk_filename)
        self.mesh.core.mb.write_file(write_filename, [meshset])
        return vtk_filename
    
    # ----------- Private Methods ----------------#
    def __verbose(self, step_name):
        """
        Imprime o passo atual da simulação
        """
        print(" " * 100, end ="\r")
        if step_name == "Pre-Processing":
            helpers.verbose("== Done with pre-processing!", "OUT")
            helpers.verbose("Time for step [{}] : {:.5} s".format(step_name, self.times[step_name]), "TIME")
            helpers.verbose("NVOLS : {} ; NFACES : {} ; NVOLS_PAIRS : {}".format(self.nvols, self.nfaces, self.nvols_pairs), "INFO")

        if step_name == "TPFA System Preparation":
            helpers.verbose("== Done with TPFA system preparation!", "OUT")
            helpers.verbose("Time for step [{}] : {:.5} s".format(step_name, self.times[step_name]), "TIME")
            mx, argmx = np.max(self.faces_trans), np.argmax(self.faces_trans)
            mn, argmn = np.min(self.faces_trans), np.argmin(self.faces_trans)
            helpers.verbose("MAX FACE TRANS : ({:.5}, {})".format(mx, argmx), "INFO")
            helpers.verbose("MIN FACE TRANS : ({:.5}, {})".format(mn, argmn), "INFO")

            mx, argmx = np.max(self.bt_TPFA), np.argmax(self.bt_TPFA)
            mn, argmn = np.min(self.bt_TPFA), np.argmin(self.bt_TPFA)
            helpers.verbose("MAX FLUX : ({:.5}, {})".format(mx, argmx), "INFO")
            helpers.verbose("MIN FLUX : ({:.5}, {})".format(mn, argmn), "INFO")

        if step_name == "TPFA Boundary Conditions":
            helpers.verbose("== Done with TPFA boundary conditions!", "OUT")
            helpers.verbose("Time for step [{}] : {:.5} s".format(step_name, self.times[step_name]), "TIME")
            helpers.verbose("Nº FACES DIRICHLET : {} ; Nº VOLUMES DIRICHLET : {}".format(self.d_faces.shape[0], self.d_volumes.shape[0]), "INFO")
            mx, argmx = np.max(self.d_values), self.d_faces[np.argmax(self.d_values)]
            mn, argmn = np.min(self.d_values), self.d_faces[np.argmin(self.d_values)]
            helpers.verbose("MAX DIRICHLET VALUES : ({:.5}, {})".format(mx, argmx), "INFO")
            helpers.verbose("MIN DIRICHLET VALUES : ({:.5}, {})".format(mn, argmn), "INFO")

           
            try:
                helpers.verbose("Nº FACES NEUMANN : {} ; Nº VOLUMES NEUMANN : {}".format(self.n_faces.shape[0], self.n_volume.shape[0]), "INFO")
                mx, argmx = np.max(self.n_values), self.n_faces[np.argmax(self.n_values)]
                mn, argmn = np.min(self.n_values), self.n_faces[np.argmin(self.n_values)]
                helpers.verbose("MAX NEUMANN VALUES : ({:.5}, {})".format(mx, argmx), "INFO")
                helpers.verbose("MIN NEUMANN VALUES : ({:.5}, {})".format(mn, argmn), "INFO")
            except:
                helpers.verbose("NO NEUMANN BOUNDARY CONDITIONS", "INFO")

        if step_name == "TPFA System Solving":
            helpers.verbose("== Done with TPFA system solving!", "OUT")
            if self.check:
                helpers.verbose("++ TPFA Solution is consistent", "CHECK")
            helpers.verbose("Time for step [{}] : {:.5} s".format(step_name, self.times[step_name]), "TIME")
            self.times[step_name] = self.__set_time(self.times[step_name], step_name)
            mx, argmx = np.max(self.p_TPFA), np.argmax(self.p_TPFA)
            mn, argmn = np.min(self.p_TPFA), np.argmin(self.p_TPFA)
            helpers.verbose("MACHINE ERROR : {:.5}".format(np.finfo(float).eps), "INFO")
            helpers.verbose("MAX NUM. PRESSURE : ({:.5}, {})".format(mx, argmx), "INFO")
            helpers.verbose("MIN NUM. PRESSURE : ({:.5}, {})".format(mn, argmn), "INFO")

        if step_name == "Post-Processing":
            helpers.verbose("== Done with post-processing!", "OUT")
            helpers.verbose("Time for step [{}] : {:.5} s".format(step_name, self.times[step_name]), "TIME")
            helpers.verbose("VTK FILENAME : {}".format(self.vtk_filename), "INFO")
            if self.analytical_p is not None:
                mx, argmx = np.max(self.analytical_p), np.argmax(self.analytical_p)
                mn, argmn = np.min(self.analytical_p), np.argmin(self.analytical_p)
                helpers.verbose("MAX ANALYTICAL PRESSURE : ({:.5}, {})".format(mx, argmx), "INFO")
                helpers.verbose("MIN ANALYTICAL PRESSURE : ({:.5}, {})".format(mn, argmn), "INFO")
                mx, argmx = np.max(self.irel), np.argmax(self.irel)
                mn, argmn = np.min(self.irel), np.argmin(self.irel)
                helpers.verbose("MAX ERROR : ({:.5}, {})".format(mx, argmx), "INFO")
                helpers.verbose("MIN ERROR : ({:.5}, {})".format(mn, argmn), "INFO")
                

        if step_name == "Total Time":
            helpers.verbose("== End of simulation {}.".format(counter), "OUT")
            helpers.verbose("Time for step [{}] : {:.5} s".format(step_name, self.times[step_name]), "TIME")

    def __set_time(self, step_time : float, step_name: str) -> float:
        """
        Atualiza o tempo de execução
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(step_time, float):
            raise TypeError("\t step_time must be a float!")
        if not isinstance(step_name, str):
            raise TypeError("\t step_name must be a string!")
        # ===========================================================================================================#

        return_time = time.time() - step_time
        return return_time
    
    def __set_mesh_variables(self) -> None:
        """
        Atualiza as variáveis da malha
        """
        self.mesh.permeability[:]   = self.permeability.reshape((self.nvols, 9))
        self.mesh.numerical_p[:]    = self.p_TPFA
        if self.analytical_p is not None:
            self.mesh.analytical_p[:]   = self.analytical_p
            self.mesh.error[:]          = self.irel
    
    
    def __get_volumes_pairs(self, mesh : FineScaleMesh, faces_index : np.ndarray) -> np.ndarray:
        """
        Retorna os pares de volumes que compartilham cada face
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(mesh, FineScaleMesh):
            raise TypeError("\t mesh must be a FineScaleMesh!")
        if not isinstance(faces_index, np.ndarray):
            raise TypeError("\t faces_index must be a numpy.ndarray!")
        if len(faces_index.shape) != 1:
            raise ValueError("\t faces_index must be a 1D numpy.ndarray!")
        # ===========================================================================================================#

        vols_pairs = mesh.faces.bridge_adjacencies(faces_index, 2, 3)
        nvols_pairs = vols_pairs.shape[0]

        if vols_pairs.shape[1] == 2:                                                # Se as faces forem internas,
                                                                                    # Assegurar que os volumes da esquerda
                                                                                    # Estão sempre no sentido negativo dos eixos
            faces_centers   = mesh.faces.center[faces_index]
            volumes_centers = mesh.volumes.center[vols_pairs.flatten()]
            volumes_centers = volumes_centers.reshape(nvols_pairs, 2, 3)

            L = volumes_centers[:, 0]

            vL = (L - faces_centers).sum(axis = 1)

            vols_pairs[vL > 0, 0], vols_pairs[vL > 0, 1] = vols_pairs[vL > 0, 1], vols_pairs[vL > 0, 0]

        return vols_pairs
    
    def __get_normal(self, faces_index : np.ndarray, volumes_pairs : np.ndarray = None) -> np.ndarray:
        """
        Retorna o vetor normal de cada face para o centro dos volumes adjacentes 
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not isinstance(faces_index, np.ndarray):
            raise TypeError("\t faces_index must be a numpy.ndarray!")
        if len(faces_index.shape) != 1:
            raise ValueError("\t faces_index must be a 1D numpy.ndarray!")
        # ===========================================================================================================#

        faces_nodes = self.faces_connectivities[faces_index]                        # Nós das faces

        i = self.nodes_coords[faces_nodes[:, 0]]                                    # Coordenadas dos primeiros nós das faces
        j = self.nodes_coords[faces_nodes[:, 1]]                                    # Coordenadas dos segundos nós das faces
        k = self.nodes_coords[faces_nodes[:, 2]]                                    # Coordenadas dos terceiros nós das faces

        if volumes_pairs is None:
            volumes_pairs = self.mesh.faces.bridge_adjacencies(faces_index, 2, 3)   # Volumes que compartilham as faces
        
        nvols_pairs = volumes_pairs.shape[0]                                        # Número de pares de volumes
        
        nvols = volumes_pairs.shape[1] if len(volumes_pairs.shape) > 1 else 1

        volumes_centers = self.volumes_center[volumes_pairs.flatten()]              # Centros dos volumes
        volumes_centers = volumes_centers.reshape((nvols_pairs, nvols, 3))          # Redimensiona para que cada seja um par de volumes
        faces_centers   = self.faces_center[faces_index]                              # Centros das faces

        L = volumes_centers[:, 0]
        vL = faces_centers - L                                                      # Distância entre os centros

        if nvols > 1:                                                               # Se houverem dois volumes por face (internas)
            R = volumes_centers[:, 1]
            vR = faces_centers - R
        else:
            vR = None
        N = np.cross(i - j, k - j)                                                  # Calcula os vetores normais
        return N, vL, vR
    
    def __assemble_faces_transmissibilities(self) -> None:
        """
        Monta os vetores normais das faces internas da malha e as transmissibilidades
        por face.
        """
        
        N, vL, vR = self.__get_normal(self.internal_faces, self.vols_pairs)
        
        self.N = np.abs(N) / np.linalg.norm(N, axis = 1)[:, None]                   # Normaliza os vetores normais
    
        self.h_L = np.abs(np.einsum("ij,ij->i", self.N, vL))                        # Distância normal entre o centro da face e o centro do volume da esquerda
        self.h_R = np.abs(np.einsum("ij,ij->i", self.N, vR))                        # Distância normal entre o centro da face e o centro do volume da direita

        KL = self.permeability[self.vols_pairs[:, 0]]                               # Permeabilidade do volume da esquerda
        KR = self.permeability[self.vols_pairs[:, 1]]                               # Permeabilidade do volume da direita

        KL = KL.reshape((self.nvols_pairs, 3, 3))                                   # Redimensiona para preservar a matriz de permeabilidade
        KR = KR.reshape((self.nvols_pairs, 3, 3))

        KnL = np.einsum("ij,ij->i", self.N, np.einsum("ij,ikj->ik", self.N, KL))    # Calcula a permeabilidade normal      
        KnR = np.einsum("ij,ij->i", self.N, np.einsum("ij,ikj->ik", self.N, KR))    

        Keq = (KnL * KnR) / ((KnL * self.h_R) + (KnR * self.h_L))                   # Calcula a permeabilidade equivalente
        self.faces_trans = Keq * self.faces_areas[self.internal_faces]              # Calcula a transmissibilidade por face

    
    def __assemble_TPFA_matrix(self, q: callable) -> None:
        """
        Monta a matriz A e o vetor b do sistema linear TPFA
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not callable(q):
            raise TypeError("\t q must be a callable function!")
        # ===========================================================================================================#
        row_index_p = self.vols_pairs[:, 0]
        col_index_p = self.vols_pairs[:, 1]
        
        row_index   = np.hstack((row_index_p, col_index_p))
        col_index   = np.hstack((col_index_p, row_index_p))
        data        = np.hstack((-self.faces_trans, -self.faces_trans))

        self.At_TPFA = csr_matrix((data, (row_index, col_index)), 
                                   shape=(self.nvols, self.nvols))
        
        
        xv, yv, zv = self.volumes_center[:, 0], self.volumes_center[:, 1], self.volumes_center[:, 2]
        self.bt_TPFA = q(xv, yv, zv, self.permeability) * self.volumes              # Calcula o vetor b

        self.At_TPFA = self.At_TPFA.tolil()                                         # Converte para lil para facilitar a montagem da matriz A
        self.At_TPFA.setdiag(-self.At_TPFA.sum(axis=1))                             # Soma as contribuições de cada volume na diagonal
        self.At_TPFA = self.At_TPFA.tocsr()
        if self.check:
            assert np.allclose(self.At_TPFA.sum(axis = 1), np.zeros(shape = (self.nvols, 1))), "Matriz A não foi montada corretamente"

    def __set_dirichlet_boundary_conditions(self, bc : callable) -> None:
        """
        Calcula as contribuições da condição de contorno de Dirichlet
        bc deve aceitar como parâmetro x, y e z + indices das faces do contorno e a malha
        e retornar None para os volumes que não possuem condição de contorno
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not callable(bc):
            raise TypeError("\t bc must be a callable function!")
        # ===========================================================================================================#
        coords = self.faces_center[self.boundary_faces]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]


        d_values    = (bc(x, y, z, self.boundary_faces, self.mesh))                 # Calcula os valores da CC de Dirichlet
        d_faces     = self.boundary_faces[d_values != None]                         # Pega as faces que possuem CC de Dirichlet     
        d_values    = d_values[d_values != None].astype('float')                    # Pega os valores da CC de Dirichlet nas faces que às possuem
        
        assert len(d_faces) > 1, "Não há condições de contorno de Dirichlet"
        mask        = np.isin(d_faces, self.faces_with_bc)                          # Verifica se as faces com CC de Dirichlet estão no conjunto de faces com CC de Dirichlet
        d_faces     = d_faces[mask == False]                                        # Remove as faces que estão no conjunto de faces com CC
        d_values    = d_values[mask == False]          
        
        self.faces_with_bc = np.setdiff1d(self.faces_with_bc, d_faces)

        d_volumes   = self.mesh.faces.bridge_adjacencies(d_faces, 2, 3).astype('int').flatten()

        self.d_faces, self.d_values, self.d_volumes = d_faces, d_values, d_volumes

        Nb, vL  = self.__get_normal(d_faces, d_volumes)[:2]                         # Calcula os vetores normais e os vetores que vão do centro da face para o centro do volume
        
        "TODO: Os vetores normais devem estar alinhados com vL? Ou devem todos ter um mesmo sentido?"

        Nb = np.abs(Nb) / np.linalg.norm(Nb, axis = 1)[:, None]                     # Normaliza os vetores normais
        
        hb = np.abs(np.einsum("ij,ij->i", Nb, vL))

        K = self.permeability[d_volumes].reshape((len(d_volumes), 3, 3))
        K = np.einsum("ij,ikj->ik", Nb, K)
        K = np.einsum("ij,ij->i", Nb, K) 
        
        factor = (K * self.faces_areas[self.d_faces]) / (hb)
        Ad_V = np.zeros(shape = (self.nvols))
        np.add.at(Ad_V, d_volumes, factor)
        
        row_index = np.arange(self.nvols)
        col_index = np.arange(self.nvols)
        
        self.Ad_TPFA = csr_matrix((Ad_V, (row_index, col_index)), 
                                   shape=(self.nvols, self.nvols))
        
        self.bd_TPFA = np.zeros(shape = (self.nvols))
        np.add.at(self.bd_TPFA, d_volumes, d_values * factor)
    
    def __set_neumann_boundary_conditions(self, bc : callable) -> None:
        """
        Calcula as contribuições da condição de contorno de Neumann
        bc deve aceitar como parâmetro x, y e z + indices das faces do contorno e a malha
        e retornar None para os volumes que não possuem condição de contorno
        """
        # ==== ERROR HANDLING =======================================================================================#
        if not callable(bc):
            raise TypeError("\t bc must be a callable function!")
        # ===========================================================================================================#
        coords = self.mesh.faces.center[self.boundary_faces]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        n_values = bc(x, y, z, self.boundary_faces, self.mesh)
        n_faces = self.boundary_faces[n_values != None]
        n_values = n_values[n_values != None]
        
        mask = np.isin(n_faces, self.faces_with_bc)
        n_faces = n_faces[mask == False]
        n_values = n_values[mask == False]
        self.faces_with_bc = np.setdiff1d(self.faces_with_bc, n_faces)

        self.n_faces, self.n_values = n_faces, n_values
        if len(n_faces) > 1:
            n_volumes = self.mesh.faces.bridge_adjacencies(n_faces, 2, 3).astype('int').flatten()
            self.n_volumes = n_volumes

            self.bn_TPFA = np.zeros(shape = (self.nvols))
            np.add.at(self.bn_TPFA, n_volumes, -n_values * self.volumes[n_volumes])

    def __solve_TPFA_system(self) -> None:
        """
        Resolve o sistema TPFA
        """
        A = self.At_TPFA
        b = self.bt_TPFA
        if self.Ad_TPFA is not None:
            A += self.Ad_TPFA
            b += self.bd_TPFA
        if self.bn_TPFA is not None:
            b += self.bn_TPFA
        A.eliminate_zeros()
        self.p_TPFA = spsolve(A, b)
        if self.check:
            assert np.allclose(A.dot(self.p_TPFA), b), "Solução do sistema TPFA não satisfaz a equação"
        
        
def main():
    directory = "mesh"
    # Open the mesh directory and run the solver for each mesh file
    solver = TPFAsolver(verbose=True)
    for meshfile in os.listdir(directory):
        mesh_name = meshfile.split(".")[0]
        if not meshfile.endswith(".msh"):
            continue
        meshfile = os.path.join(directory, meshfile)
        solver.set_name(mesh_name)
        mesh = helpers.load_mesh(meshfile)
        nvols = len(mesh.volumes)
        q = (np.arange(nvols) + 1).astype('float')
        K = helpers.get_random_tensor(1., 2., nvols)

        def neumann(x, y, z, faces, mesh):
            return np.full_like(faces, 0)

        def dirichlet(x, y, z, faces, mesh):
            return np.full_like(faces, 1.)
        
        args = {
                "meshfile"      : meshfile,
                "mesh"          : mesh,
                "mesh_params"   : None,
                "dim"           : 3,
                "permeability"  : K,
                "source"        : q,
                "neumann"       : neumann,
                "dirichlet"     : dirichlet,
                "analytical"    : None,
                "vtk"           : False
        }
        results = solver.solveTPFA(args)
        print(results.keys())
        break

if __name__ == "__main__":
    main()