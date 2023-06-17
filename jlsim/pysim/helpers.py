import numpy as np
import sys
import os
import solver

from solver import TPFAsolver
from mesh_generator import MeshGenerator


def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout.close()
    sys.stdout = sys.__stdout__

# ----------- Print Helpers --------------------------------#
def print_idx():
    i = 0
    while True:
        yield i
        i += 1

print_generator = print_idx()

def reset_verbose():
    global print_generator
    print_generator = print_idx()
    print("-" * 100)

def verbose(msg, type_msg):
    lim = 100
    suf1 = " " + "=" * lim
    suf2 = " " + '+' * lim
    pref1 = "-- "
    if type_msg == "OUT":
        print("\n")
        msg += suf1
    elif type_msg == "CHECK":
        msg += suf2
    elif type_msg == "INFO":
        msg = pref1 + msg
    msg = msg[:lim] if len(msg) > lim else msg
    print("[{}]\t{:<100}".format(next(print_generator), msg), end="")
    print("\t{:<3}".format(type_msg))
#-------------------------------------------#

def create_meshes(box_dimensions, num = 10, suffix = "", check = True) -> list:
    """
    Creates num meshes, with different nvols, and return the list of the names of the meshfiles + suffix
    If check, we check that every mesh has a different number of volumes, and that the number of volumes is increasing
    """
    meshes = []
    volumes_list = []
    Lx, Ly, Lz = box_dimensions
    incremento = 4
    fator = 2.6
    def get_order(last_order):
        return int(np.ceil((last_order ** (1/fator)) + incremento) ** fator)
    
    order = 10
    meshlist = []
    for i in range(num):
        order = get_order(last_order = order)
        verbose("Creating mesh {} with order {} and dimension {}.".format(i, order, box_dimensions), type_msg = "INFO")
        meshfile = "box_{}{}.msh".format(i, suffix)
        meshfile = MeshGenerator().create_box(box_dimensions, order, filename=meshfile)
        if check:
            mesh = load_mesh(meshfile)
            nvols = len(mesh.volumes)
            
            if i > 0:
                assert nvols not in volumes_list, "Repeated number of volumes"
                assert nvols > volumes_list[-1], "Number of volumes not increasing"
            volumes_list.append(nvols)
            verbose("Checked mesh {} with {} volumes".format(meshfile, nvols), type_msg = "CHECK")
        meshlist.append(meshfile)
    return meshlist

def load_mesh(meshfile):
    sys.path.append('../')
    from preprocessor.meshHandle.finescaleMesh import FineScaleMesh                     # type: ignore
    block_print()
    mesh = FineScaleMesh(meshfile)
    enable_print()
    return mesh

def get_mesh_params(mesh):
    
    faces_areas = solver.TPFAsolver.get_areas(mesh, mesh.faces.all)
    volumes = solver.TPFAsolver.get_volumes(mesh, mesh.volumes.all, faces_areas)
    return faces_areas, volumes

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