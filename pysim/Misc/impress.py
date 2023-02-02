import numpy as np
import time
import warnings

from preprocessor.meshHandle.finescaleMesh import FineScaleMesh
from scipy.sparse import csr_matrix, SparseEfficiencyWarning
from scipy.sparse.linalg import spsolve


def get_random_tensor(a, b, size):
    """
    Retorna um array de tamanho size cujos elementos são tensores aleatórios diagonais 3 x 3, com valores entre a e b

    """
    diags = np.random.uniform(a, b, size = size + (3,))
    return np.apply_along_axis(np.diag, len(size), diags)

def main():
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    start_time = time.time()
    mesh = FineScaleMesh("./mesh/20.h5m")
    n_vols = len(mesh.volumes)
    q = np.zeros((n_vols))
    permeability = get_random_tensor(1, 1, (n_vols,))

    is_dirichlet_node = np.array([True] + [False for i in range(n_vols - 1)])
    mesh.is_neumann_node = np.array([False for i in range(n_vols - 1)] + [True] )
    dirichlet = np.array([1.] + [0. for i in range(n_vols - 1)])
    neumann = np.array([0. for i in range(n_vols - 1)]+ [1.])

    internal_faces = mesh.faces.internal[:]

    # Retorna os volumes (3) que compartilham uma face (2)
    in_vols_pairs = mesh.faces.bridge_adjacencies(internal_faces, 2, 3)
    internal_faces = mesh.faces.internal[:]
    # Retorna o centro dos volumes que compartilham uma face
    L = mesh.volumes.center[in_vols_pairs[:, 0]]
    R = mesh.volumes.center[in_vols_pairs[:, 1]]

    # Retorna os nós que compoẽm as faces internas
    internal_faces_nodes = mesh.faces.bridge_adjacencies(internal_faces, 0, 0)
    I_idx = internal_faces_nodes[:, 0]
    J_idx = internal_faces_nodes[:, 1]
    K_idx = internal_faces_nodes[:, 2]

    I = mesh.nodes.coords[I_idx]
    J = mesh.nodes.coords[J_idx]
    K = mesh.nodes.coords[K_idx]

    

    # Set the normal vectors.
    Ns = 0.5 * np.cross(I - J, K - J)
    Ns_norm = np.linalg.norm(Ns, axis=1)

    n_vols_pairs = len(mesh.faces.internal)

    lvols = in_vols_pairs[:, 0]
    rvols = in_vols_pairs[:, 1]

    KL = permeability[lvols]
    KR = permeability[rvols]

    KnL_pre = np.einsum("ij,ikj->ik", Ns, KL)
    KnR_pre = np.einsum("ij,ikj->ik", Ns, KR)

    Kn_L = np.einsum("ij,ij->i", KnL_pre, Ns) / Ns_norm ** 2
    Kn_R = np.einsum("ij,ij->i", KnR_pre, Ns) / Ns_norm ** 2

    LJ = J - L
    LR = J - R

    h_L = np.abs(np.einsum("ij,ij->i", Ns, LJ) / Ns_norm)
    h_R = np.abs(np.einsum("ij,ij->i", Ns, LR) / Ns_norm)

    # Compute the face transmissibilities.
    Kn_prod = Kn_L * Kn_R
    Keq = Kn_prod / ((Kn_L * h_R) +
                        (Kn_R * h_L))
    faces_trans = Keq * Ns_norm
    
    # Set transmissibilities in matrix.
    trans_time = time.time()
    n_vols = len(mesh.volumes)

    data = np.hstack((-faces_trans, -faces_trans))
    row_idx = np.hstack((in_vols_pairs[:, 0], in_vols_pairs[:, 1]))
    col_idx = np.hstack((in_vols_pairs[:, 1],in_vols_pairs[:, 0]))
    print("Faces T time: \t\t\t {} s".format(np.around(time.time() - trans_time, 5)))

    tpfam_time = time.time()
    A_tpfa = csr_matrix((data, (row_idx, col_idx)), shape=(n_vols, n_vols))

    A_tpfa.setdiag(-A_tpfa.sum(axis=1))
    print("TPFA matrix time: \t\t {} s".format(np.around(time.time() - tpfam_time, 5)))

    # Boundary
    boundary_time = time.time()
    dirichlet_nodes = np.nonzero(is_dirichlet_node)[0]
    neumann_nodes = np.setdiff1d(np.nonzero(mesh.is_neumann_node)[0], dirichlet_nodes)
    for i in dirichlet_nodes:
        A_tpfa.data[A_tpfa.indptr[i] : A_tpfa.indptr[i + 1]] = 0.
    A_tpfa[dirichlet_nodes, dirichlet_nodes] = 1
    q[dirichlet_nodes] = dirichlet[dirichlet_nodes]

    q[neumann_nodes] += neumann[neumann_nodes]
    A_tpfa.eliminate_zeros()
    print("Boundary condition time: \t {} s".format(np.around(time.time() - boundary_time, 5)))

    pressure_time = time.time()
    mesh.pressure[:] = spsolve(A_tpfa, q)
    print("Pressure solving time: \t\t {} s".format(np.around(time.time() - pressure_time, 5)))
    print("Total simulation time: \t\t {} s".format(np.around(time.time() - start_time, 5)))

    vtk_time = time.time()
    mesh.permeability[:] = np.reshape(permeability, (n_vols,9))
    mesh.is_dirichlet_node[:] = is_dirichlet_node
    mesh.dirichlet[:] = dirichlet
    mesh.neumann[:] = neumann

    meshset = mesh.core.mb.create_meshset()
    mesh.core.mb.add_entities(meshset, mesh.core.all_volumes)
    mesh.core.mb.write_file("./mesh_impress.vtk", [meshset])
    print("Saving VTK time: \t\t {} s".format(np.around(time.time() - vtk_time, 5)))
    np.save("Atpfa.npy", A_tpfa.todense())
    
if __name__ == "__main__":
    main()