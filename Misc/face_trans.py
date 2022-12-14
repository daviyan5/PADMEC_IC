import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

nx = 3
ny = 4
K = np.array([[(ny - i - 1) * nx + j + 1 for j in range(nx)] for i in range(ny)])
Kv = 2 / (1/K[:-1, :] + 1/K[1:, :])
Kv = np.insert(Kv,  0, K[ 0, :], axis = 0)
Kv = np.insert(Kv, ny, K[-1, :], axis = 0)

Kh = 2 / (1/K[:, :-1] + 1/K[:, 1:])
Kh = np.insert(Kh,  0, K[:,  0], axis = 1)
Kh = np.insert(Kh, nx, K[:, -1], axis = 1)
print(K)
print(np.flip(Kv, 0))
print(np.flip(Kh, 0))

face_trans_v = np.flip(Kv, 0).flatten()
face_trans_h = np.flip(Kh, 0).flatten()
print(face_trans_v)
print(face_trans_h)

face_trans = np.hstack((-face_trans_v, -face_trans_h))
print(face_trans)

n_faces = nx * (ny + 1) + ny * (nx + 1)

# montar lista de adj
# calcular csr_matrix
        

# K * (p - P)/h[1/2 + i] /h[1,i]