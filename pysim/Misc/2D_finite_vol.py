# Solving the 2D TPFA equation using finite volume method

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

# ----------------- 2D TPFA equation -----------------
# => -∇ * (K * ∇p) = q
# Where: K = Permeability
#        p = Pressure
#        q = Source/sink term
# => -(K[i+1/2][j] * (p[i+1][j] - p[i][j]) - K[i-1/2][j] * (p[i][j] - p[i-1][j])) / dx 
#    -(K[i][j+1/2] * (p[i][j+1] - p[i][j]) - K[i][j-1/2] * (p[i][j] - p[i][j-1])) / dy = q[i][j]
# K[i + 1/2][j] = 2/(1/K[i][j] + 1/K[i + 1][j]) = Keqyf[i][j]
# K[i - 1/2][j] = 2/(1/K[i][j] + 1/K[i - 1][j]) = Keqyb[i][j]
# K[i][j + 1/2] = 2/(1/K[i][j] + 1/K[i][j + 1]) = Keqxf[i][j]
# K[i][j - 1/2] = 2/(1/K[i][j] + 1/K[i][j - 1]) = Keqxb[i][j]
# dx[i + 1/2] = x[i + 1] - x[i] = dxf[i]
# dx[i - 1/2] = x[i] - x[i - 1] = dxb[i]
# dy[j + 1/2] = y[j + 1] - y[j] = dyf[j]
# dy[j - 1/2] = y[j] - y[j - 1] = dyb[j]
# Contour conditions: P[0][0] = 1, q[-1][-1] = 1
# => -Keqyf * (p[i+1][j] - p[i][j]) / dxf + Keqyb * (p[i][j] - p[i-1][j]) / dxb 
#    -Keqxf * (p[i][j+1] - p[i][j]) / dyf + Keqxb * (p[i][j] - p[i][j-1]) / dyb = q[i][j]
# ----------------- Finite volume method -----------------
# Ghost Nodes: 
# i < 0     -> P[i][j] = -P[0][j]
# i > n - 1 -> P[i][j] = -P[n-1][j]
# j < 0     -> P[i][j] = -P[i][0]
# j > n - 1 -> P[i][j] = -P[i][n-1]

# Linear: Solve the matrix equation
# A * P = b
# Let k = (n - 1 - i) * m + j
# i + 1 -> k - m
# i - 1 -> k + m
# j + 1 -> k + 1
# j - 1 -> k - 1
# p[k - m] * (-Keqyf / (dxf*dx)) + p[k + m] * (-Keqyb / (dxb*dx)) + 
# p[k] * (Keqxf / (dxf*dx) + Keqxb / (dxb*dx) + Keqyf / (dyf*dx) + Keqyb / (dyb*dy)) +
# p[k + 1] * (-Keqxf / (dyf*dx)) + p[k - 1] * (-Keqxb / (dyb*dy)) = q[k]

def flip_all(A):
    for ar in A:
        ar = np.flip(ar,0)
    return A

def solve(K, P, x, y, q):
    n = len(x)
    m = len(y)
    Keqxf = 2 / (1 / np.delete(K, m - 1, axis=1) + 1 / np.delete(K, 0, axis=1))
    # Insert column at the beginning
    Keqxb = np.insert(Keqxf, 0, K[:,0], axis = 1)
    # Insert column at the end
    Keqxf = np.insert(Keqxf, n-1, K[:,-1], axis = 1)

    Keqyf = 2 / (1 / np.delete(K, n - 1, axis=0) + 1 / np.delete(K, 0, axis=0))
    # Insert 0 row at the end
    Keqyb = np.insert(Keqyf, m-1, K[-1], axis = 0)
    # Insert row at the beginning
    Keqyf = np.insert(Keqyf, 0, K[0], axis = 0)

    dxf = x[1:] - x[:-1]
    dxb = np.insert(dxf, 0, dxf[0])
    dxf = np.append(dxf, dxf[-1])

    dyf = y[1:] - y[:-1]
    dyb = np.insert(dyf, 0, dyf[0])
    dyf = np.append(dyf, dyf[-1])

    p = np.zeros((n * m))
    A = np.zeros((n * m, n * m))
    b = np.zeros((n * m))

    #print(np.around(Keqxf, 4))
    #print(np.around(Keqxb, 4))
    #print(np.around(Keqyf, 4))
    #print(np.around(Keqyb, 4))
    #Keqxf, Keqxb, Keqyf, Keqyb, dxf, dxb, dyf, dyb = flip_all((Keqxf, Keqxb, Keqyf, Keqyb, dxf, dxb, dyf, dyb))
    for i in range(n):
        for j in range(m):
            k = i * m + j
            b[k] = q[(n - 1 - i)][j]
            i = (n - 1 - i)
            # p[i+1][j], k - m
            c1 = (-Keqyf[i][j] / dxf[i])
            # p[i][j], k
            c2 = (Keqxf[i][j] / dxf[i] + Keqxb[i][j] / dxb[i] + Keqyf[i][j] / dyf[j] + Keqyb[i][j] / dyb[j])
            # p[i-1][j], k + m
            c3 = (-Keqyb[i][j] / dxb[i])
            # p[i][j+1], k + 1
            c4 = (-Keqxf[i][j] / dyf[j])
            # p[i][j-1], k - 1
            c5 = (-Keqxb[i][j] / dyb[j])
            i = (n - 1 - i)
            if i == n - 1:
                c2 += c1
                c1 = 0
            else:
                A[k][k + m] = c1
            if i == 0:
                c2 += c3
                c3 = 0
            else:
                A[k][k - m] = c3
            if j == 0:
                c2 += c5
                c5 = 0
            else:
                A[k][k - 1] = c5
            if j == m - 1:
                c2 += c4
                c4 = 0
            else:
                A[k][k + 1] = c4
            A[k][k] = c2
            #print("For node {}, {}:".format(i, j))
            #print("{} = {}, {} = {}, {} = {}, {} = {}, {} = {}".format((i+1,j), round(c1,3),(i-1, j), round(c3,3), 
            #                                                           (i,j), round(c2,3), (i,j+1), round(c4,3),(i,j-1), round(c5,3)))
    
    A[0] = 0
    A[0][0] = 1
    np.set_printoptions(suppress=True)
    p = np.linalg.solve(A, b)
    return np.flip(p.reshape(n, m),0), A




def main():
    # Define the grid
    n = 3
    x = np.linspace(0, n-1, n)
    y = np.linspace(0, n-1, n)
    # Define the permeability and source term
    K = np.ones((n, n))
    np.fill_diagonal(K, 1/1000)
    #K[0:n//2] = 1
    #K[n//2:n] = 1000
    q = np.zeros((n, n))

    # Define the initial pressure
    P = np.zeros((n, n))
    Po = 1
    Pf = 1
    q[-1][0] = Po
    q[0][-1] = Pf

    # Solve the equation
    start = time.time()
    P, A = solve(K, P, x, y, q)
    print("Finished linear solve in {} seconds".format(time.time() - start))

    # Plot P, K , q and A as heatmaps with values
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].set_title("Permeability")
    im = axs[0, 0].imshow(K, cmap='hot', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            axs[0, 0].text(j, i, round(K[i, j], 3), ha="center", va="center", color="w" if im.norm(K[i, j]) < 0.5 else "black")
    axs[0, 1].set_title("Pressure")
    im = axs[0, 1].imshow(P, cmap='hot', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            axs[0, 1].text(j, i, round(P[i, j], 3), ha="center", va="center", color="w" if im.norm(P[i, j]) < 0.5 else "black")
    axs[1, 0].set_title("Source term")
    im = axs[1, 0].imshow(q, cmap='hot', interpolation='nearest')
    for i in range(n):
        for j in range(n):
            axs[1, 0].text(j, i, round(q[i, j], 3), ha="center", va="center", color="w" if im.norm(q[i, j]) < 0.5 else "black")
    axs[1, 1].set_title("Matrix A")
    im = axs[1, 1].imshow(A, cmap='hot', interpolation='nearest')
    for i in range(n * n):
        for j in range(n * n):
            axs[1, 1].text(j, i, round(A[i, j], 4), ha="center", va="center", color="w" if im.norm(A[i, j]) < 0.5 else "black")
    
    plt.tight_layout()
    plt.show()




    



if __name__ == '__main__':
    main()