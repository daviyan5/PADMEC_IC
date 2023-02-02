# Solving the 1D TPFA equation using finite volume method

import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------- 1D TPFA equation -----------------
# => -∇ * (K * ∇p) = q
# Where: K = Permeability
#        p = Pressure
#        q = Source/sink term
# => -(d(K * dP/dx) / dx) = q
# => -(K[i + 1/2] * (P[i + 1] - P[i]) / dx[i+1/2]) - (K[i - 1/2] * (P[i] - P[i - 1]) / dx[i-1/2]) = q[i]
# K[i + 1/2] = 2/(1/K[i] + 1/K[i + 1]) = Keqf
# K[i - 1/2] = 2/(1/K[i] + 1/K[i - 1]) = Keqb 
# dx[i+1/2] = x[i + 1] - x[i] = dxf
# dx[i-1/2] = x[i] - x[i - 1] = dxb
# Contour conditions: P[0] = Po, P[-1] = Pf

# ----------------- Finite volume method -----------------

# Naive: For loop
# P[i+1] = P[i] + (Keqb/Keqf) * (dxf/dxb) * (P[i] - P[i-1]) + q[i] * dxf/Keqf

def naive_solve(K, P, x, q):
    n = len(P)
    for i in range(n - 1):
        Keqf = 2 / (1 / K[i] + 1 / K[i + 1])
        Keqb = 2 / (1 / K[i] + 1 / K[i - 1])
        dxf = x[i + 1] - x[i]
        dxb = x[i] - x[i - 1]
        P[i + 1] = P[i] + (Keqb / Keqf) * (dxf / dxb) * (P[i] - P[i - 1]) + q[i] * dxf / Keqf
    return P

# Linear: Solve the matrix equation
# A * P = b
# (Keqf/dxf) * P[i + 1] - (Keqf/dxf) * P[i] - (Keqb/dxb) * P[i] + (Keqb/dxb) * P[i - 1] = q[i]
# (Keqf/dxf) * P[i + 1] - (Keqf/dxf + Keqb/dxb) * P[i] + (Keqb/dxb) * P[i - 1] = q[i]
# A = (Keqf/dxf + Keqb/dxb) * I - (Keqf/dxf) * E + (Keqb/dxb) * E
# b = q

def linear_solve(K, P, x, q):
    
    n = len(P)
    Keqf = 2 / (1 / K[:-1] + 1 / K[1:])
    Keqb = np.insert(Keqf, 0, 0)
    Keqf = np.append(Keqf, 0)

    dxf = x[1:] - x[:-1]
    dxb = np.insert(dxf, 0, dxf[0])
    dxf = np.append(dxf, dxf[-1])

    assert len(Keqf) == len(Keqb) == len(dxf) == len(dxb) == n
    A = -np.diag(Keqf / dxf + Keqb / dxb) + np.diag(Keqf / dxf, k = 1)[0:n, 0:n] + np.diag(Keqb / dxb, k = -1)[1:n+1, 1:n+1]
    A[0] = A[-1] = np.zeros(n)
    A[0, 0] = A[n-1, n-1] = 1

    b = q
    b[0] = P[0]
    b[-1] = P[-1]
    P = np.linalg.solve(A, b)
    print(A)
    print(P)
    print(b)
    return P

def main():
    # Define the grid
    n = 200
    x = np.linspace(0, n-1, n)

    # Define the permeability and source term
    K = np.ones(n)
    #K[0:n//2] = 1
    #K[n//2:n] = 1000
    q = np.zeros(n)

    # Define the initial pressure
    P = np.zeros(n)
    Po = 200
    Pf = 100
    P[0] = Po
    P[-1] = Pf

    # Solve the equation
    start = time.time()
    Pn = naive_solve(K, P, x, q)
    print("Finished naive solve in {} seconds".format(time.time() - start))
    start = time.time()
    Pl = linear_solve(K, P, x, q)
    print("Finished linear solve in {} seconds".format(time.time() - start))


    # Plot Pn, Pl, K and q, difference between Pn and Pl
    plt.figure(figsize = (12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(x, Pn, 'o-', label = 'Naive')
    plt.plot(x, Pl, 'o-', label = 'Linear')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('P')
    plt.title('Pressure')
    
    plt.subplot(2, 2, 2)
    plt.plot(x, K, 'o-')
    plt.xlabel('x')
    plt.ylabel('K')
    plt.title('Permeability')

    plt.subplot(2, 2, 3)
    plt.plot(x, q, 'o-')
    plt.xlabel('x')
    plt.ylabel('q')
    plt.title('Source/sink term')

    plt.subplot(2, 2, 4)
    plt.plot(x, abs(Pn - Pl), 'o-')
    plt.xlabel('x')
    plt.ylabel('Pn - Pl')
    plt.title('Difference between Pn and Pl')

    plt.tight_layout()
    plt.show()
    



if __name__ == '__main__':
    main()