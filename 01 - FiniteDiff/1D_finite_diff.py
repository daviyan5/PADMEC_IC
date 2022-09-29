import matplotlib.pyplot as plt
import numpy as np
import math

# Solve -u'' = f(x) on [a,b] with u(a) = u(b) = K 
# using finite differences
def FiniteDiff(f, N = 1000, a = 0, b = 1, K = 0):
    h =  (b - a) / (N + 1)
    x = np.linspace(a + h, b - h, N)
    print(x)
    F = f(x) * h**2
    F[0] += K
    F[-1] += K 
    A = np.diag(v = 2 * np.ones(N), k = 0) + np.diag(v = -np.ones(N - 1), k = 1) + np.diag(v = -np.ones(N - 1), k = -1)
    #print(A)
    #print(F)
    u = np.linalg.solve(A, F)
    u = np.append(K, u)
    u = np.append(u, K)
    x = np.append(a, x)
    x = np.append(x, b)
    
    return x, u

def main():
    f = lambda x: np.log(x) * np.sin(x)
    x, u = FiniteDiff(f, N = 4, a = 1000, b = 2000, K = 5)
    print("Done!")
    plt.plot(x, u)
    plt.grid()
    plt.show()
    #print(x)
    #print(u)


if __name__ == '__main__':
    main()