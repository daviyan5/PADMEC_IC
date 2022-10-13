import matplotlib.pyplot as plt
import math
import numpy as np

# Solve -u'' = f(x) on (l1,l2) x (l3,l4) and u = K on the boundary
# using finite differences
def FiniteDiff(f, n1, n2, l1 = 0, l2 = 1, l3 = 0, l4 = 1, K = 0):

    h1 = (l2 - l1) / (n1 + 1)
    h2 = (l4 - l3) / (n2 + 1)
    ratio = (h1 / h2) ** 2
    x1 = np.linspace(l1 + h1, l2 - h1, n1)
    x2 = np.linspace(l3 + h2, l4 - h2, n2)
    F = np.array([[f(x1[i],x2[j]) + (K if (i == 0 or j == 0 or i == n1-1 or j == n2-1) else 0) for j in range(n2)] for i in range(n1)]).flatten()
    F *= h1**2

    print("\n############### F ###############\n\n",F,"\n\n#################################\n")

    # u[i][j] * 2 * (1 + ratio) * h2**2 = f(x1[i], x2[j]) * h1**2 * h2**2 + u[i-1][j] * h2**2 + u[i+1][j] * h2**2 + u[i][j-1] * h1**2 + u[i][j+1] * h1**2
    # u[i][j] = (f(x1[i], x2[j]) * h1**2 + u[i-1][j] + u[i+1][j] + u[i][j-1] * ratio + u[i][j+1] * ratio) / (2 * (1 + ratio) )
    # let k = i * n2 + j
    # u[k] = (f[k] * h1**2 + u[k-n2] + u[k+n2] + u[k-1] * ratio + u[k+1] * ratio) / (2 * (1 + ratio) )
    # u[k] = (F + u[k-n2] + u[k+n2] + u[k-1] * ratio + u[k+1] * ratio) / (2 * (1 + ratio) )
    # F = u[k] * (2 * (1 + ratio)) - u[k-n2] - u[k+n2] - u[k-1] * ratio - u[k+1] * ratio
    
    A = np.diag(v = 2 * (1 + ratio) * np.ones(n1 * n2), k = 0)
    A += np.diag(v = -np.ones(n1 * n2 - n2), k = n2) 
    A += np.diag(v = -np.ones(n1 * n2 - n2), k = -n2) 
    A += np.diag(v = -ratio * np.ones(n1 * n2 - 1), k = 1) 
    A += np.diag(v = -ratio * np.ones(n1 * n2 - 1), k = -1)       # u[k+1]

    print("\n############### A ###############\n\n",A,"\n\n#################################\n")

    u = np.linalg.solve(A, F)
    #print(x1)
    #print(x2)
    #print(u)
    u = np.array([[K if (i == 0 or j == 0 or i == n1+1 or j == n2+1) else u[(i-1) * n2 + j-1] for j in range(n2 + 2)] for i in range(n1 + 2)])
    x1 = np.append(l1, x1)
    x1 = np.append(x1, l2)
    x2 = np.append(l3, x2)
    x2 = np.append(x2, l4)

    return x1, x2, u

def main():
    f = lambda x1, x2: x2 + x1
    x1, x2, U = FiniteDiff(f, 3, 3, K = 0)
    X1, X2 = np.meshgrid(x1, x2)

    print("\n############### x1 ###############\n\n",x1,"\n\n#################################\n")
    print("\n############### x2 ###############\n\n",x2,"\n\n#################################\n")
    print("\n############### U ###############\n\n",U,"\n\n#################################\n")
    print("Done!")
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X1, X2, U, 50, cmap='binary')
    ax.set_title('surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



if __name__ == '__main__':
    main()