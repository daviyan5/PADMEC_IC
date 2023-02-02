import numpy as np

def get_random_tensor(a, b, size, n = 3, m = 3, only_diagonal = True):
    """
    Retorna um array de tamanho size cujos elementos são tensores aleatórios n x m, com valores entre a e b
    Se only_diagonal for True, os elementos fora da diagonal serão iguais a zero, e o tensor será n x n

    """
    if only_diagonal:
        diags = np.random.uniform(a, b, size = size + (n,))
        return np.apply_along_axis(np.diag, len(diags.shape) - 1, diags)
    else:
        return np.random.uniform(a, b, size = size + (n, m))

def get_tensor(a, size, n = 3, m = 3, only_diagonal = True):
    """
    Retorna um array de tamanho size cujos elementos são tensores n x m, com valores iguais a a
    Se only_diagonal for True, os elementos fora da diagonal serão iguais a zero, e o tensor será n x n
    
    """
    if only_diagonal:
        diags = np.full(size + (n,), a)
        return np.apply_along_axis(np.diag, len(diags.shape) - 1, diags)
    else:
        return np.full(size + (n, m), a)