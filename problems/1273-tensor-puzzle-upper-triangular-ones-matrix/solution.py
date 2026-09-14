import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            mat[i][j] = 1.0
    return mat
