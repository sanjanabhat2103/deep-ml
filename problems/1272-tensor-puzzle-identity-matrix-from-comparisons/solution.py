import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    result = np.zeros((n, n), dtype = float)
    for i in range(n):
        result[i, i] = 1
    return result