import numpy as np

def flatten(A: np.ndarray) -> np.ndarray:
    """Row-major flatten of 2-D array A without reshape/ravel."""
    n_rows = len(A)
    n_cols = len(A[0])
    out = np.zeros(n_rows * n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            out[i * n_cols + j] = A[i, j]
    return out