import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    out = []
    for i in range(A.shape[0]):
        out.append(A[i][i])
    return np.array(out)