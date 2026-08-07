import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.

    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)

    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    try:
        A = np.asarray(A, dtype = float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return -1
        if not np.allclose(A, A.T):
            return -1
        L = np.linalg.cholesky(A)
        return L.tolist()
    except (np.linalg.LinAlgError, ValueError):
        return -1