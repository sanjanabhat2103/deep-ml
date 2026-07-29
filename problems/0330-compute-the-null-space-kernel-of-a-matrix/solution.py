import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.

    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero

    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    U, S, Vt = np.linalg.svd(A)

    # Number of nonzero singular values
    rank = np.sum(S > tol)

    # Columns of V corresponding to zero singular values
    null_space = Vt[rank:].T

    return null_space