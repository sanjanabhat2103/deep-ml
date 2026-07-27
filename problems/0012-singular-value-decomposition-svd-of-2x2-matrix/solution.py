import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 numpy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ np.diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    ATA = A.T @ A
    a = ATA[0, 0]
    b = ATA[0, 1]
    d = ATA[1, 1]
    if abs(b) < 1e-12:
        c, s = 1.0, 0.0
    else:
        theta = 0.5 * np.arctan2(2 * b, a - d)
        c = np.cos(theta)
        s = np.sin(theta)
    V = np.array([[c, -s],
                  [s,  c]])
    B = A @ V
    sigma1 = np.sqrt(B[0, 0]**2 + B[1, 0]**2)
    sigma2 = np.sqrt(B[0, 1]**2 + B[1, 1]**2)
    if sigma1 < sigma2:
        sigma1, sigma2 = sigma2, sigma1
        B = B[:, [1, 0]]
        V = V[:, [1, 0]]
    S = np.array([sigma1, sigma2])
    U = np.zeros((2, 2))
    if sigma1 > 1e-12:
        U[:, 0] = B[:, 0] / sigma1
    else:
        U[:, 0] = np.array([1.0, 0.0])
    if sigma2 > 1e-12:
        U[:, 1] = B[:, 1] / sigma2
    else:
        U[:, 1] = np.array([-U[1, 0], U[0, 0]])
    Vt = V.T
    return U, S, Vt