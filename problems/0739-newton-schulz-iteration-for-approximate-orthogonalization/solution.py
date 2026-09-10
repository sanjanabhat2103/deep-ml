import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    M = np.asarray(M, dtype = float)
    frob_norm = np.linalg.norm(M, ord = "fro")
    if frob_norm == 0:
        return M.tolist()
    M = M / frob_norm
    for i in range(num_iters):
        prod = M @ M.T
        M = a * M + b * (prod @ M) + c * (prod ** 2 @ M)
    return M.tolist()