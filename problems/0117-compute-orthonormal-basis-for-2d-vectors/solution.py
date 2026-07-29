import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    """
    Compute an orthonormal basis from a set of vectors using Gram-Schmidt.

    Args:
        vectors: List of vectors.
        tol: Tolerance for considering a vector as zero.

    Returns:
        List of orthonormal basis vectors.
    """
    basis = []

    for v in vectors:
        v = np.array(v, dtype = float)

        # Subtract projections onto existing basis vectors
        for u in basis:
            v = v - np.dot(v, u) * u

        # Normalize if vector is not linearly dependent
        norm = np.linalg.norm(v)

        if norm > tol:
            basis.append(v / norm)

    return basis