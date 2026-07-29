import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """
    Compute the change-of-basis matrix from basis B to basis C.

    Args:
        B: Basis matrix whose columns are the basis vectors of B.
        C: Basis matrix whose columns are the basis vectors of C.

    Returns:
        The change-of-basis matrix P such that [v]_C = P @ [v]_B.
    """
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    P = np.linalg.inv(C) @ B
    return P.tolist()
