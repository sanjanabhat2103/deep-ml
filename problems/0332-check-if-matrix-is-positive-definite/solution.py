import numpy as np

def check_positive_definite(matrix: list) -> dict:
    matrix = np.array(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square")

    ev = np.linalg.eigvals(matrix)
    ev = [round(float(x), 4) for x in np.sort(ev)]

    is_pd = all(x > 1e-10 for x in ev)

    return {
        "is_positive_definite": is_pd,
        "eigenvalues": ev,
    }