import numpy as np

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: Numpy array representing the projection of v onto L
    """
    v = np.array(v, dtype=float)
    L = np.array(L, dtype=float)

    if len(v) != len(L):
        raise ValueError("Vectors must have the same dimension")

    if np.dot(L, L) == 0:
        raise ValueError("Projection line vector cannot be zero")

    scalar = np.dot(v, L) / np.dot(L, L)

    return scalar * L