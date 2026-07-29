import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    # Augmented matrix
    Ab = np.hstack((A, b.reshape(-1, 1)))

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot row
        max_row = i + np.argmax(np.abs(Ab[i:, i]))

        # Swap rows
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        # Check for singular matrix
        if abs(Ab[i, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")

        # Eliminate entries below pivot
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x