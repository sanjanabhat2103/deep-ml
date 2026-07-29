import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    """
    Solve Ax = b using the Gauss-Seidel iterative method.

    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector
        n: Number of iterations
        x_ini: Initial guess (optional)

    Returns:
        Approximate solution vector x
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if x_ini is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.array(x_ini, dtype=float)

    for _ in range(n):
        for i in range(len(b)):
            # Sum of A[i][j] * x[j] excluding diagonal element
            sigma = 0
            for j in range(len(b)):
                if j != i:
                    sigma += A[i, j] * x[j]

            # Update x[i] using latest values
            x[i] = (b[i] - sigma) / A[i, i]

    return x