import numpy as np

def matrix_image(A):
    A = np.array(A, dtype = float)
    m, n = A.shape
    R = A.copy()
    pivot_cols = []
    row = 0
    tol = 1e-10
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if abs(R[r, col]) > tol:
                pivot = r
                break
        if pivot is None:
            continue
        R[[row, pivot]] = R[[pivot, row]]
        for r in range(row + 1, m):
            factor = R[r, col] / R[row, col]
            R[r] -= factor * R[row]
        pivot_cols.append(col)
        row += 1
        if row == m:
            break
    return A[: , pivot_cols]