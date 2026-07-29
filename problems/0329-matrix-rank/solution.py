import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.

    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero

    Returns:
        The rank of the matrix (integer)
    """
    A = np.array(A, dtype=float, copy=True)

    rows, cols = A.shape
    rank = 0

    for col in range(cols):
        # Find pivot row with largest value in this column
        pivot_row = rank + np.argmax(np.abs(A[rank:, col]))

        # Check if pivot is effectively zero
        if rank >= rows or abs(A[pivot_row, col]) < tol:
            continue

        # Swap pivot row into place
        A[[rank, pivot_row]] = A[[pivot_row, rank]]

        # Eliminate entries below pivot
        for row in range(rank + 1, rows):
            factor = A[row, col] / A[rank, col]
            A[row, col:] -= factor * A[rank, col:]

        rank += 1

        # Stop if all rows are processed
        if rank == rows:
            break

    return rank