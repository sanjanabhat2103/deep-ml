import numpy as np

def rref(matrix):
    """
    Compute the Reduced Row Echelon Form (RREF) of a matrix.

    Args:
        matrix: Input matrix as a list of lists or numpy array

    Returns:
        RREF matrix as a numpy array
    """
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape

    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find row with largest pivot element
        max_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))

        # Skip if pivot is zero
        if abs(A[max_row, col]) < 1e-10:
            continue

        # Swap rows
        A[[pivot_row, max_row]] = A[[max_row, pivot_row]]

        # Normalize pivot row
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]

        # Eliminate column entries in other rows
        for row in range(rows):
            if row != pivot_row:
                factor = A[row, col]
                A[row] -= factor * A[pivot_row]

        pivot_row += 1

    # Remove numerical noise
    A[np.abs(A) < 1e-10] = 0

    return A