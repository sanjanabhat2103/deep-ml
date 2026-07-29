import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    if len(vectors) == 0:
        return True
    matrix = np.array(vectors, dtype = float).T
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)