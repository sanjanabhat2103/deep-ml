import numpy as np

def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of 1-D arrays a and b via broadcasting."""
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)
    return  a[:, None] * b[None:, ]