import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    a = np.asarray(a, dtype = float)
    indices = np.arange(len(a) - 1, -1, -1)
    return a[indices]