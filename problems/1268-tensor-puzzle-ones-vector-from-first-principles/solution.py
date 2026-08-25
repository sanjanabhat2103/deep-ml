import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    o = []
    for i in range(n):
        o.append(1)
    return np.asarray(o, dtype = float)