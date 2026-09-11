import numpy as np

def linspace(start, stop, n: int) -> np.ndarray:
    """n evenly spaced values from start to stop inclusive."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return np.asarray([start])
    diff = (stop - start) / (n - 1)
    return np.asarray([start + i * diff for i in range(n)])
    