import numpy as np

def pad_to(a: np.ndarray, j: int) -> np.ndarray:
    """Pad a with zeros (or truncate) to length j."""
    if j < len(a):
        return a[: j]
    else:
        out = np.zeros(j, dtype = a.dtype)
        out[: len(a)] = a
        return out
        