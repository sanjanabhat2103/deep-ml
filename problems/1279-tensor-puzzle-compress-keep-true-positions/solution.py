import numpy as np

def compress(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack v[g] into the front of a zero vector of length len(v)."""
    out = np.zeros_like(v)
    selected = v[g]
    out[: len(selected)] = selected
    return out
