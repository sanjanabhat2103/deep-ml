import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    out = []
    for i in range(len(a)):
        out.append(a[(i + 1) % len(a)])
    return np.array(out)
