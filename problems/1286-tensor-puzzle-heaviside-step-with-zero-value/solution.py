import numpy as np

def heaviside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise heaviside with zero-value b."""
    out = []
    for k in range(len(a)):
        if a[k] < 0:
            out.append(0)
        elif a[k] == 0:
            out.append(float(b[k]))
        else:
            out.append(1)
    return out