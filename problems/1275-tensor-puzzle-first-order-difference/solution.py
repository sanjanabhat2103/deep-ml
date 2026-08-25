import numpy as np

def diff(a: np.ndarray) -> np.ndarray:
    """out[0]=a[0]; out[i]=a[i]-a[i-1] for i>0."""
    out = np.empty_like(a)
    if len(a) == 0:
        return out
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = a[i] - a[i - 1]
    return out