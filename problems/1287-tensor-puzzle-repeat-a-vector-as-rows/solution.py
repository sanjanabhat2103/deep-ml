import numpy as np

def repeat_rows(a: np.ndarray, d: int) -> np.ndarray:
    """Stack d copies of 1-D array a as rows."""
    ans = []
    for i in range(d):
        ans.append(a.tolist())
    return np.array(ans)
