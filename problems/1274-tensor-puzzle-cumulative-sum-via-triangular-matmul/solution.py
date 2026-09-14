import numpy as np

def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""
    return np.tril(np.ones((len(a), len(a)), dtype = a.dtype)) @ a