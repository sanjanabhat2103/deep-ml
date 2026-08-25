import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    o = np.ones(len(a))
    return np.dot(o, a)