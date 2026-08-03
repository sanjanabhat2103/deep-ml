import numpy as np

def to_categorical(x, n_col=None):
    """
    Convert class labels to one-hot encoded vectors.

    Args:
        x: Array-like class labels
        n_col: Number of classes (optional)

    Returns:
        One-hot encoded numpy array
    """
    x = np.asarray(x, dtype=int).ravel()

    if n_col is None:
        n_col = np.max(x) + 1

    categorical = np.zeros((x.shape[0], n_col))
    categorical[np.arange(x.shape[0]), x] = 1

    return categorical