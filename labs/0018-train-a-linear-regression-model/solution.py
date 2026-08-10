import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.

    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)

    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)
    X_aug = np.column_stack((X, np.ones(X.shape[0])))
    params, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    W = params[:-1]
    b = params[-1]
    return W, b