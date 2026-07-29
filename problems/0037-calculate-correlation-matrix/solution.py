import numpy as np

def calculate_correlation_matrix(X, Y=None):
    """
    Calculate the Pearson correlation matrix.

    Args:
        X: Array-like of shape (n_samples, n_features)
        Y: Optional array-like of shape (n_samples, n_features)

    Returns:
        Correlation matrix as a NumPy array.
        - If Y is None: returns the correlation matrix of X.
        - If Y is provided: returns the cross-correlation matrix between X and Y.
    """
    X = np.asarray(X, dtype=float)
    if Y is None:
        return np.corrcoef(X, rowvar=False)
    Y = np.asarray(Y, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples (rows).")
    combined = np.concatenate((X, Y), axis=1)
    corr = np.corrcoef(combined, rowvar=False)
    n_features_X = X.shape[1]
    n_features_Y = Y.shape[1]
    return corr[:n_features_X, n_features_X:n_features_X + n_features_Y]
