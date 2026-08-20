import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    X = np.asarray(X)
    X_centred = X - np.mean(X, axis = 0)
    cov_matrix = np.cov(X_centred, rowvar = False)
    ev = np.linalg.eigvalsh(cov_matrix)
    evs = np.sort(ev)[: : -1]
    eig_sum = np.sum(evs)
    return evs / eig_sum