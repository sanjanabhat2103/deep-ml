import numpy as np

def pairwise_cosine_similarity(X):
    X = np.asarray(X, dtype = float)
    norms = np.linalg.norm(X, axis = 1, keepdims = True)
    X_normalized = X / np.where(norms == 0, 1, norms)
    return X_normalized @ X_normalized.T