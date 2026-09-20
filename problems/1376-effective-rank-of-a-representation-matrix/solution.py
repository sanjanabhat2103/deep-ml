import numpy as np

def effective_rank(H, tol=1e-12):
    """
    Exponentiated Shannon entropy of the normalized singular values.

    Args:
        H: 2D array
        tol: singular values below this are discarded

    Returns:
        float in [1, number of retained singular values]
    """
    sing_vals = np.linalg.svd(H, compute_uv = False)
    sing_vals = sing_vals[sing_vals > tol]
    if len(sing_vals) == 0:
        return 1.0  
    n_sing_vals = sing_vals / np.sum(sing_vals)
    entropy = -np.sum(n_sing_vals * np.log(n_sing_vals))
    return np.exp(entropy)