import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    """
    Calculate the margin width of a linear SVM classifier.
    
    Parameters:
    w : np.ndarray - weight vector defining the hyperplane
    
    Returns:
    float - the total margin width
    """
    w = np.asarray(w, dtype = float)
    norm_w = np.linalg.norm(w)
    return 2 / norm_w if norm_w != 0 else 0.0