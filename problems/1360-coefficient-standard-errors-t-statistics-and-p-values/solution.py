import math
import numpy as np

def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)
    n, p = X.shape
    D = np.column_stack((np.ones(n), X))
    DTD_inv = np.linalg.inv(D.T @ D)
    coef = DTD_inv @ D.T @ y
    residuals = y - D @ coef
    rss = np.sum(residuals ** 2)
    df = n - p - 1
    sigma_sq = rss / df
    cov = sigma_sq * DTD_inv
    se = np.sqrt(np.diag(cov))
    t = coef / se
    p_values = np.array([1.0 - math.erf(abs(value) / math.sqrt(2.0)) for value in t])
    return coef, se, t, p_values
