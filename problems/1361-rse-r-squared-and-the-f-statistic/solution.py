import numpy as np


def model_fit_stats(X: np.ndarray, y: np.ndarray) -> tuple:
    """Residual standard error, R-squared and the overall F-statistic.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (rse, r2, f_stat) as floats. RSS == 0 gives (0.0, 1.0, inf).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    mean = np.mean(y)
    X_design = np.column_stack((np.ones(n), X))
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    residuals = y - y_hat
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - mean) ** 2))
    if np.isclose(rss, 0.0):
        return 0.0, 1.0, np.inf
    r2 = 1.0 - rss / tss if not np.isclose(tss, 0.0) else 0.0
    df_resid = n - p - 1
    rse = float(np.sqrt(rss / df_resid))
    ssr = tss - rss
    f_stat = float((ssr / p) / (rss / df_resid))
    return rse, float(r2), f_stat
