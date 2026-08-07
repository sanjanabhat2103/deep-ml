import numpy as np

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio.

    Args:
        cov_matrix (list[list[float]]): Covariance matrix of asset returns.
        weights (list[float]): Portfolio weights.

    Returns:
        float: Portfolio variance.
    """
    cov_matrix = np.asarray(cov_matrix, dtype = float)
    weights = np.asarray(weights, dtype = float)
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")
    if cov_matrix.shape[0] != weights.shape[0]:
        raise ValueError("The number of weights must match the size of the covariance matrix.")
    return float(weights.T @ cov_matrix @ weights)