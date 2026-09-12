import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions, dtype = float)
    y_true = np.asarray(y_true, dtype = float)
    mean = np.mean(predictions, axis = 0)
    bias_sq = np.mean((mean - y_true) ** 2)
    variance = np.mean(np.var(predictions, axis = 0))
    mse = bias_sq + variance
    return {'bias_squared': bias_sq, 'variance': variance, 'mse': mse}

