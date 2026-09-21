import numpy as np

def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple:
    """
    Calculate AIC and BIC for model selection.
    
    Args:
        y_true: True target values
        y_pred: Predicted values from the model
        k: Number of parameters in the model
    
    Returns:
        Tuple of (AIC, BIC)
    """
    rss = np.sum((y_true - y_pred) ** 2)
    n = len(y_true)
    aic = n * np.log(rss / n) + k * 2
    bic = n * np.log(rss / n) + k * np.log(n)
    return (aic, bic)
