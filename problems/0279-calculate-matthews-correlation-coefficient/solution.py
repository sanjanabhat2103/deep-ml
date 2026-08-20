import numpy as np

def matthews_correlation_coefficient(y_true: list, y_pred: list) -> float:
    """
    Calculate the Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: List of actual binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
    
    Returns:
        MCC value rounded to 4 decimal places
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("The arrays must be of equal length.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("The array cannot be empty.")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num / den if den != 0 else 0.0
    return round(float(mcc), 4)