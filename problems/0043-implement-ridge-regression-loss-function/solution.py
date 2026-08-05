import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    y_pred = X @ w
    mse = np.mean((y_pred - y_true) ** 2)
    l2_penalty = alpha * np.sum(w ** 2)
    return mse + l2_penalty