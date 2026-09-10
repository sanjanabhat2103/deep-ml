import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    predictions = X_batch @ weights + bias
    errors = predictions - y_batch
    grad_weights = (2 / len(batch_indices)) * (X_batch.T @ errors)
    grad_bias = 2 * np.mean(errors)
    updated_weights = weights - lr * grad_weights
    updated_bias = bias - lr * grad_bias
    return np.concatenate([updated_weights, [updated_bias]])
