import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression,
    optimizing parameters with Binary Cross Entropy loss.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    losses = []
    for i in range(iterations):
        z = X @ weights + bias
        predictions = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-15
        p = np.clip(predictions, eps, 1 - eps)
        loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        losses.append(round(float(loss), 4))
        error = predictions - y
        grad_weights = X.T @ error
        grad_bias = np.sum(error)
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
    params = np.insert(weights, 0, bias)
    return np.round(params, 4).tolist(), losses
