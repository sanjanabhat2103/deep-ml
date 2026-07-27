import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> (np.ndarray, float, list[float]):
    weights = initial_weights.astype(float).copy()
    bias = float(initial_bias)
    n_samples = features.shape[0]
    mse_values = []
    for _ in range(epochs):
        z = features @ weights + bias
        predictions = 1 / (1 + np.exp(-z))
        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(float(mse), 4))
        dL_dy = 2 * errors / n_samples
        dy_dz = predictions * (1 - predictions)
        delta = dL_dy * dy_dz
        grad_weights = features.T @ delta
        grad_bias = np.sum(delta)
        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias
    updated_weights = np.round(weights, 4)
    updated_bias = round(float(bias), 4)
    return updated_weights, updated_bias, mse_values