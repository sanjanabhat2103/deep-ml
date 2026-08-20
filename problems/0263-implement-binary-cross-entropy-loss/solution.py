import numpy as np

def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	if len(y_true) != len(y_pred):
		raise ValueError("The arrays must be of equal length.")
	if len(y_true) == 0:
		raise ValueError("The array cannot be empty.")
    loss = 0.0
    for y, p in zip(y_true, y_pred):
        p = max(epsilon, min(1 - epsilon, p))
        loss += y * np.log(p) + (1 - y) * np.log(1 - p)
    return -loss / len(y_true)