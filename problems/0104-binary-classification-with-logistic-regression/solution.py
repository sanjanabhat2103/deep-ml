import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	X = np.asarray(X, dtype = float)
	weights = np.asarray(weights, dtype = float)
	z = X @ weights + bias 
	res = []
	for element in z:
		if sigmoid(element) >= 0.5:
			res.append(1)
		else:
			res.append(0)
	return np.asarray(res)
		
	