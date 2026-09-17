import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	y_true = np.asarray(y_true, dtype = float)
	y_pred = np.asarray(y_pred, dtype = float)
	a = y_true - y_pred
	return np.mean(np.where(np.abs(a) < delta, 0.5 * a ** 2, delta * (np.abs(a) - 0.5 * delta)))