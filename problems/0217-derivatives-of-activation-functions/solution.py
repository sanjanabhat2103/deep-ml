import math
import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	sig = 1 / (1 + np.exp(-x))
	sigmoid = sig * (1 - sig)
	tanh_val = math.tanh(x)
	tanh = 1 - (tanh_val) ** 2
	relu = 1.0 if x >0 else 0.0
	return {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}