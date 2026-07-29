import numpy as np

def calculate_dot_product(vec1, vec2):
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The dot product of the two vectors.
	"""
	dot_prod = 0
	if len(vec1) != len(vec2):
		raise ValueError("Lengths of the two vectors must be equal")
	n = len(vec1)
	for i in range(n):
		dot_prod += vec1[i] * vec2[i]
	return dot_prod