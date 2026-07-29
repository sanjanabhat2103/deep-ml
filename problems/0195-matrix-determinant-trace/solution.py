import numpy as np
def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists
	
	Returns:
		Tuple of (determinant, trace)
	"""
	arr = np.array(matrix, dtype = float)
	if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
		raise ValueError
	det = np.linalg.det(arr)
	trace = np.trace(arr)
	return (det, trace)