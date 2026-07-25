import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	x = np.zeros(len(A[0]))
	D = np.diag(A)
	R = A - np.diagflat(D) 
	for i in range(n):
		x = (b - np.dot(R, x)) / D
	return x