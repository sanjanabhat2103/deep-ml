def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues = []
	trace = matrix[0][0] + matrix[1][1]
	det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	disc = trace ** 2 - 4 * det 
	e1 = (trace + disc ** 0.5) / 2
	e2 = det / e1
	eigenvalues.extend([e1, e2])
	eigenvalues.sort(reverse = True) 
	return eigenvalues