def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	m = len(matrix)
	n = len(matrix[0])
	for i in range(m):
		for j in range(n):
			matrix[i][j] *= scalar
	return matrix