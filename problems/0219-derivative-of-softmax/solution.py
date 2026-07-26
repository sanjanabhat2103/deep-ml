import math
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	n = len(x)
	max_x = max(x)
	exps = [math.exp(v - max_x) for v in x]
	sum_exps = sum(exps)
	s = [exp / sum_exps for exp in exps]
	J = [[0.0] * n for i in range(n)]
	for i in range(n):
		for j in range(n):
			if i == j:
				J[i][j] = s[i] * (1 - s[i])
			else:
				J[i][j] = -s[i] * s[j]
	return J 