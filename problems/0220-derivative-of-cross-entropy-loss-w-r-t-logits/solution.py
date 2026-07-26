import math
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	max_logit = max(logits)
	shifted = [x - max_logit for x in logits]
	exps = [math.exp(x) for x in shifted]
	sum_exps = sum(exps)
	softmax_probs = [exp / sum_exps for exp in exps]
	gradient = softmax_probs.copy()
	gradient[target] -= 1
	return gradient