import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	funcs = {'square': lambda v: (v ** 2, 2 * v), 'sin': lambda v: (np.sin(v), np.cos(v)), 'exp': lambda v: (np.exp(v), np.exp(v)), 'log': lambda v: (np.log(v),  1.0 / v)}
	curr_val = x
	grad_acc = 1.0
	for func in reversed(functions):
		if func not in funcs:
			raise ValueError
		f_val, f_grad = funcs[func](curr_val)
		grad_acc *= f_grad
		curr_val = f_val
	return float(grad_acc)