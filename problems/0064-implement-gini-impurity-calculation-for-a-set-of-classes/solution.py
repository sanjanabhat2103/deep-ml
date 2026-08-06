
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	if len(y) == 0:
		return 0.0
	_, counts = np.unique(y, return_counts = True)
	probabilities = counts / len(y)
	val = 1 - np.sum(probabilities ** 2)
	return round(val,3)