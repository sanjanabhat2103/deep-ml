import numpy as np 

def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
	"""
	Predict the probability of choosing each Thanksgiving dish using softmax.
	
	Args:
		preference_scores: List of preference scores for each dish
		(e.g., [turkey_score, stuffing_score, cranberry_score, pie_score])
		
	Returns:
		List of probabilities for each dish
	"""
	scores = np.asarray(preference_scores, dtype = float)
	exps = np.exp(scores - np.max(scores))
	total = np.sum(exps)
	return (exps / total).tolist()