import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	data = np.array(data, dtype = float)
	mean = np.mean(data, axis = 0)
	std = np.std(data, axis = 0)
	standardized_data = (data - mean) / std	
	min_value = np.min(data, axis = 0)
	max_value = np.max(data, axis = 0)
	normalized_data = (data - min_value) / (max_value - min_value)
	return standardized_data, normalized_data