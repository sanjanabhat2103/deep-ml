import numpy as np
from sklearn.metrics import accuracy_score
def accuracy_score(y_true, y_pred):
	y_true = np.array(y_true, dtype = float)
	y_pred = np.array(y_pred, dtype = float)
	if len(y_true) != len(y_pred):
		raise ValueError("Lengths of arrays must be equal")
	correct = 0
	for i in range(len(y_true)):
		if y_true[i] == y_pred[i]:
			correct += 1
	accuracy = correct / len(y_true)
	return accuracy