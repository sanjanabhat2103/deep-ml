from collections import Counter
import math 

def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	if not apples:
		return 0.0
	n = len(apples)
	counts = Counter(apples)
	entropy = 0.0
	for count in counts.values():
		prob = count / n 
		entropy -= prob * math.log2(prob)
	return entropy