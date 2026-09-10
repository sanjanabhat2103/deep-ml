import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	singular_values = np.linalg.svd(delta_W, compute_uv = False)
	energy = singular_values ** 2
	cumulative_energy = np.cumsum(energy)
	total_energy = energy.sum()
	if total_energy == 0:
		return 0
	k = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
	return int(k)