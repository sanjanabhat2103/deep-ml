import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """
    accepted = [i for i in scores if i >= threshold]
    acc_copy = sorted(accepted.copy(), reverse = True)[0: n_select]
    accepted_indices = [scores.index(i) for i in acc_copy]
    acceptance_rate = round(len(accepted) / len(scores), 4)
    mean_quality = round(sum(acc_copy) / len(acc_copy), 4) if len(acc_copy) != 0 else 0.0
    return {'accepted_indices': accepted_indices, 'acceptance_rate': acceptance_rate, 'mean_quality': mean_quality}
