import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    probabilities = np.asarray(probabilities, dtype = float)
    log_probs = np.log(probabilities)
    mean_log_prob = np.mean(log_probs)
    perplexity = np.exp(-mean_log_prob)
    return perplexity