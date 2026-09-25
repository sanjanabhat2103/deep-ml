import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    d_k = K.shape[1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    scores = scores - np.max(scores, axis = 1, keepdims = True)
    exps = np.exp(scores)
    weights = exps / np.sum(exps, axis = 1, keepdims = True)
    return np.round(weights, 4).tolist()