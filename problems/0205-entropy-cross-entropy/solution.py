import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.

    Args:
        P: True probability distribution
        Q: Predicted probability distribution

    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    if len(P) != len(Q):
        return (0.0, 0.0)

    entropy = 0.0
    cross = 0.0

    for p, q in zip(P, Q):
        if p > 0:
            entropy -= p * np.log(p)
            if q > 0:
                cross -= p * np.log(q)
            else:
                cross = float("inf")
                break
    return (entropy, cross)