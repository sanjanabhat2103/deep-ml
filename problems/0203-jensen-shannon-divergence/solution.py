import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.

    Args:
        P: First probability distribution
        Q: Second probability distribution

    Returns:
        Jensen-Shannon Divergence value (using natural logarithm, in nats)
    """
    if len(P) != len(Q):
        return 0.0

    P = np.array(P, dtype=float)
    Q = np.array(Q, dtype=float)

    # Mixture distribution
    M = 0.5 * (P + Q)

    def kl_divergence(A, B):
        kl = 0.0
        for a, b in zip(A, B):
            if a > 0:
                kl += a * np.log(a / b)
        return kl

    jsd = 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

    return float(jsd)