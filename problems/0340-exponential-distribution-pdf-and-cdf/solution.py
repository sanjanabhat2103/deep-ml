import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.

    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution

    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    if lam <= 0:
        return {
            "pdf": None,
            "cdf": None,
            "mean": None,
            "variance": None,
        }

    pdf = [
        round(float(lam * np.exp(-lam * xi)), 4) if xi >= 0 else 0.0
        for xi in x
    ]

    cdf = [
        round(float(1 - np.exp(-lam * xi)), 4) if xi >= 0 else 0.0
        for xi in x
    ]

    return {
        "pdf": pdf,
        "cdf": cdf,
        "mean": round(1 / lam, 4),
        "variance": round(1 / (lam ** 2), 4),
    }