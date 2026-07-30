import numpy as np

def simulate_clt(distribution, n, runs, seed):
    """
    Demonstrate the Central Limit Theorem by sampling from a distribution,
    computing standardized sample means (Z-scores), and returning their
    mean and population standard deviation.

    Parameters
    ----------
    distribution : str
        One of: 'uniform', 'exponential', 'bernoulli'
    n : int
        Sample size for each run.
    runs : int
        Number of independent samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            'mean': float,
            'std': float
        }
    """
    np.random.seed(seed)

    if distribution == "uniform":
        samples = np.random.uniform(0, 1, size=(runs, n))
        mu = 0.5
        sigma = np.sqrt(1 / 12)

    elif distribution == "exponential":
        samples = np.random.exponential(1.0, size=(runs, n))
        mu = 1.0
        sigma = 1.0

    elif distribution == "bernoulli":
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        mu = 0.3
        sigma = np.sqrt(0.3 * 0.7)

    else:
        raise ValueError("Unsupported distribution")

    sample_means = samples.mean(axis=1)
    z_scores = (sample_means - mu) / (sigma / np.sqrt(n))

    return {
        "mean": float(z_scores.mean()),
        "std": float(z_scores.std(ddof=0)),
    }