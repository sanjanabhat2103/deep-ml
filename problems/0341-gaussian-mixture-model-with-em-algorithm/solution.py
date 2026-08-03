import numpy as np

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    """
    Fit a 1D Gaussian Mixture Model using the EM algorithm.
    
    Args:
        X: List of data points
        K: Number of mixture components
        initial_means: List of initial means for each component
        initial_variances: List of initial variances for each component
        initial_weights: List of initial mixture weights (should sum to 1)
        n_iterations: Number of EM iterations to run
    
    Returns:
        Dictionary with 'means', 'variances', 'weights' as lists rounded to 4 decimals
    """
    X = np.asarray(X, dtype=float)

    means = np.asarray(initial_means, dtype=float)
    variances = np.asarray(initial_variances, dtype=float)
    weights = np.asarray(initial_weights, dtype=float)

    N = len(X)
    eps = 1e-6

    for _ in range(n_iterations):
        # E-step
        responsibilities = np.zeros((N, K))

        for k in range(K):
            var = max(variances[k], eps)
            coeff = 1.0 / np.sqrt(2 * np.pi * var)
            exponent = np.exp(-((X - means[k]) ** 2) / (2 * var))
            responsibilities[:, k] = weights[k] * coeff * exponent

        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= row_sums

        # M-step
        Nk = responsibilities.sum(axis=0)

        weights = Nk / N

        for k in range(K):
            means[k] = np.sum(responsibilities[:, k] * X) / Nk[k]

            variances[k] = (
                np.sum(responsibilities[:, k] * (X - means[k]) ** 2)
                / Nk[k]
            )

        # Prevent zero variances during computation
        variances = np.maximum(variances, eps)

    return {
        "means": [round(float(m), 4) for m in means],
        "variances": [
            0.0 if v <= 1e-5 else round(float(v), 4)
            for v in variances
        ],
        "weights": [round(float(w), 4) for w in weights],
    }