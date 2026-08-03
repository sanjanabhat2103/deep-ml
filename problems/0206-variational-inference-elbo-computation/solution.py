import numpy as np

def compute_elbo(x: list[float], q_mean: float, q_std: float,
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    x = np.asarray(x)

    # Sample from q(z)
    z = np.random.normal(q_mean, q_std, n_samples)

    # Expected log-likelihood
    log_likelihood = np.sum(
        -0.5 * np.log(2 * np.pi * likelihood_std**2)
        - (x[None, :] - z[:, None])**2 / (2 * likelihood_std**2),
        axis=1
    )
    expected_log_likelihood = np.mean(log_likelihood)

    # Expected log-prior
    log_prior = (
        -0.5 * np.log(2 * np.pi * prior_std**2)
        - (z - prior_mean)**2 / (2 * prior_std**2)
    )
    expected_log_prior = np.mean(log_prior)

    # Analytical entropy of q(z)
    entropy = 0.5 * np.log(2 * np.pi * np.e * q_std**2)

    return expected_log_likelihood + expected_log_prior + entropy