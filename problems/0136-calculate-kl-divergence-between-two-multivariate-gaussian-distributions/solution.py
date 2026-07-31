import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    k = mu_p.shape[0]
    
    # Inverse and log-determinant of Cov_q
    inv_Cov_q = np.linalg.inv(Cov_q)
    logdet_Cov_p = np.linalg.slogdet(Cov_p)[1]
    logdet_Cov_q = np.linalg.slogdet(Cov_q)[1]
    
    # Difference between means
    diff = mu_q - mu_p
    
    # Terms of the multivariate KL divergence formula:
    # KL(P || Q) = 0.5 * (tr(Cov_q^{-1} Cov_p) + (mu_q - mu_p)^T Cov_q^{-1} (mu_q - mu_p) - k + ln(|Cov_q| / |Cov_p|))
    trace_term = np.trace(inv_Cov_q @ Cov_p)
    quad_term = diff.T @ inv_Cov_q @ diff
    det_term = logdet_Cov_q - logdet_Cov_p
    
    kl_div = 0.5 * (trace_term + quad_term - k + det_term)
    
    return float(kl_div)