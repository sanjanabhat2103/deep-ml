import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):

    variance_ratio_log = np.log(sigma_q / sigma_p)
    term_trace = (sigma_p**2 + (mu_p - mu_q)**2) / (2.0 * sigma_q**2)
    return variance_ratio_log + term_trace - 0.5