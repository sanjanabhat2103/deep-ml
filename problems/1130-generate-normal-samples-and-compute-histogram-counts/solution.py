import numpy as np

def fn(seed, mean, std, n, bins):
    # return (counts, edges) as plain Python lists
    np.random.seed(seed)
    samples = np.random.normal(loc = mean, scale = std, size = n)
    counts, edges = np.histogram(samples, bins = bins)
    return counts.tolist(), [float(e) for e in edges]