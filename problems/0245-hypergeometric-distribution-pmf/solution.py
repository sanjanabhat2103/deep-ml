import math
def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution.
    
    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes
    
    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    def ncr(n, r):
        if r < 0 or r > n:
            return 0.0
        return (math.factorial(n) / ((math.factorial(r)) * (math.factorial(n - r))))
    value = ncr(K, k) * ncr(N - K, n - k) / ncr(N, n)
    return round(value, 4)