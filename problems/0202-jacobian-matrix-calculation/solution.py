import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
        
    Returns:
        Jacobian matrix as list of lists
    """
    n = len(x)
    f_base = f(x)
    m = len(f_base)
    jacobian = [[0.0] * n for _ in range(m)]
    for j in range(n):
        x_perturbed = list(x)
        x_perturbed[j] += h
        f_perturbed = f(x_perturbed)
        for i in range(m):
            jacobian[i][j] = (f_perturbed[i] - f_base[i]) / h   
    return jacobian
