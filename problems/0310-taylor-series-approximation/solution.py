import numpy as np
import math
from math import factorial

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    Compute Taylor series approximation for common functions.
    
    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of terms in the series
    
    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    f = 0
    if func_name == 'exp':
        for i in range(n_terms):
            f += (x ** i) / math.factorial(i)
    elif func_name == 'sin':
        for i in range(n_terms):
            j = 2 * i + 1
            f += (-1) ** i * (x ** j) / math.factorial(j)
    elif func_name == 'cos':
        for i in range(n_terms):
            j = 2 * i
            f += (-1) ** i * (x ** j) / math.factorial(j)
    return f