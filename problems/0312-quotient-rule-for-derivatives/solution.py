import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g = np.poly1d(g_coeffs)
    h = np.poly1d(h_coeffs)
    g_prime = g.deriv()
    h_prime = h.deriv()
    g_val = g(x)
    h_val = h(x)
    g_prime_val = g_prime(x)
    h_prime_val = h_prime(x)
    num = g_prime_val * h_val - h_prime_val * g_val
    den = h_val ** 2
    return float(num / den)