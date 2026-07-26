import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    product = np.polynomial.polynomial.polymul(f_coeffs, g_coeffs)
    derivative = [(i + 1) * product[i + 1] for i in range(len(product) - 1)]
    if not derivative:
        return [0.0]
    return [round(float(c), 4) for c in derivative]
    
    