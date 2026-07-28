import math

def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
        x (list[int]): A list of binary values (0 or 1).
        y (list[int]): A list of binary values (0 or 1).

    Returns:
        float: The Phi coefficient rounded to 4 decimal places.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("Input lists cannot be empty.")
    a = b = c = d = 0
    for xi, yi in zip(x, y):
        if xi == 1 and yi == 1:
            a += 1
        elif xi == 1 and yi == 0:
            b += 1
        elif xi == 0 and yi == 1:
            c += 1
        elif xi == 0 and yi == 0:
            d += 1
        else:
            raise ValueError("Inputs must contain only 0s and 1s.")
    numerator = a * d - b * c
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        val = 0.0
    else:
        val = numerator / denominator
    return round(val, 4)