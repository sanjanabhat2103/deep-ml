import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    c = counts.copy()
    for i in range(len(counts)):
        s = sum(counts[i])
        for j in range(len(counts[i])):
            c[i][j] = counts[i][j] / s if s != 0 else 0.0
    return c