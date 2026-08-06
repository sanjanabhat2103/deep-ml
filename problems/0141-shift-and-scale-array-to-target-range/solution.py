import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    values = np.asarray(values, dtype = float)
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.full_like(values, (c + d) / 2, dtype=float)
    scaled = (values - min_val) / (max_val - min_val)
    return c + scaled * (d - c)