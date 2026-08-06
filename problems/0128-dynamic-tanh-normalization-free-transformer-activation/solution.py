import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    x = np.asarray(x, dtype = float)
    x = gamma * np.tanh(alpha * x) + beta
    return x