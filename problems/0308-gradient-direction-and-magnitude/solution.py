import numpy as np
def gradient_direction_magnitude(gradient: list) -> dict:
    g = np.array(gradient, dtype = float)
    magnitude = np.linalg.norm(g)
    if magnitude == 0:
        direction = np.zeros_like(g).tolist()
        descent_direction = np.zeros_like(g).tolist()
    else:
        direction = (g / magnitude).tolist()
        descent_direction = (-g / magnitude).tolist()
    return {
        "magnitude": float(magnitude),
        "direction": direction,
        "descent_direction": descent_direction
    }
