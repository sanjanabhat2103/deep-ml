import numpy as np 

def clip_gradients_by_global_norm(gradients: list, max_norm: float) -> list[np.ndarray]:
    """
    Clip gradients by global norm.
    
    Args:
        gradients: List of gradient arrays or lists of varying shapes.
        max_norm: Maximum allowed global norm.
    
    Returns:
        List of clipped gradient arrays as numpy arrays.
    """
    grads_arr = [np.asarray(g, dtype = float) for g in gradients]
    global_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads_arr))
    if global_norm > max_norm:
        scaling_factor = max_norm / (global_norm + 1e-6)
        return [g * scaling_factor for g in grads_arr]
    return grads_arr