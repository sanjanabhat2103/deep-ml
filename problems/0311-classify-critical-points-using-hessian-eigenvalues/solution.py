import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    hess = np.array(hessian, dtype = float)
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
        raise ValueError
    if not np.allclose(hess, hess.T, atol = tol):
        raise ValueError
    ev = np.linalg.eigvalsh(hess)
    if np.any(np.abs(ev) <= tol):
        return None
    elif np.all(ev > tol):
        return -1
    elif np.all(ev < -tol):
        return 1
    else:
        return 0