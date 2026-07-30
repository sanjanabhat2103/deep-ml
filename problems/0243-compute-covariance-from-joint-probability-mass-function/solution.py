import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.

    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])

    Returns:
        Covariance of X and Y as a float
    """
    x = np.array(x_values)
    y = np.array(y_values)
    px = np.sum(joint_pmf, axis = 1)
    py = np.sum(joint_pmf, axis = 0)
    Ex = np.sum(x * px)
    Ey = np.sum(y * py)
    Exy = np.sum(joint_pmf * np.outer(x, y))
    return Exy - Ex * Ey
