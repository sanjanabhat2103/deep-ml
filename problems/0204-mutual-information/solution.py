import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables.

    Args:
        joint_prob: 2D joint probability distribution P(X,Y)

    Returns:
        Mutual information I(X;Y) in bits.
    """
    joint = np.array(joint_prob, dtype=float)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            pxy = joint[i, j]
            if pxy > 0:
                mi += pxy * np.log(pxy / (px[i] * py[j]))
    return float(mi)