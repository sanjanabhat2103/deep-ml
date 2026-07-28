import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        return 0.0
    if not p or not q:
        return 0.0
    BC = 0
    n = len(p)
    for i in range(n):
        BC += (p[i] * q[i]) ** 0.5
    dist = -np.log(BC)
    return np.round(dist, 4)