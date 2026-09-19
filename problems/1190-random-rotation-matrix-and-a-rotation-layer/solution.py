import numpy as np

def rotation_layer(X, angle):
    X = np.asarray(X, dtype = float)
    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return X @ rot_matrix