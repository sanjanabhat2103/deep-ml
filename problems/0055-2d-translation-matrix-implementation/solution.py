import numpy as np

def translate_object(points, tx, ty):
    points = np.array(points)
    translation = np.array([tx, ty])
    return points + translation