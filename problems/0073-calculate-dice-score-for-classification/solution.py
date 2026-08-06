import numpy as np

def dice_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        res = 0.0
    else:
        res = (2 * intersection) / denominator
    return round(res, 3)