import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.size
    rmse_res = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
    return round(rmse_res, 3)