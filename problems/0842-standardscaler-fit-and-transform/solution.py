import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    X_train = np.asarray(X_train, dtype = float)
    X_test = np.asarray(X_test, dtype = float)
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    std[std == 0.0] = 1.0
    return (X_test - mean) / std
