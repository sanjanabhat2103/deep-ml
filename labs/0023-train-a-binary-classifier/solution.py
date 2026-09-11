import numpy as np
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    model = LogisticRegression(max_iter = 1000, random_state = 0)
    model.fit(X_train, y_train)
    def predict(X):
        return model.predict(X).astype(int)
    return predict