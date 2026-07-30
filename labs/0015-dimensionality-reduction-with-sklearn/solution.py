import numpy as np

class MyReducer:
    def __init__(self):
        self.n_components = 10
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # PCA via SVD
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
