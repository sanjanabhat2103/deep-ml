import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    results = []
    w = w0.copy()
    for i in range(n_steps):
        if n_steps == 1:
            lr = 10 ** a
        else:
            lr = 10 ** (a + (b - a) * i / (n_steps - 1))
        y_pred = X @ w
        error = y_pred - y
        grad = (2 / len(y)) * (X.T @ error)
        w -= lr * grad
        loss = np.mean((X @ w - y) ** 2)
        results.append(float(loss))
    return results