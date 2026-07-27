import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m = X.shape[0]

    for _ in range(n_epochs):

        if method == 'batch':
            predictions = X.dot(weights)
            error = predictions - y
            gradient = (2 / m) * X.T.dot(error)
            weights -= learning_rate * gradient

        elif method == 'stochastic':
            for i in range(m):
                xi = X[i]
                yi = y[i]

                prediction = xi.dot(weights)
                error = prediction - yi
                gradient = 2 * error * xi

                weights -= learning_rate * gradient

        elif method == 'mini_batch':
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)

                X_batch = X[start:end]
                y_batch = y[start:end]

                predictions = X_batch.dot(weights)
                error = predictions - y_batch
                gradient = (2 / len(X_batch)) * X_batch.T.dot(error)

                weights -= learning_rate * gradient

        else:
            raise ValueError("Invalid method")

    return weights