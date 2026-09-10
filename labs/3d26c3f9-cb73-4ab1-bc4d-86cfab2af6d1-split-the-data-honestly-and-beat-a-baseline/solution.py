import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """
    Seeded shuffle split of (X, y), fit a mean baseline on train, evaluate MAE on test.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    train_frac, val_frac, test_frac : float
        Target fractions. Use int(n * frac) for train and val; remainder -> test.
    seed : int
        RNG seed for the shuffle.

    Returns
    -------
    mae : float
        Mean absolute error of the train-mean baseline on the test set.
    train_idx, val_idx, test_idx : np.ndarray
        1-D integer index arrays (a partition of range(n_samples)).
    """
    # TODO: permute indices with a seeded Generator
    # TODO: slice train / val / test index blocks
    # TODO: mu = mean of y[train_idx]; MAE of mu on y[test_idx]
    n = len(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(n * train_frac)
    val_end = int(n * val_frac) + train_end
    train = indices[: train_end]
    val = indices[train_end: val_end]
    test = indices[val_end: ]
    mu = np.mean(y[train])
    mae = np.mean(np.abs(y[test] - mu))
    return mae, train, val, test
