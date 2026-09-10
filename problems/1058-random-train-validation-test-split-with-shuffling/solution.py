import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = data.shape[0]
    indices = np.random.default_rng(seed).permutation(n)
    data_shuffled = data[indices]
    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)
    train = data_shuffled[0: train_end]
    validation = data_shuffled[train_end: validation_end]
    test = data_shuffled[validation_end: ]
    return [train, validation, test]