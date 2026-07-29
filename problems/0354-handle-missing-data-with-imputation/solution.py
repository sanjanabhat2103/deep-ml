import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.

    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        2D numpy array with missing values imputed
    """
    data = np.array(data, dtype=float, copy=True)

    if strategy not in ('mean', 'median', 'mode'):
        raise ValueError("strategy must be 'mean', 'median', or 'mode'")

    for col in range(data.shape[1]):
        column = data[:, col]
        mask = np.isnan(column)

        if not np.any(mask):
            continue

        if strategy == 'mean':
            fill_value = np.nanmean(column)

        elif strategy == 'median':
            fill_value = np.nanmedian(column)

        else:  
            valid = column[~mask]
            if valid.size == 0:
                fill_value = np.nan
            else:
                values, counts = np.unique(valid, return_counts=True)
                fill_value = values[np.argmax(counts)]

        column[mask] = fill_value
        data[:, col] = column

    return data
