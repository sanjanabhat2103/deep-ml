import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    pred = np.asarray(pred, dtype = float)
    target = np.asarray(target, dtype = float)
    x = pred - target
    loss = np.where(np.abs(x) < beta, 0.5 * x ** 2 / beta, np.abs(x) - 0.5 * beta)
    if beta <= 0:
        raise ValueError
    if reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    elif reduction == 'none':
        return loss
    else:
        raise ValueError
