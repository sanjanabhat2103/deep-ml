import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)

    Returns: (loss, grad) where grad has the same shape as preds
    """
    eps = 1e-12
    probs = np.clip(preds, eps, 1.0)
    N = probs.shape[0]
    idx = np.arange(N)
    losses = -np.log(probs[idx, target])
    grad = np.zeros_like(probs)
    grad[idx, target] = -1.0 / probs[idx, target]
    if reduction == "mean":
        loss = losses.mean()
        grad /= N
    elif reduction == "sum":
        loss = losses.sum()
    elif reduction == "none":
        loss = losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    return loss, grad