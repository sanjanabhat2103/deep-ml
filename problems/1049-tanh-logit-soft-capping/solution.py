import numpy as np

def tanh_soft_cap(logits, softcap):
    """Apply tanh soft-capping to logits."""
    logits = np.asarray(logits, dtype = float)
    if softcap is None or softcap <= 0:
        out = logits
    else:
        out = softcap * np.tanh(logits / softcap)
    return np.round(out, 6)
