import numpy as np

def engram_context_gating(h, e, W_K, W_V, eps=1e-6):
    # Project memory
    K = e @ W_K
    V = e @ W_V

    # RMSNorm
    rms = np.sqrt(np.mean(K**2, axis=-1, keepdims=True) + eps)
    K = K / rms

    # Sigmoid gate
    gate = 1 / (1 + np.exp(-K))

    # Gate the value projection
    return gate * V