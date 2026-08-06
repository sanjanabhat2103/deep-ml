from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    if not val_losses:
        return (-1, -1)
    best_epoch = 0
    best_loss = val_losses[0]
    wait = 0
    stop_epoch = len(val_losses) - 1
    for epoch in range(1, len(val_losses)):
        if best_loss - val_losses[epoch] > min_delta:
            best_loss = val_losses[epoch]
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            stop_epoch = epoch
            break
    return stop_epoch, best_epoch