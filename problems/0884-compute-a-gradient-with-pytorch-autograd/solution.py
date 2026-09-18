import torch

def grad_of_quadratic(x_value: float) -> float:
    x = torch.tensor(float(x_value), requires_grad = True)
    f = x ** 2 + 3 * x + 2
    f.backward()
    return x.grad.item()