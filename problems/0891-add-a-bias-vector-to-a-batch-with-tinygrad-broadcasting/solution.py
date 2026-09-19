from tinygrad import Tensor

def add_bias(x: Tensor, b: Tensor) -> Tensor:
    return x + b