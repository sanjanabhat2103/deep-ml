from tinygrad import Tensor, dtypes

def to_float_tensor(values):
    return Tensor(values, dtype = dtypes.float32)