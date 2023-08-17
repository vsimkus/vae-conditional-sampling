import numpy as np
from einops import rearrange

class Int8bToFloatStandardTransform:

    def __init__(self, to_num_bits=8):
        self.from_bits = 8
        self.to_bits = to_num_bits

    def __call__(self, x):
        if self.to_bits != self.from_bits:
            x = np.floor(x / 2 ** (self.from_bits - self.to_bits))
            x /= (2 ** self.to_bits - 1)
            return x.astype(np.float32)
        return (x / (2 ** self.from_bits - 1)).astype(np.float32)

class ImgSpatialFlattenTransform:
    def __call__(self, x):
        return rearrange(x, '... h w c -> ... c (h w)')
