import torch

def torch_softplus_inverse(x):
    """
    Numerically stable inverse of the softplus function.
    """
    return x + torch.log(-torch.expm1(-x))
