import torch
from torch.distributions import LogisticNormal as LogisticNormalTorch
from torch.distributions import constraints

__all__ = ['LogisticNormal']

class LogitNormal(torch.distributions.TransformedDistribution):
    r"""
    Creates a logit-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `SigmoidTransform` such that::

        X ~ LogitNormal(loc, scale)
        Y = logit(X) ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = torch.distributions.Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, torch.distributions.SigmoidTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogitNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        # Estimate the mean via sample average
        return self.sample((100,)).mean(0)
