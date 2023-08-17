import math
import functools
import operator

import torch
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


# Adapted from https://docs.pyro.ai/en/1.5.0/_modules/pyro/distributions/conjugate.html#BetaBinomial

class BetaBinomial(Distribution):
    r"""
    Compound distribution comprising of a beta-binomial pair. The probability of
    success (``probs`` for the :class:`~pyro.distributions.Binomial` distribution)
    is unknown and randomly drawn from a :class:`~pyro.distributions.Beta` distribution
    prior to a certain number of Bernoulli trials given by ``total_count``.

    :param concentration1: 1st concentration parameter (alpha) for the
        Beta distribution.
    :type concentration1: float or torch.Tensor
    :param concentration0: 2nd concentration parameter (beta) for the
        Beta distribution.
    :type concentration0: float or torch.Tensor
    :param total_count: Number of Bernoulli trials.
    :type total_count: float or torch.Tensor
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive,
                       'total_count': constraints.nonnegative_integer}
    has_enumerate_support = True
    support = torch.distributions.Binomial.support

    # EXPERIMENTAL If set to a positive value, the .log_prob() method will use
    # a shifted Sterling's approximation to the Beta function, reducing
    # computational cost from 9 lgamma() evaluations to 12 log() evaluations
    # plus arithmetic. Recommended values are between 0.1 and 0.01.
    approx_log_prob_tol = 0.

    def __init__(self, concentration1, concentration0, total_count=1, validate_args=None):
        concentration1, concentration0, total_count = broadcast_all(
            concentration1, concentration0, total_count)
        self._beta = torch.distributions.Beta(concentration1, concentration0)
        self.total_count = total_count
        super().__init__(self._beta._batch_shape, validate_args=validate_args)

    @property
    def concentration1(self):
        return self._beta.concentration1

    @property
    def concentration0(self):
        return self._beta.concentration0

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(BetaBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new._beta = self._beta.expand(batch_shape)
        new.total_count = self.total_count.expand_as(new._beta.concentration0)
        super(BetaBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=(), generator=None):
        # NOTE: override to allow passing a generator
        # probs = self._beta.sample(sample_shape)
        # sample = Binomial(self.total_count, probs, validate_args=False).sample()
        shape = self._beta._dirichlet._extended_shape(sample_shape)
        concentration = self._beta._dirichlet.concentration.expand(shape)
        probs = torch._sample_dirichlet(concentration, generator=generator).select(-1, 0)
        with torch.no_grad():
            total_count = self.total_count
            if sample_shape:
                total_count = total_count.expand(sample_shape + total_count.shape)
            sample = torch.binomial(total_count.float(), probs, generator=generator)
        return sample

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        n = self.total_count
        k = value
        a = self.concentration1
        b = self.concentration0
        tol = self.approx_log_prob_tol
        return log_binomial(n, k, tol) + log_beta(k + a, n - k + b, tol) - log_beta(a, b, tol)


    @property
    def mean(self):
        return self._beta.mean * self.total_count

    @property
    def variance(self):
        return self._beta.variance * self.total_count * (self.concentration0 + self.concentration1 + self.total_count)

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
        values = torch.arange(1 + total_count, dtype=self.concentration1.dtype, device=self.concentration1.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

def log_beta(x, y, tol=0.0):
    """
    Computes log Beta function.

    When ``tol >= 0.02`` this uses a shifted Stirling's approximation to the
    log Beta function. The approximation adapts Stirling's approximation of the
    log Gamma function::

        lgamma(z) ≈ (z - 1/2) * log(z) - z + log(2 * pi) / 2

    to approximate the log Beta function::

        log_beta(x, y) ≈ ((x-1/2) * log(x) + (y-1/2) * log(y)
                          - (x+y-1/2) * log(x+y) + log(2*pi)/2)

    The approximation additionally improves accuracy near zero by iteratively
    shifting the log Gamma approximation using the recursion::

        lgamma(x) = lgamma(x + 1) - log(x)

    If this recursion is applied ``n`` times, then absolute error is bounded by
    ``error < 0.082 / n < tol``, thus we choose ``n`` based on the user
    provided ``tol``.

    :param torch.Tensor x: A positive tensor.
    :param torch.Tensor y: A positive tensor.
    :param float tol: Bound on maximum absolute error. Defaults to 0.1. For
        very small ``tol``, this function simply defers to :func:`log_beta`.
    :rtype: torch.Tensor
    """
    assert isinstance(tol, (float, int)) and tol >= 0
    if tol < 0.02:
        # At small tolerance it is cheaper to defer to torch.lgamma().
        return x.lgamma() + y.lgamma() - (x + y).lgamma()

    # This bound holds for arbitrary x,y. We could do better with large x,y.
    shift = int(math.ceil(0.082 / tol))

    xy = x + y
    factors = []
    for _ in range(shift):
        factors.append(xy / (x * y))
        x = x + 1
        y = y + 1
        xy = xy + 1

    log_factor = functools.reduce(operator.mul, factors).log()

    return (
        log_factor
        + (x - 0.5) * x.log()
        + (y - 0.5) * y.log()
        - (xy - 0.5) * xy.log()
        + (math.log(2 * math.pi) / 2 - shift)
    )

@torch.no_grad()
def log_binomial(n, k, tol=0.0):
    """
    Computes log binomial coefficient.

    When ``tol >= 0.02`` this uses a shifted Stirling's approximation to the
    log Beta function via :func:`log_beta`.

    :param torch.Tensor n: A nonnegative integer tensor.
    :param torch.Tensor k: An integer tensor ranging in ``[0, n]``.
    :rtype: torch.Tensor
    """
    assert isinstance(tol, (float, int)) and tol >= 0
    n_plus_1 = n + 1
    if tol < 0.02:
        # At small tolerance it is cheaper to defer to torch.lgamma().
        return n_plus_1.lgamma() - (k + 1).lgamma() - (n_plus_1 - k).lgamma()

    return -n_plus_1.log() - log_beta(k + 1, n_plus_1 - k, tol=tol)
