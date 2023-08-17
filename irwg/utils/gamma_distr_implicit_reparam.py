import torch

class GammaImplicitReparam(torch.distributions.Gamma):
    def cdf(self, x):
        return torch.special.gammainc(self.concentration, self.rate*x)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape).detach()

        log_prob_x = self.log_prob(x)

        # This will be differentiated.
        cdf = self.cdf(x)

        surrogate_x = -cdf*torch.exp(-log_prob_x.detach())

        # Replace gradients of x with gradients of surrogate_x, but keep the value.
        return x + (surrogate_x - surrogate_x.detach())

class BetaImplicitReparam(torch.distributions.Beta):
    def rsample(self, sample_shape=torch.Size()):
        gamma1 = GammaImplicitReparam(self.concentration1, 1.)
        gamma2 = GammaImplicitReparam(self.concentration0, 1.)

        gamma1_samples = gamma1.rsample(sample_shape)
        gamma2_samples = gamma2.rsample(sample_shape)

        beta = gamma1_samples / (gamma1_samples + gamma2_samples)

        return beta
