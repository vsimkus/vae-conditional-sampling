from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import asnumpy, rearrange, reduce
from pytorch_lightning.utilities.types import STEP_OUTPUT

from irwg.models.neural_nets import ResidualFCNetwork
from irwg.utils.disc_mix_logistic import DiscMixLogistic
from irwg.utils.gamma_distr_implicit_reparam import BetaImplicitReparam
from irwg.utils.logit_normal import LogitNormal
from irwg.utils.softplus_inverse import torch_softplus_inverse

EPSILON=1e-8


class DISTRIBUTION(Enum):
    normal = auto()
    bern = auto()
    normal_with_eps = auto()
    studentt_with_eps22 = auto()
    studentt = auto()
    disc_mix10_logistic_8bit = auto()
    beta = auto()
    logitnormal_with_eps = auto()
    beta_implrep = auto()

def get_beta_schedule(scheduler_type, scheduler_conf_string):
    def linear_monotonic(low, high, begin, end):
        def f(x):
            if x < begin:
                return low
            if x > end:
                return high
            return low + (high - low) * (x - begin) / (end - begin)
        return f

    def linear_cyclic(low, high, begin, period, end):
        def f(x):
            if x < begin:
                return low
            if x > end:
                return high
            x = x-begin
            x = x % period
            return low + (high - low) * x / period
        return f

    if scheduler_type is None:
        return lambda x: 1.0
    elif scheduler_type == "linear_monotonic":
        low, high, begin, end = scheduler_conf_string.split(',')
        low, high = float(low), float(high)
        begin, end = int(begin), int(end)
        assert low >= 0. and high <= 1.0
        return linear_monotonic(low, high, begin, end)
    elif scheduler_type == "linear_cyclic":
        low, high, begin, period, end = scheduler_conf_string.split(',')
        low, high = float(low), float(high)
        begin, end = int(begin), int(end)
        period = int(period)
        assert low >= 0. and high <= 1.0
        return linear_cyclic(low, high, begin, period, end)
    else:
        raise NotImplementedError(f"Unknown scheduler type {scheduler_type}")


class VAE(pl.LightningModule):
    """
    A VAE model with missing data.

    Args:
        generator_network:          The neural network of the generator.
        generator_distribution:     The distribution of the generator.
        var_latent_network:         The neural network of the variational latent network.
        var_latent_distribution:    The distribution of the variational latents.
        encoder_use_mis_mask:       if true appends the missingness mask to the encoder.
        var_latent_STL:             if true uses the gradients from "Sticking the Landing" by Roeder et al. 2017
        num_latent_samples:         The number of samples used in Monte Carlo averaging of the ELBO
        kl_analytic:                If true the KL term is computed analytically.

        lr_generator:               learning rate of the generator model
        amsgrad_generator:          if true use AMSGrad version of Adam for the generator model

        lr_latent:                  learning rate of the latent model
        amsgrad_latent:             if true use AMSGrad version of Adam for the latent model

        use_lr_scheduler:           if true uses Cosine learning rate scheduler
        max_scheduler_steps:        maximum number of steps in the lr scheduler

        use_uniform_prior:          Uses union prior [0,1]

        kl_beta_scheduler:          The scheduler for the beta parameter in the KL term (default: None)
        kl_beta_scheduler_conf_string:      The configuration string for the scheduler (default: None)
    """

    def __init__(self,
                 generator_network: nn.Module,
                 generator_distribution: DISTRIBUTION,
                 var_latent_network: nn.Module,
                 var_latent_distribution: DISTRIBUTION,
                 num_latent_samples: int,
                 lr_generator: float,
                 lr_latent: float,
                 *args,
                 amsgrad_generator: bool = False,
                 amsgrad_latent: bool = False,
                 use_lr_scheduler: bool = False,
                 max_scheduler_steps: int = -1,
                 encoder_use_mis_mask: bool = False,
                 var_latent_STL: bool = False,
                 kl_analytic: bool = True,
                 use_uniform_prior: bool = False,

                 kl_beta_scheduler: Optional[str] = None,
                 kl_beta_scheduler_conf_string: Optional[str] = None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        assert (not (var_latent_STL and kl_analytic)),\
            '"Sticking the Landing" cannot be used with analytical KL.'

        self.generator_network = generator_network
        self.var_latent_network = var_latent_network

    def freeze_generator_params(self):
        self.generator_network.requires_grad_(False)
        # NOTE: Should fix the prior model if one is used.

    def reset_encoder_params(self: nn.Module) -> None:
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.var_latent_network.apply(fn=weight_reset)

    def partial_encoder_params_reset(self, n):
        if isinstance(self.var_latent_network, ResidualFCNetwork):
            self.var_latent_network.reset_first_n_blocks(n)
        else:
            raise NotImplementedError()

    def get_prior(self):
        if self.hparams.use_uniform_prior:
            prior_distr = torch.distributions.Uniform(low=torch.tensor(0.).to(self.device), high=torch.tensor(1.+1e-8).to(self.device))
        else:
            prior_distr = torch.distributions.Normal(loc=0., scale=1.)

        return prior_distr

    def sample(self, num_samples, rng=None, *, return_latents=False):
        if self.hparams.use_uniform_prior:
            latents = torch.rand(num_samples, self.generator_network.input_dim, generator=rng, device=self.device)*(1.+1e-8)
        else:
            latents = torch.randn(num_samples, self.generator_network.input_dim, generator=rng, device=self.device)
        generator_params = self.generator_network(latents)
        generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)
        X = generator_distr.sample()

        if return_latents:
            return X, latents
        else:
            return X

    def predict_var_latent_params(self, X, M):
        if self.hparams.encoder_use_mis_mask:
            XM = rearrange([X, M], 'f ... d -> ... (f d)')
            params = self.var_latent_network(XM)
            # NOTE: If necessary here can also add network-specific code
        else:
            params = self.var_latent_network(X)

        return params

    def split_params(self, params: torch.Tensor, num_params: int) -> torch.Tensor:
        """
        Rearranges parameter tensor such that each parameter-group is in its own "row".
        E.g. for gaussian parameters (b (means+logvars)) -> ((means logvars) b d)
        """
        return rearrange(params, '... (params d) -> params ... d', params=num_params)

    def join_params(self, params: List[torch.Tensor]) -> torch.Tensor:
        """
        Undo split_params
        """
        return rearrange(params, 'params ... d -> ... (params d)')

    def create_distribution(self, params: torch.Tensor, distribution: DISTRIBUTION,
                            *, validate_args: bool = None) -> torch.distributions.Distribution:
        """
        Creates a torch.Distribution object from the parameters
        """
        if distribution is DISTRIBUTION.normal:
            # loc, logvar = self.split_params(params, num_params=2)
            # scale = torch.exp(logvar*0.5)
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw)
            distr = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.normal_with_eps:
            # loc, logvar = self.split_params(params, num_params=2)
            # scale = torch.exp(logvar*0.5)
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON
            distr = torch.distributions.Normal(loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.studentt:
            df_raw, loc, scale_raw = self.split_params(params, num_params=3)
            df = torch.nn.functional.softplus(df_raw)
            scale = torch.nn.functional.softplus(scale_raw)
            distr = torch.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.studentt_with_eps22:
            df_raw, loc, scale_raw = self.split_params(params, num_params=3)
            df = torch.nn.functional.softplus(df_raw)
            scale = torch.nn.functional.softplus(scale_raw) + 2**-2
            distr = torch.distributions.StudentT(df=df, loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.disc_mix10_logistic_8bit:
            num_mix = 10
            rgb = 3
            logistic_p = 3
            total_p = num_mix + rgb*logistic_p*num_mix
            params = rearrange(params, '... (p d) -> ... p d', p=total_p)
            distr = DiscMixLogistic(params, num_mix=num_mix, num_bits=8, validate_args=validate_args)
        elif distribution is DISTRIBUTION.bern:
            distr = torch.distributions.Bernoulli(logits=params, validate_args=validate_args)
        elif distribution is DISTRIBUTION.beta:
            alpha, beta = self.split_params(params, num_params=2)
            alpha = torch.nn.functional.softplus(alpha)
            beta = torch.nn.functional.softplus(beta)
            distr = torch.distributions.Beta(concentration1=alpha, concentration0=beta, validate_args=validate_args)
        elif distribution is DISTRIBUTION.logitnormal_with_eps:
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON
            distr = LogitNormal(loc=loc, scale=scale, validate_args=validate_args)
        elif distribution is DISTRIBUTION.beta_implrep:
            alpha, beta = self.split_params(params, num_params=2)
            alpha = torch.nn.functional.softplus(alpha)
            beta = torch.nn.functional.softplus(beta)
            distr = BetaImplicitReparam(concentration1=alpha, concentration0=beta, validate_args=validate_args)
        else:
            raise NotImplementedError(f'Method not implemented for {distribution = }')

        return distr

    def distribution_param_tempering(self, params: torch.Tensor, distribution: DISTRIBUTION, *, temperature: float) -> torch.Tensor:
        if distribution is DISTRIBUTION.normal:
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw)
            scale = scale * temperature
            scale_raw = torch_softplus_inverse(scale)
            params = self.join_params([loc, scale_raw])
        elif distribution is DISTRIBUTION.normal_with_eps:
            loc, scale_raw = self.split_params(params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON
            scale = (scale * temperature) - EPSILON
            scale_raw = torch_softplus_inverse(scale)
            params = self.join_params([loc, scale_raw])
        elif distribution is DISTRIBUTION.studentt:
            df_raw, loc, scale_raw = self.split_params(params, num_params=3)
            scale = torch.nn.functional.softplus(scale_raw)
            scale = scale * temperature
            scale_raw = torch_softplus_inverse(scale)
            params = self.join_params([df_raw, loc, scale_raw])
        elif distribution is DISTRIBUTION.studentt_with_eps22:
            df_raw, loc, scale_raw = self.split_params(params, num_params=3)
            scale = torch.nn.functional.softplus(scale_raw) + 2**-2
            scale = (scale * temperature) - 2**-2
            scale_raw = torch_softplus_inverse(scale)
            params = self.join_params([df_raw, loc, scale_raw])
        elif distribution is DISTRIBUTION.disc_mix10_logistic_8bit:
            raise NotImplementedError()
        elif distribution is DISTRIBUTION.bern:
            raise NotImplementedError()
        elif distribution is DISTRIBUTION.beta:
            alpha_raw, beta_raw = self.split_params(params, num_params=2)
            alpha = torch.nn.functional.softplus(alpha_raw)
            beta = torch.nn.functional.softplus(beta_raw)
            alpha_raw = torch_softplus_inverse(alpha * temperature)
            beta_raw = torch_softplus_inverse(beta * temperature)
            params = self.join_params([alpha_raw, beta_raw])
        elif distribution is DISTRIBUTION.logitnormal_with_eps:
            raise NotImplementedError()
        elif distribution is DISTRIBUTION.beta_implrep:
            alpha_raw, beta_raw = self.split_params(params, num_params=2)
            alpha = torch.nn.functional.softplus(alpha_raw)
            beta = torch.nn.functional.softplus(beta_raw)
            alpha_raw = torch_softplus_inverse(alpha * temperature)
            beta_raw = torch_softplus_inverse(beta * temperature)
            params = self.join_params([alpha_raw, beta_raw])
        else:
            raise NotImplementedError(f'Method not implemented for {distribution = }')

        return params

    def anneal_var_distribution_to_prior_based_on_missingness(self,
            var_params: torch.Tensor, M: torch.Tensor, var_distribution: DISTRIBUTION, *, anneal_type: str = None) -> torch.Tensor:
        # M = M.squeeze(1)

        prior = None
        if self.hparams.use_uniform_prior:
            prior = 'uniform'
        else:
            prior = 'normal'

        if anneal_type == 'linear':
            def anneal(k, M):
                miss_fraction = (M.shape[-1] - M.sum(dim=-1, keepdim=True))/M.shape[-1]
                return 1 * (1-miss_fraction) + k * miss_fraction
        elif anneal_type == 'linear_0.25':
            def anneal(k, M):
                const = 0.25
                miss_fraction = (M.shape[-1] - M.sum(dim=-1, keepdim=True))/M.shape[-1]
                return 1 * (1-miss_fraction) + k*const * miss_fraction
        else:
            raise NotImplementedError(f'Method not implemented for {anneal_type = }')

        if var_distribution is DISTRIBUTION.normal and prior == 'normal':
            loc, scale_raw = self.split_params(var_params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw)

            # Compute scaling factor
            k = 1.0 / scale
            k_m = anneal(k, M)
            scale = scale * k_m
            scale_raw = torch_softplus_inverse(scale)

            # Compute mean
            k_m = anneal(0, M)
            loc = loc * k_m

            var_params = self.join_params([loc, scale_raw])
        elif var_distribution is DISTRIBUTION.normal_with_eps and prior == 'normal':
            loc, scale_raw = self.split_params(var_params, num_params=2)
            scale = torch.nn.functional.softplus(scale_raw) + EPSILON

            # Compute scaling factor
            k = 1.0 / scale
            k_m = anneal(k, M)
            scale = (scale * k_m) - EPSILON
            scale_raw = torch_softplus_inverse(scale)

            # Compute mean
            k_m = anneal(0, M)
            loc = loc * k_m

            var_params = self.join_params([loc, scale_raw])
        elif var_distribution is DISTRIBUTION.studentt:
            raise NotImplementedError()
        elif var_distribution is DISTRIBUTION.studentt_with_eps22:
            raise NotImplementedError()
        elif var_distribution is DISTRIBUTION.disc_mix10_logistic_8bit:
            raise NotImplementedError()
        elif var_distribution is DISTRIBUTION.bern:
            raise NotImplementedError()
        elif var_distribution is DISTRIBUTION.beta and prior == 'uniform':
            alpha_raw, beta_raw = self.split_params(var_params, num_params=2)
            alpha = torch.nn.functional.softplus(alpha_raw)
            beta = torch.nn.functional.softplus(beta_raw)

            # Compute scaling factor
            k_alpha = 1.0 / alpha
            k_beta = 1.0 / beta
            k_alpha_m = anneal(k_alpha, M)
            k_beta_m = anneal(k_beta, M)

            alpha_raw = torch_softplus_inverse(alpha * k_alpha_m)
            beta_raw = torch_softplus_inverse(beta * k_beta_m)
            var_params = self.join_params([alpha_raw, beta_raw])
        elif var_distribution is DISTRIBUTION.logitnormal_with_eps:
            raise NotImplementedError()
        elif var_distribution is DISTRIBUTION.beta_implrep and prior == 'uniform':
            alpha_raw, beta_raw = self.split_params(var_params, num_params=2)
            alpha = torch.nn.functional.softplus(alpha_raw)
            beta = torch.nn.functional.softplus(beta_raw)

            # Compute scaling factor
            k_alpha = 1.0 / alpha
            k_beta = 1.0 / beta
            k_alpha_m = anneal(k_alpha, M)
            k_beta_m = anneal(k_beta, M)

            alpha_raw = torch_softplus_inverse(alpha * k_alpha_m)
            beta_raw = torch_softplus_inverse(beta * k_beta_m)
            var_params = self.join_params([alpha_raw, beta_raw])
        else:
            raise NotImplementedError(f'Method not implemented for {var_distribution = } and {prior = }')

        return var_params

    def compute_elbo_w_analytic_kl(self,
                     X: torch.Tensor,
                     M: torch.Tensor,
                     var_latent_distr: torch.distributions.Distribution,
                     prior_distr: torch.distributions.Distribution,
                     generator_distr: torch.distributions.Distribution,
                     *,
                     kl_beta: float = 1.0) -> torch.Tensor:
        # Compute cross-entropy term of observed data
        log_prob = generator_distr.log_prob(X)*M
        log_prob = reduce(log_prob, 'z b ... d -> z b ...', 'sum')
        log_prob = reduce(log_prob, 'z b ... -> b ...', 'mean')

        # Compute analytical -KL(q(z|x) || p(z)) term
        KL_neg = -torch.distributions.kl_divergence(var_latent_distr, prior_distr)
        KL_neg = reduce(KL_neg, 'b ... d -> b ...', 'sum')

        # Compute per-data-point elbos
        elbo = log_prob + KL_neg
        loss = -(log_prob + kl_beta*KL_neg)
        return elbo, loss

    def compute_elbo(self,
                    X: torch.Tensor,
                    M: torch.Tensor,
                    Z: torch.Tensor,
                    var_latent_distr: torch.distributions.Distribution,
                    var_latent_distr_detached: torch.distributions.Distribution,
                    prior_distr: torch.distributions.Distribution,
                    generator_distr: torch.distributions.Distribution,
                    *,
                    kl_beta: float = 1.0) -> torch.Tensor:
        # Compute cross-entropy term of observed data
        generator_log_prob = generator_distr.log_prob(X)*M
        generator_log_prob = reduce(generator_log_prob, 'z b ... d -> z b ...', 'sum')
        generator_log_prob = reduce(generator_log_prob, 'z b ... -> b ...', 'mean')

        # Compute prior latent probability
        prior_logprob = prior_distr.log_prob(Z)
        prior_logprob = reduce(prior_logprob, 'z b ... d -> z b ...', 'sum')

        # Compute the log-prob of samples under the latent distribution
        if self.hparams.var_latent_STL:
            # NOTE: alternatively could use this https://github.com/pyro-ppl/pyro/pull/2599/
            latent_logprob = var_latent_distr_detached.log_prob(Z)
        else:
            latent_logprob = var_latent_distr.log_prob(Z)
        latent_logprob = reduce(latent_logprob, 'z b ... d -> z b ...', 'sum')

        # Compute per-data-point elbo
        total = generator_log_prob + prior_logprob - latent_logprob
        elbos = reduce(total, 'z b ... -> b ...', 'mean')

        loss = -(generator_log_prob + kl_beta*(prior_logprob - latent_logprob))
        loss = reduce(loss, 'z b ... -> b ...', 'mean')

        return elbos, loss

    def compute_var_distr(self, X, M):
        var_latent_params = self.predict_var_latent_params(X, M)
        return self.create_distribution(var_latent_params, self.hparams.var_latent_distribution)

    def estimate_elbo(self, batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        X, M = batch[:2]

        # Create latent distribution and sample
        var_latent_params = self.predict_var_latent_params(X, M)
        var_latent_distr = self.create_distribution(var_latent_params, self.hparams.var_latent_distribution)
        var_latent_distr_detached = None
        if self.hparams.var_latent_STL:
            var_latent_distr_detached = self.create_distribution(var_latent_params.detach(), self.hparams.var_latent_distribution)
        Z = var_latent_distr.rsample(sample_shape=(self.hparams.num_latent_samples,))

        # Create prior distribution
        prior_distr = self.get_prior()

        # Create generator distribution
        generator_params = self.generator_network(Z)
        generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)

        kl_beta_scheduler = get_beta_schedule(self.hparams.kl_beta_scheduler, self.hparams.kl_beta_scheduler_conf_string)
        kl_beta = kl_beta_scheduler(self.trainer.global_step)

        # Compute per data-point elbo
        if self.hparams.kl_analytic:
            elbo, loss = self.compute_elbo_w_analytic_kl(X, M, var_latent_distr, prior_distr, generator_distr, kl_beta=kl_beta)
        else:
            elbo, loss = self.compute_elbo(X, M, Z, var_latent_distr, var_latent_distr_detached, prior_distr, generator_distr, kl_beta=kl_beta)

        # Averaged elbo
        elbo = reduce(elbo, 'b -> ', 'mean')
        loss = reduce(loss, 'b -> ', 'mean')

        return elbo, loss

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        elbo, loss = self.estimate_elbo(batch)

        # logs metrics
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            elbo, loss = self.estimate_elbo(batch)

        # logs metrics
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.generator_network.parameters(),
             'amsgrad': self.hparams.amsgrad_generator,
             'lr': self.hparams.lr_generator},
            {'params': self.var_latent_network.parameters(),
             'amsgrad': self.hparams.amsgrad_latent,
             'lr': self.hparams.lr_latent}])

        opts = {
            'optimizer': optimizer
        }

        if self.hparams.use_lr_scheduler:
            max_steps = self.hparams.max_scheduler_steps if self.hparams.max_scheduler_steps != -1 else self.trainer.estimated_stepping_batches
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, eta_min=0, last_epoch=-1)

            opts['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1,
            }

        return opts


class VAE_trained_by_sampling_prior(VAE):
    def estimate_avg_log_prob_by_mc(self, batch):
        prior = self.get_prior()
        Z = prior.sample(sample_shape=(self.hparams.num_latent_samples, 1, self.generator_network.input_dim))

        generator_params = self.generator_network(Z)
        generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)

        X, M = batch[:2]
        generator_log_prob = generator_distr.log_prob(X)*M
        generator_log_prob = reduce(generator_log_prob, 'z b ... d -> z b ...', 'sum')

        log_prob = torch.logsumexp(generator_log_prob, dim=0) - torch.log(torch.tensor(self.hparams.num_latent_samples))
        log_prob = reduce(log_prob, 'b -> ', 'mean')

        return log_prob

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        # elbo = self.estimate_elbo(batch)
        # loss = -elbo

        log_prob = self.estimate_avg_log_prob_by_mc(batch)
        loss = -log_prob

        # logs metrics
        self.log('log_prob/train', log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            # elbo = self.estimate_elbo(batch)
            # loss = -elbo

            log_prob = self.estimate_avg_log_prob_by_mc(batch)
            loss = -log_prob

        # logs metrics
        self.log('log_prob/val', log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


class VAE_trained_with_mog_components(VAE):
    def estimate_avg_log_prob_by_mc(self, batch, *, num_targets):
        X, M = batch[:2]
        targets = batch[-2]

        Z = torch.rand((self.hparams.num_latent_samples, len(targets), self.generator_network.input_dim), device=X.device)
        Z = Z / num_targets + (targets).unsqueeze(-1) / num_targets

        generator_params = self.generator_network(Z)
        generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)

        generator_log_prob = generator_distr.log_prob(X)*M
        generator_log_prob = reduce(generator_log_prob, 'z b ... d -> z b ...', 'sum')

        log_prob = torch.logsumexp(generator_log_prob, dim=0) - torch.log(torch.tensor(self.hparams.num_latent_samples))
        log_prob = reduce(log_prob, 'b -> ', 'mean')

        return log_prob

    def compute_loss_for_encoder(self, batch, *, num_targets):
        X, M = batch[:2]
        targets = batch[-2]

        Z = torch.rand((self.hparams.num_latent_samples, len(targets), self.generator_network.input_dim), device=X.device)
        Z = Z / num_targets + (targets).unsqueeze(-1) / num_targets
        Z = torch.clamp(Z, torch.tensor(1e-6, device=X.device), torch.tensor(1-1e-6, device=X.device))

        latent_params = self.var_latent_network(X)
        latent_distr = self.create_distribution(latent_params, self.hparams.var_latent_distribution)

        latent_log_prob = latent_distr.log_prob(Z)

        latent_log_prob = reduce(latent_log_prob, 'z b ... d -> z b ...', 'sum')

        latent_log_prob = reduce(latent_log_prob, 'z b ... -> ...', 'mean')

        return latent_log_prob

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        # elbo = self.estimate_elbo(batch)
        # loss = -elbo

        num_targets = self.trainer.datamodule.orig_train_data.get_num_components()
        log_prob = self.estimate_avg_log_prob_by_mc(batch, num_targets=num_targets)
        latent_log_prob = self.compute_loss_for_encoder(batch, num_targets=num_targets)

        loss = -log_prob - latent_log_prob

        # Not training on elbo yet just logging
        with torch.inference_mode():
            elbo = self.estimate_elbo(batch)

        # logs metrics
        self.log('log_prob/train', log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('latent_log_prob/train', latent_log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                        batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        with torch.inference_mode():
            # elbo = self.estimate_elbo(batch)
            # loss = -elbo

            num_targets = self.trainer.datamodule.orig_train_data.get_num_components()
            log_prob = self.estimate_avg_log_prob_by_mc(batch, num_targets=num_targets)
            latent_log_prob = self.compute_loss_for_encoder(batch, num_targets=num_targets)
            loss = -log_prob - latent_log_prob

            # Not training on elbo yet just logging
            with torch.inference_mode():
                elbo = self.estimate_elbo(batch)

        # logs metrics
        self.log('log_prob/val', log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('latent_log_prob/val', latent_log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

class VAE_encoder_finetuning_on_fowkl(VAE):
    def finetune_encoder_on_fowkl(self, batch):
        # Data should come from the model (x, z) ~ p(x, z) (setup in dataloader)
        X, M = batch[:2]
        Z = batch[-2]

        latent_params = self.var_latent_network(X)
        latent_distr = self.create_distribution(latent_params, self.hparams.var_latent_distribution)
        latent_log_prob = latent_distr.log_prob(Z)

        latent_log_prob = reduce(latent_log_prob, 'b ... d -> b ...', 'sum')
        latent_log_prob = reduce(latent_log_prob, 'b ... -> ...', 'mean')

        return latent_log_prob

    def training_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        # elbo = self.estimate_elbo(batch)
        # loss = -elbo

        latent_log_prob = self.finetune_encoder_on_fowkl(batch)
        loss = -latent_log_prob

        # Not training on elbo just logging
        with torch.inference_mode():
            elbo = self.estimate_elbo(batch)

        # logs metrics
        self.log('latent_log_prob/train', latent_log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('elbo/train', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self,
                      batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                      batch_idx: int) -> STEP_OUTPUT:
        with torch.inference_mode():
            # elbo = self.estimate_elbo(batch)
            # loss = -elbo

            latent_log_prob = self.finetune_encoder_on_fowkl(batch)
            loss = -latent_log_prob

            # Not training on elbo just logging
            elbo = self.estimate_elbo(batch)

        # logs metrics
        self.log('latent_log_prob/val', latent_log_prob, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('elbo/val', elbo, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
