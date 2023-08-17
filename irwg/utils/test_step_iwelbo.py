from typing import Tuple, Union, List

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from irwg.utils.weights import compute_log_unnormalised_importance_weights
from irwg.utils.test_step_base import TestBase

class TestIWELBO(TestBase):
    """
    Computes IWELBO for a trained model

    Args:
        num_importance_samples:         The number of importance samples to estimate IWELBO.
    """
    def __init__(self,
                 num_importance_samples: int
                ):
        super().__init__()
        self.save_hyperparameters()

    def set_model(self, model: pl.LightningModule):
        self.model = model

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        X, M = batch[:2]

        with torch.inference_mode():
            iwelbo, log_weights = compute_iwelbo(self.model, X, M, num_importance_samples=self.hparams.num_importance_samples, return_weights=True)

            avg_iwelbo = torch.mean(iwelbo, dim=0)

            avg_elbo = log_weights.mean()

            avg_kl = avg_iwelbo - avg_elbo

            self.log('iwelbo/test', avg_iwelbo, on_epoch=True, prog_bar=True, logger=True)
            self.log('elbo/test', avg_elbo, on_epoch=True, prog_bar=True, logger=True)
            self.log('kl/test', avg_kl, on_epoch=True, prog_bar=True, logger=True)


def compute_iwelbo(model, X, M, num_importance_samples, return_weights=False):
    # Create latent distribution and sample
    var_latent_params = model.predict_var_latent_params(X, M)
    var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
    Z = var_latent_distr.rsample(sample_shape=(num_importance_samples,))
    # Shape = (i b d)

    # Create prior distribution
    # NOTE: assuming standard prior
    prior_distr = torch.distributions.Normal(loc=0., scale=1.)

    # Create generator distribution
    generator_params = model.generator_network(Z)
    generator_distr = model.create_distribution(generator_params, model.hparams.generator_distribution)

    log_weights = compute_log_unnormalised_importance_weights(X, M, Z,
                                                            var_latent_distr=var_latent_distr,
                                                            prior_distr=prior_distr,
                                                            generator_distr=generator_distr)


    iwelbo = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(log_weights.shape[0], device=log_weights.device))

    if return_weights:
        return iwelbo, log_weights
    else:
        return iwelbo

def compute_log_p_xm_given_xo_with_iwelbo(model, X, M, num_importance_samples):
    # Create latent distribution and sample
    var_latent_params = model.predict_var_latent_params(X, M)
    var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
    Z = var_latent_distr.rsample(sample_shape=(num_importance_samples,))
    # Shape = (i b d)

    # Create prior distribution
    # NOTE: assuming standard prior
    prior_distr = torch.distributions.Normal(loc=0., scale=1.)

    # Create generator distribution
    generator_params = model.generator_network(Z)
    generator_distr = model.create_distribution(generator_params, model.hparams.generator_distribution)

    log_weights = compute_log_unnormalised_importance_weights(X, M, Z,
                                                            var_latent_distr=var_latent_distr,
                                                            prior_distr=prior_distr,
                                                            generator_distr=generator_distr)

    # log p(xo)
    iwelbo = torch.logsumexp(log_weights, dim=0) - torch.log(torch.tensor(log_weights.shape[0], device=log_weights.device))

    log_weights_joint = compute_log_unnormalised_importance_weights(X, torch.ones_like(M), Z,
                                                            var_latent_distr=var_latent_distr,
                                                            prior_distr=prior_distr,
                                                            generator_distr=generator_distr)

    # log p(xm, xo)
    iwelbo_joint = torch.logsumexp(log_weights_joint, dim=0) - torch.log(torch.tensor(log_weights.shape[0], device=log_weights.device))

    log_p_xm_given_xo_iwelbo = iwelbo_joint - iwelbo

    return log_p_xm_given_xo_iwelbo
