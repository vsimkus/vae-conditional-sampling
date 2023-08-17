import math
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from irwg.data.toy import ToyDataset
from irwg.utils.mog_utils import (batched_fit_mogs_sparse,
                                  compute_conditional_mog_parameters, compute_kl_div_for_sparse_mogs,
                                  sample_sparse_mog)
from irwg.utils.test_step_iwelbo import compute_iwelbo
from irwg.utils.test_step_base import TestBase


class TestMoGVAEKL(TestBase):
    """
    Computes reference KL between ground truth MoG and a fitted VAE

    Args:

    """
    def __init__(self,
                 num_samples: int,
                 num_iwelbo_importance_samples: int,
                 iwelbo_batchsize: int,
                ):
        super().__init__()
        self.save_hyperparameters()

    def set_model(self, model: pl.LightningModule):
        self.model = model

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def set_mog_params_from_datamodule(self):
        assert isinstance(self.datamodule.test_data_core, ToyDataset)
        self.true_comp_probs = torch.tensor(self.datamodule.test_data_core.data_file['comp_probs'])
        self.true_means = torch.tensor(self.datamodule.test_data_core.data_file['means'])
        self.true_covs = torch.tensor(self.datamodule.test_data_core.data_file['covs'])

    def on_test_epoch_start(self):
        self.set_mog_params_from_datamodule()

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        # NOTE: true data is ignored.
        # X, M = batch[:2]

        assert batch_idx < 1

        with torch.inference_mode():
            mix = torch.distributions.Categorical(probs=self.true_comp_probs[0].to(self.model.device))
            multi_norms = torch.distributions.MultivariateNormal(loc=self.true_means.to(self.model.device), covariance_matrix=self.true_covs.to(self.model.device))
            comp = torch.distributions.Independent(multi_norms, 0)
            true_mog = torch.distributions.MixtureSameFamily(mix, comp)

            # Use samples from the true MoG model
            X = true_mog.sample(sample_shape=(self.hparams.num_samples,)).float()
            M = torch.ones_like(X)

            mog_logprob = true_mog.log_prob(X)

            vae_iwelbo = self.batched_iwelbo_compute(X, M, batchsize=self.hparams.iwelbo_batchsize)

            forward_kl = mog_logprob - vae_iwelbo
            forward_kl_avg = forward_kl.mean()

            # Use samples from the VAE

            X = self.model.sample(self.hparams.num_samples)
            mog_logprob = true_mog.log_prob(X)
            # NOTE: due to amortisation gap this can sometimes be off for samples unseed during training.
            vae_iwelbo = self.batched_iwelbo_compute(X, M, batchsize=self.hparams.iwelbo_batchsize)

            reverse_kl = vae_iwelbo - mog_logprob
            reverse_kl_avg = reverse_kl.mean()

            # Symmetric KL

            symmetric_kl = 0.5*(forward_kl + reverse_kl)
            symmetric_kl_avg = symmetric_kl.mean()


            self.logger.experiment.add_scalar('mog_vae_forward_kl/test', forward_kl_avg, 0)
            self.logger.experiment.add_scalar('mog_vae_reverse_kl/test', reverse_kl_avg, 0)
            self.logger.experiment.add_scalar('mog_vae_symmetric_kl/test', symmetric_kl_avg, 0)

    def batched_iwelbo_compute(self, X, M, batchsize):
        iwelbo = torch.empty(X.shape[0], device=X.device)
        for t in tqdm(range(math.ceil(X.shape[0] / batchsize)), desc='IWELBO estimator'):
            X_t = X[t*batchsize:min((t+1)*batchsize, X.shape[0])]
            M_t = M[t*batchsize:min((t+1)*batchsize, X.shape[0])]

            vae_iwelbo = compute_iwelbo(self.model, X_t, M_t,
                                        num_importance_samples=self.hparams.num_iwelbo_importance_samples,
                                        return_weights=False)

            iwelbo[t*batchsize:min((t+1)*batchsize, X.shape[0])] = vae_iwelbo

        return iwelbo
