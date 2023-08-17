from typing import List, Tuple, Union

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from irwg.data.toy import ToyDataset
from irwg.data.mnist_gmm import MNIST_GMM, MNIST_GMM_customtest
from irwg.utils.mog_utils import (batched_fit_mogs_sparse,
                                  batched_per_datapoint_fit_mogs_sparse,
                                  compute_conditional_mog_parameters,
                                  compute_kl_div_for_sparse_mogs,
                                  compute_kl_div_for_sparse_mogs_perdatapoint,
                                  sample_sparse_mog)
from irwg.utils.test_step_base import TestBase


class TestMoG(TestBase):
    """
    Computes reference conditional MoG KL

    Args:

    """
    def __init__(self,
                 num_samples: int,
                 num_em_iterations: int,
                 num_kl_samples: int,
                 estimate_jsd: bool,
                 use_solver: bool = False,
                 use_batched_per_datapoint_computation: bool = False,
                 em_init_params_to_true_joint: bool = False,
                 em_init_params_to_true_cond: bool = False,
                ):
        super().__init__()
        self.save_hyperparameters()

    # def set_model(self, model: pl.LightningModule):
    #     self.model = model

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def set_mog_params_from_datamodule(self):
        assert isinstance(self.datamodule.test_data_core, (ToyDataset, MNIST_GMM, MNIST_GMM_customtest))
        self.true_comp_probs = torch.tensor(self.datamodule.test_data_core.data_file['comp_probs'])
        self.true_means = torch.tensor(self.datamodule.test_data_core.data_file['means'])
        self.true_covs = torch.tensor(self.datamodule.test_data_core.data_file['covs'])

    def on_test_epoch_start(self):
        self.set_mog_params_from_datamodule()

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        X, M = batch[:2]

        if isinstance(self.datamodule.test_data_core, MNIST_GMM_customtest):
            assert torch.allclose(X.isnan(), ~M)
            X[X.isnan()] = 0.

        with torch.inference_mode():
            true_cond_comp_log_probs, true_cond_means, true_cond_covs = \
                compute_conditional_mog_parameters(X, M,
                                                   torch.log(self.true_comp_probs),
                                                   self.true_means,
                                                   self.true_covs)

            X_imp_true = sample_sparse_mog(num_samples=self.hparams.num_samples,
                                           M=M,
                                           comp_log_probs=true_cond_comp_log_probs,
                                           cond_means=true_cond_means,
                                           cond_covs=true_cond_covs)

            if self.hparams.use_batched_per_datapoint_computation:
                comp_log_probs, means, covs = batched_per_datapoint_fit_mogs_sparse(X_imp_true, M,
                                        num_components=true_cond_comp_log_probs.shape[-1],
                                        num_iterations=self.hparams.num_em_iterations,
                                        init_params_to_true_joint=self.hparams.em_init_params_to_true_joint,
                                        init_params_to_true_cond=self.hparams.em_init_params_to_true_cond,
                                        true_comp_log_probs=torch.log(self.true_comp_probs),
                                        true_means=self.true_means,
                                        true_covs=self.true_covs,
                                        true_cond_comp_log_probs=torch.log(true_cond_comp_log_probs),
                                        true_cond_means=true_cond_means,
                                        true_cond_covs=true_cond_covs,
                                        )
            else:
                comp_log_probs, means, covs = batched_fit_mogs_sparse(X_imp_true, M,
                                        num_components=true_cond_comp_log_probs.shape[-1],
                                        num_iterations=self.hparams.num_em_iterations,
                                        use_solver=self.hparams.use_solver,
                                        init_params_to_true_joint=self.hparams.em_init_params_to_true_joint,
                                        init_params_to_true_cond=self.hparams.em_init_params_to_true_cond,
                                        true_comp_log_probs=torch.log(self.true_comp_probs),
                                        true_means=self.true_means,
                                        true_covs=self.true_covs,
                                        true_cond_comp_log_probs=torch.log(true_cond_comp_log_probs),
                                        true_cond_means=true_cond_means,
                                        true_cond_covs=true_cond_covs,
                                        )

            if self.hparams.use_batched_per_datapoint_computation:
                kldivs1, jsd_term1 = compute_kl_div_for_sparse_mogs_perdatapoint(params1={
                                                            'comp_log_probs': true_cond_comp_log_probs,
                                                            'means': true_cond_means,
                                                            'covs': true_cond_covs,
                                                            },
                                                        params2={
                                                            'comp_log_probs': comp_log_probs,
                                                            'means': means,
                                                            'covs': covs,
                                                            },
                                                        M=M,
                                                        N=self.hparams.num_kl_samples,
                                                        return_jsd_midpoint=self.hparams.estimate_jsd)
                kldivs2, jsd_term2 = compute_kl_div_for_sparse_mogs_perdatapoint(params1={
                                                            'comp_log_probs': comp_log_probs,
                                                            'means': means,
                                                            'covs': covs,
                                                            },
                                                        params2={
                                                            'comp_log_probs': true_cond_comp_log_probs,
                                                            'means': true_cond_means,
                                                            'covs': true_cond_covs,
                                                            },
                                                        M=M,
                                                        N=self.hparams.num_kl_samples,
                                                        return_jsd_midpoint=self.hparams.estimate_jsd)
            else:
                kldivs1, jsd_term1 = compute_kl_div_for_sparse_mogs(params1={
                                                            'comp_log_probs': true_cond_comp_log_probs,
                                                            'means': true_cond_means,
                                                            'covs': true_cond_covs,
                                                            },
                                                        params2={
                                                            'comp_log_probs': comp_log_probs,
                                                            'means': means,
                                                            'covs': covs,
                                                            },
                                                        M=M,
                                                        N=self.hparams.num_kl_samples,
                                                        return_jsd_midpoint=self.hparams.estimate_jsd,
                                                        use_solver=self.hparams.use_solver,)
                kldivs2, jsd_term2 = compute_kl_div_for_sparse_mogs(params1={
                                                            'comp_log_probs': comp_log_probs,
                                                            'means': means,
                                                            'covs': covs,
                                                            },
                                                        params2={
                                                            'comp_log_probs': true_cond_comp_log_probs,
                                                            'means': true_cond_means,
                                                            'covs': true_cond_covs,
                                                            },
                                                        M=M,
                                                        N=self.hparams.num_kl_samples,
                                                        return_jsd_midpoint=self.hparams.estimate_jsd,
                                                        use_solver=self.hparams.use_solver,)

            kldiv1_mean = kldivs1.mean()
            kldiv1_median = kldivs1.median()
            kldiv2_mean = kldivs2.mean()
            kldiv2_median = kldivs2.median()

            skldivs = 0.5*(kldivs1 + kldivs2)
            skldiv_mean = skldivs.mean()
            skldiv_median = skldivs.median()

            jsd = 0.5*(jsd_term1 + jsd_term2)
            jsd_mean = jsd.mean()
            jsd_median = jsd.median()

            self.logger.experiment.add_scalar('mog_kldiv_mean/test', kldiv1_mean, 0)
            self.logger.experiment.add_scalar('mog_kldiv_median/test', kldiv1_median, 0)
            self.logger.experiment.add_scalar('mog_kldiv_rev_mean/test', kldiv2_mean, 0)
            self.logger.experiment.add_scalar('mog_kldiv_rev_median/test', kldiv2_median, 0)
            self.logger.experiment.add_scalar('mog_skldiv_mean/test', skldiv_mean, 0)
            self.logger.experiment.add_scalar('mog_skldiv_median/test', skldiv_median, 0)
            self.logger.experiment.add_scalar('mog_jsd_mean/test', jsd_mean, 0)
            self.logger.experiment.add_scalar('mog_jsd_median/test', jsd_median, 0)
