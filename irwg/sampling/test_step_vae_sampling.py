import math
import os.path
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from einops import asnumpy, rearrange, reduce, repeat
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm, trange

from irwg.data.toy import ToyDataset
from irwg.sampling.imputation_metrics import (imputation_bin_accuracy_metric,
                                              imputation_f1_metric,
                                              imputation_inv_f1_metric,
                                              imputation_latent_wass,
                                              imputation_rmse_metric,
                                              imputation_mae_metric,
                                              imputation_ssim_metric,
                                              mmd_linear_time_metric,
                                              mmd_metric)
from irwg.sampling.sampler_KL_metric import compute_KL_div
from irwg.sampling.vae_imputation import (
    importance_resampling_gibbs_iteration, pseudo_gibbs_iteration, standard_importance_resampling, expand_M_dim)
from irwg.sampling.utils import ImputationHistoryQueue, ImputationHistoryQueue_with_restrictedavailability
from irwg.utils.basic_imputation import imputation_fn
from irwg.utils.mog_utils import (batched_compute_joint_log_probs_sparse_mogs,
                                  batched_fit_mogs_sparse,
                                  compute_conditional_mog_parameters,
                                  compute_kl_div_for_sparse_mogs,
                                  compute_joint_p_c_given_xo_xm,
                                  compute_kldivs_between_categoricals)
from irwg.utils.test_step_base import TestBase
from irwg.utils.test_step_iwelbo import compute_iwelbo


def init_map(vae, X, M, num_particles, sgd_steps, sgd_lr, *, use_lr_schedule=False, data_channel_dim=None):
    M = expand_M_dim(M, data_channel_dim=data_channel_dim)
    Z_dim = vae.generator_network.input_dim

    prior = vae.get_prior()
    if data_channel_dim is not None:
        Z = prior.sample((num_particles, *X.shape[:data_channel_dim],Z_dim))
    else:
        Z = prior.sample((num_particles, *X.shape[:-1], Z_dim))
    Z = Z.to(X.device)
    Z.requires_grad_(True)

    optim = torch.optim.SGD([Z], lr=sgd_lr)
    if use_lr_schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim,
                                                           T_max=sgd_steps,
                                                           eta_min=1e-6)

    with torch.enable_grad():
        # Deal with the case when the X has nans for missing values
        X[~repeat(M, 'b 1 ... -> b k ...', k=X.shape[1])] = 0.
        for it in trange(sgd_steps, desc='MAP initialisation using SGD'):
            optim.zero_grad()
            prior_logprob = prior.log_prob(Z).sum(-1)

            gen_params = vae.generator_network(Z)
            gen_distr = vae.create_distribution(gen_params, vae.hparams.generator_distribution)

            log_prob = reduce(gen_distr.log_prob(X)*M, '... d -> ...', 'sum')
            log_prob = prior_logprob + log_prob # p(z, xo)

            loss = -log_prob.mean()

            loss.backward()
            optim.step()
            if use_lr_schedule:
                sched.step()

    Z.requires_grad_(False)
    prior_logprob = prior.log_prob(Z).sum(-1)
    gen_params = vae.generator_network(Z)
    gen_distr = vae.create_distribution(gen_params, vae.hparams.generator_distribution)
    log_prob = reduce(gen_distr.log_prob(X)*M, '... d -> ...', 'sum')
    log_prob = prior_logprob + log_prob # p(z, xo)

    # Find the maximiser
    # idx = torch.argmax(log_prob, dim=0)
    # Z = Z[idx]
    log_prob = rearrange(log_prob, 'z b k -> z (b k)')
    idx = torch.argmax(log_prob, dim=0)
    Z = rearrange(Z, 'z b k ... -> z (b k) ...')
    Z = Z[idx, torch.arange(Z.shape[1])]
    Z = rearrange(Z, '(b k) ... -> b k ...', b=X.shape[0])

    gen_params = vae.generator_network(Z)
    gen_distr = vae.create_distribution(gen_params, vae.hparams.generator_distribution)
    Xm = gen_distr.sample()

    X_new =  X*M + Xm*(~M)

    return X_new


class TestVAESamplerBase(TestBase):
    """
    Samplers for VAE, base class

    Args:
        store_last_n_iterations:        Stores the last N iterations
        store_every_n_iterations:       Stores every N iterations

        imputation_metric_quantiles:    List of quantiles at which to report imputation metrics (in addition to mean+std)
        compute_f1_metric:              Computes F1 imputation score
        compute_inv_f1_metric:          Computes inverted F1 imputation score
        compute_bin_accuracy_metric:    Computes binary imputation accuracy
        compute_rmse_metric:            Computes rmse accuracy
        compute_mae_metric:             Compute mae imputation metric
        compute_ssim_metric:            Computes SSIM accuracy
        compute_latent_wass_metric:     Computes Latent wasserstein distance

        num_copies:                     Number of copies to make of each sample
        imputation_fn:                  Name of the basic imputation function
        save_img_snapshot_in_tb:        Whether to same a random sample of sampler path snapshots in Tensorboard
        snapshot_image_dims:            Image dimensions for Tensorboard snapshot

        estimate_sampler_ess:           Estimate ESS of the sampler sequence (might use a lot of memory)
        sampler_ess_estimator_warmup:   If estimating ESS, how many samples should be thrown out.
        ess_evaluation_batchsize:       If the tensor size for ESS is too large, here can iteratively compute it for batches of data
        store_is_norm_ess:              Stores normalised IS ESS

        estimate_sampler_variance:      Estimates variance of the imputations (all times)

        estimate_mmd_with_model:        Whether to estimate MMD
        mmd_estimator_kernel:           Kernel for MMD
        mmd_estimator_size:             Size of reference data
        mmd_estimator_frequency:        How often estimate MMD
        mmd_avg_of_chains:              If true, estimates MMD for each chain separately and then averages over the chains
        estimate_mmd_for_full_chain:    If true, estimates MMD (linear time) over the full chain length
        mmd_full_chain_linear_estimator_repeats: When estimating full chain mmd with linear-time estimator, how many times to repeat.

        estimate_logprob:               Uses IWELBO to estimate log-probability

        estimate_kl_div_for_full_chain: Estimated kl-div using IWELBO for the full chain
        kl_div_kde_kernel:              KDE kernel to used in KL-div
        kl_div_iwelbo_num_importance_samples: Number of importance samples to use in KL div

        estimate_fid_chain:                   Whether to estimate FID using the inception features
        fid_estimator_size:             Number of reference datapoint for FID

        estimate_kid_chain:                   Whether to estimate KID using the inception features
        kid_estimator_kernel      :                   Whether to estimate KID using the inception features
        kid_full_chain_linear_estimator_repeats: When estimating full chain kid with linear-time estimator, how many times to repeat.

        estimate_mog_kldivs:            Estimate MoGs based on imputations and compare to the true conditional mogs using Kl-div
        num_mog_em_iterations:          Number of EM steps in MoG estimation from imputations
        num_mog_kl_samples:             Number of samples to estimate kl between MoGs
        mog_kldivs_skip_first_num_imps: Number of first imputations to skip
        mog_kldivs_use_every_nth:       Every nth iteration is used for mog estimation
        mog_em_use_solver:              If true uses solver, which is slower but more numerically stable
        estimate_mog_jsd:               If true, estimate JSD too between the MoGs.

        data_channel_dim:               Used to expand the missingness mask on this dimension.
    """
    def __init__(self,
                 store_last_n_iterations: int,
                 store_every_n_iterations: int,

                 imputation_metric_quantiles: List[float],
                 compute_f1_metric: bool,
                 compute_inv_f1_metric: bool,
                 compute_bin_accuracy_metric: bool,
                 compute_rmse_metric: bool,
                 compute_ssim_metric: bool,
                 compute_latent_wass_metric: bool,

                 num_copies: int,
                 imputation_fn: str,

                 store_latents: bool = False,
                 dont_store_imputations: bool = False,

                 sgd_map_init_num_particles: int = None,
                 sgd_map_init_sgd_steps: int = None,
                 sgd_map_init_sgd_lr: float = None,
                 sgd_map_init_use_lr_sched: bool = False,

                 num_sequential_runs: int = 1,

                 save_img_snapshot_in_tb: bool = False,
                 snapshot_image_dims: List[int] = [28, 28],

                 estimate_sampler_ess: bool = False,
                 sampler_ess_estimator_warmup: int = -1,
                 ess_evaluation_batchsize: int = -1,
                 normalise_trajectory_ess_by_is_ess: bool = False,
                 store_is_norm_ess: bool = False,

                 estimate_mmd_with_model: bool = False,
                 mmd_estimator_kernel: str = 'rbf',
                 mmd_estimator_size: int = None,
                 mmd_estimator_frequency: int = None,
                 mmd_avg_of_chains: bool = False,
                 estimate_mmd_for_full_chain: bool = False,
                 estimate_mmd_target_num_repeats: int = 1,
                 mmd_full_chain_linear_estimator_repeats: int = 1,

                 estimate_iwelbo: bool = False,
                 iwelbo_estimator_frequency: int = None,
                 iwelbo_num_importance_samples: int = None,
                 iwelbo_batchsize: int = -1,

                 estimate_kl_div_for_full_chain: bool = False,
                 kl_div_kde_kernel: str = 'rbf',
                 kl_div_iwelbo_num_importance_samples: int = None,
                 estimate_kl_div: bool = False,
                 kl_div_estimator_frequency: int = None,

                 estimate_fid_chain: bool = False,
                 fid_estimator_size: int = None,

                 estimate_kid_chain: bool = False,
                 kid_estimator_kernel: str = None,
                 kid_full_chain_linear_estimator_repeats: int = None,

                 estimate_mog_kldivs: bool = False,
                 num_mog_em_iterations: int = None,
                 num_mog_kl_samples: int = None,
                 mog_kldivs_skip_first_num_imps: int = None,
                 mog_kldivs_use_every_nth: int = 1,
                 mog_em_use_solver: bool = False,
                 estimate_mog_jsd: bool = False,

                 estimate_sampler_variance: bool = False,

                 compute_mae_metric: bool = False,

                 data_channel_dim: int = None,

                 ssim_image_dims: List[int] = [1, 28, 28],
                ):
        super().__init__()
        self.save_hyperparameters()

        assert not estimate_sampler_ess or sampler_ess_estimator_warmup > 0

    def set_model(self, model: pl.LightningModule):
        self.model = model

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def set_mog_params_from_datamodule(self):
        assert isinstance(self.datamodule.test_data_core, ToyDataset)
        self.true_comp_probs = torch.tensor(self.datamodule.test_data_core.data_file['comp_probs'])
        self.true_means = torch.tensor(self.datamodule.test_data_core.data_file['means'])
        self.true_covs = torch.tensor(self.datamodule.test_data_core.data_file['covs'])

    def set_inception_model(self, model: pl.LightningModule):
        self.inception_model = model
        self.inception_model.freeze()
        self.inception_model.eval()

    def set_classifier_model(self, model: pl.LightningModule):
        self.classifier_model = model
        self.classifier_model.freeze()
        self.classifier_model.eval()

    def on_test_start(self):
        if hasattr(self, 'datamodule') and hasattr(self.datamodule, 'test_miss_model'):
            self.datamodule.test_miss_model = self.datamodule.test_miss_model.to(self.device)
        if hasattr(self, 'datamodule') and hasattr(self.datamodule, 'test_miss_model') and hasattr(self.datamodule.test_miss_model, 'set_classifier'):
            self.datamodule.test_miss_model.set_classifier(self.classifier_model.to(self.device))

    def sampler(self, X, M):
        # To be implemented by a subclass
        raise NotImplementedError()

    def basic_imputation(self, X, M):
        if self.hparams.imputation_fn in imputation_fn.keys():
            return imputation_fn[self.hparams.imputation_fn](X, M)
        elif self.hparams.imputation_fn == 'vae_samples':
            X_imp = rearrange(self.model.sample(X.shape[0]*X.shape[1]), '(b k) ... -> b k ...', b=X.shape[0], k=X.shape[1])
            M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)

            # Replace the missing values with 0 before doing imputation
            X = X.clone()
            X[repeat(~M_expanded, 'b 1 ... -> b k ...', k=X.shape[1])] = 0.

            return X*M_expanded + X_imp*(~M_expanded)
        elif self.hparams.imputation_fn == 'sgd_map':
            X = init_map(self.model, X, M,
                         self.hparams.sgd_map_init_num_particles,
                         self.hparams.sgd_map_init_sgd_steps,
                         self.hparams.sgd_map_init_sgd_lr,
                         use_lr_schedule=self.hparams.sgd_map_init_use_lr_sched,
                         data_channel_dim=self.hparams.data_channel_dim)
            return X
        else:
            raise NotImplementedError()

    @property
    def max_iterations(self):
        # To be implemented by a subclass
        raise NotImplementedError()

    def on_test_epoch_start(self):
        self.metrics = defaultdict(list)
        self.ess = []
        self.is_normalised_ess = []
        self.sampler_var = []
        self.avg_norm_is_ess = []

        if self.hparams.estimate_mmd_with_model:
            with torch.inference_mode():
                self.ref_samples = self.model.sample(num_samples=self.hparams.mmd_estimator_size).cpu()

        if self.hparams.estimate_fid_chain or self.hparams.estimate_kid_chain:
            with torch.inference_mode():
                ref_samples_fid = self.model.sample(num_samples=self.hparams.fid_estimator_size)
                inception_model = self.inception_model.eval()
                _, self.ref_feats_fid = inception_model(ref_samples_fid)
                self.ref_feats_fid = self.ref_feats_fid.cpu()

        if self.hparams.estimate_mog_kldivs:
            self.set_mog_params_from_datamodule()
            self.mog_kldivs = []
            self.mog_kldivs_rev = []
            self.mog_jsds = []

            self.mog_true_conditional_comp_log_probs = []
            self.mog_true_conditional_means = []
            self.mog_true_conditional_covs = []

            self.mog_conditional_comp_log_probs = []
            self.mog_conditional_means = []
            self.mog_conditional_covs = []
            self.mog_imputations_for_fitting_mogs = []
            self.mog_imputation_masks_for_fitting_mogs = []

            self.mog_condpost_kl_fow = []
            self.mog_condpost_kl_rev = []
            self.mog_condpost_jsd = []
            self.mog_condpost_logprobs = []
            self.mog_condpost_logprobs_true = []

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        X_true, M = batch[:2]
        I = batch[-1]
        X = X_true.clone()

        if (self.hparams.estimate_mmd_with_model or self.hparams.estimate_mmd_for_full_chain) and batch_idx > 0:
            # NOTE: this could be done now with the reimplemented MMD computation
            raise NotImplementedError('MMD estimation not supported when there are multiple test minibatches (implementation needs a lot of memory).')

        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X_true)

        X = repeat(X, 'b ... -> b k ...', k=self.hparams.num_copies)
        M = rearrange(M, 'b ... -> b 1 ...')
        X_true = rearrange(X_true, 'b ... -> b 1 ...')

        with torch.no_grad():
            metrics = defaultdict(list)
            sequence = []
            feature_sequence = []
            iss_sequence = []

            imputations = []
            latents = None
            if self.hparams.store_latents:
                latents = []

            for r in trange(self.hparams.num_sequential_runs, desc='Sequential runs'):
                # Impute values
                X = self.basic_imputation(X, M)
                # Log initial imputations
                if bool_data:
                    imputations.append(X.cpu().bool().numpy())
                else:
                    imputations.append(X.cpu().numpy())

                for t, outputs in enumerate(self.sampler(X, M)):
                    X = outputs['X']

                    if ((self.hparams.store_last_n_iterations == -1 or (self.max_iterations - self.hparams.store_last_n_iterations) <= t)
                        and t % self.hparams.store_every_n_iterations == 0):
                        if bool_data:
                            imputations.append(X.cpu().bool().numpy())
                        else:
                            imputations.append(X.cpu().numpy())
                        if self.hparams.store_latents:
                            latents.append(outputs['Z'].cpu().numpy())

                    if self.hparams.estimate_sampler_ess or self.hparams.estimate_mmd_for_full_chain or self.hparams.estimate_kl_div_for_full_chain or self.hparams.estimate_mog_kldivs:
                        if bool_data:
                            sequence.append(X.cpu().bool().numpy())
                        else:
                            sequence.append(X.cpu().numpy())

                    if self.hparams.estimate_fid_chain or self.hparams.estimate_kid_chain:
                        inception_model = self.inception_model.eval()
                        _, feats = inception_model(X)
                        feature_sequence.append(feats.cpu().numpy())

                    for key in outputs.keys():
                        if key in ('X', 'Z', 'is_ess_not_reduced', 'total_num_props'):
                            continue
                        if isinstance(outputs[key], torch.Tensor):
                            metrics[key].append(outputs[key].cpu().numpy())
                        else:
                            metrics[key].append(outputs[key])

                    if self.hparams.normalise_trajectory_ess_by_is_ess:
                        iss_sequence.append(outputs['is_ess_not_reduced']/outputs['total_num_props'])

                    # Imputation/Point-wise metrics

                    if self.hparams.compute_f1_metric:
                        f1 = imputation_f1_metric(X_imp=X, X_true=X_true, M=M)
                        metrics['imp_metric/f1'].append(asnumpy(f1))

                    if self.hparams.compute_inv_f1_metric:
                        f1_inv = imputation_inv_f1_metric(X_imp=X, X_true=X_true, M=M)
                        metrics['imp_metric/f1_inv'].append(asnumpy(f1_inv))

                    if self.hparams.compute_bin_accuracy_metric:
                        bin_acc = imputation_bin_accuracy_metric(X_imp=X, X_true=X_true, M=M)
                        metrics['imp_metric/bin_accuracy'].append(asnumpy(bin_acc))

                    if self.hparams.compute_rmse_metric:
                        rmse = imputation_rmse_metric(X_imp=X, X_true=X_true, M=M)
                        metrics['imp_metric/rmse'].append(asnumpy(rmse))

                    if self.hparams.compute_mae_metric:
                        mae = imputation_mae_metric(X_imp=X, X_true=X_true, M=M)
                        metrics['imp_metric/mae'].append(asnumpy(mae))

                    if self.hparams.compute_ssim_metric:
                        ssim = imputation_ssim_metric(X_imp=X, X_true=X_true, img_shape=self.hparams.ssim_image_dims)
                        metrics['imp_metric/ssim'].append(asnumpy(ssim))

                    if self.hparams.compute_latent_wass_metric:
                        latent_wass = imputation_latent_wass(X_imp=X, X_true=X_true, vae=self.model)
                        metrics['imp_metric/latent_wass'].append(asnumpy(latent_wass))

                    # Sampling metrics (use with caution - estimating sampling quality is difficult)

                    if self.hparams.estimate_mmd_with_model and t % self.hparams.mmd_estimator_frequency == 0:
                        if self.hparams.mmd_avg_of_chains:
                            mmds = []
                            for k in range(X.shape[1]):
                                mmd = mmd_metric(X_imp=X[:, k, ...].cpu(),
                                                X_ref=self.ref_samples,
                                                kernel=self.hparams.mmd_estimator_kernel)
                                mmds.append(mmd.detach().cpu().numpy())
                            metrics['mmd'].append(np.mean(mmds))
                        else:
                            mmd = mmd_metric(X_imp=rearrange(X, 'b k ... -> (b k) ...').cpu(),
                                            X_ref=self.ref_samples,
                                            kernel=self.hparams.mmd_estimator_kernel)
                            metrics['mmd'].append(mmd.detach().cpu().numpy())

                    if self.hparams.estimate_iwelbo and t % self.hparams.iwelbo_estimator_frequency == 0:
                        if self.hparams.iwelbo_batchsize > 0:
                            iwelbo = torch.zeros(X.shape[0], X.shape[1], device=X.device)
                            for i in range(math.ceil(X.shape[0] / self.hparams.iwelbo_batchsize)):
                                X_i = X[i*self.hparams.iwelbo_batchsize:min((i+1)*self.hparams.iwelbo_batchsize, X.shape[0])]
                                M_i = M[i*self.hparams.iwelbo_batchsize:min((i+1)*self.hparams.iwelbo_batchsize, X.shape[0])]
                                iwelbo_i = compute_iwelbo(self.model, X_i, torch.ones_like(M_i),
                                                        num_importance_samples=self.hparams.iwelbo_num_importance_samples, return_weights=False)
                                iwelbo[i*self.hparams.iwelbo_batchsize:min((i+1)*self.hparams.iwelbo_batchsize, X.shape[0])] = iwelbo_i
                            iwelbo = reduce(iwelbo, 'b k -> ', 'mean')
                            metrics['iwelbo'].append(iwelbo.detach().cpu().numpy())
                        else:
                            iwelbo = compute_iwelbo(self.model, X, torch.ones_like(M), num_importance_samples=self.hparams.iwelbo_num_importance_samples, return_weights=False)
                            iwelbo = reduce(iwelbo, 'b k -> ', 'mean')

                            metrics['iwelbo'].append(iwelbo.detach().cpu().numpy())

                    if self.hparams.estimate_kl_div and t % self.hparams.kl_div_estimator_frequency == 0:
                        kl_div = compute_KL_div(self.model, X, M,
                                                kde_kernel=self.hparams.kl_div_kde_kernel,
                                                num_importance_samples=self.hparams.kl_div_iwelbo_num_importance_samples)

                        metrics['kl_div_iter'].append(kl_div.detach().cpu().numpy())

        # Save last imputation if not saved
        if ((self.hparams.store_last_n_iterations == -1 or (self.max_iterations - self.hparams.store_last_n_iterations) <= t)
                and not (t % self.hparams.store_every_n_iterations == 0)):
            if bool_data:
                imputations.append(X.cpu().bool().numpy())
            else:
                imputations.append(X.cpu().numpy())
            if self.hparams.store_latents:
                latents.append(outputs['Z'].cpu().numpy())

        if self.hparams.estimate_mmd_with_model:
            # Estimate target MMD
            mmds = []
            for t in range(self.hparams.estimate_mmd_target_num_repeats):
                ref_samples2 = self.model.sample(num_samples=X.shape[0]).cpu()

                mmd = mmd_metric(X_imp=ref_samples2,
                                 X_ref=self.ref_samples,
                                 kernel=self.hparams.mmd_estimator_kernel)

                mmds.append(mmd.cpu().numpy())

            avg_mmd = np.mean(mmds)
            mmd_std = np.std(mmds, ddof=1)

            metrics['mmd_target'].append(avg_mmd)
            metrics['mmd_target_std'].append(mmd_std)

        if self.hparams.estimate_mmd_for_full_chain:
            # Estimate MMD over the full chain
            chain_mmds = []
            for t in range(self.hparams.mmd_full_chain_linear_estimator_repeats):
                if self.hparams.mmd_avg_of_chains:
                    mmds = []
                    sequence = rearrange(sequence, 't b k ... -> t b k ...')
                    for k in range(sequence.shape[2]):
                        mmd = mmd_linear_time_metric(X_imp=sequence[:, :, k, ...].cpu(),
                                            X_ref=self.ref_samples,
                                            kernel=self.hparams.mmd_estimator_kernel)
                        mmds.append(mmd.detach().cpu().numpy())
                    # metrics['mmd_chain'].append(np.mean(mmds))
                    chain_mmds.append(np.mean(mmds))
                else:
                    mmd = mmd_linear_time_metric(X_imp=rearrange(sequence, 't b k ... -> (t b k) ...'),
                                                X_ref=self.ref_samples,
                                                kernel=self.hparams.mmd_estimator_kernel)
                    # metrics['mmd_chain'].append(mmd.detach().cpu().numpy())
                    chain_mmds.append(mmd.detach().cpu().numpy())
            chain_mmd = np.mean(chain_mmds)
            chain_mmd_stderr = np.std(chain_mmds, ddof=1) / len(chain_mmds)**0.5
            metrics['mmd_chain'].append(chain_mmd)
            metrics['mmd_chain_stderr'].append(chain_mmd_stderr)

        if self.hparams.estimate_fid_chain:
            ref_feat_mean, ref_feat_cov = torch.mean(self.ref_feats_fid, dim=0), torch.cov(self.ref_feats_fid.T)

            feat_seq = rearrange(feature_sequence, 't b k ... -> (t b k) ...')
            feat_mean, feat_cov = torch.mean(torch.tensor(feat_seq), dim=0), torch.cov(torch.tensor(feat_seq.T))

            sqrt = scipy.linalg.sqrtm(feat_cov @ ref_feat_cov)
            fid = torch.norm(ref_feat_mean - feat_mean, p=2)**2 + torch.trace(ref_feat_cov + feat_cov - 2*sqrt)

            metrics['fid_chain'].append(fid)

        if self.hparams.estimate_kid_chain:
            feat_seq = rearrange(feature_sequence, 't b k ... -> (t b k) ...')

            # Estimate MMD over the full chain
            chain_kids = []
            for t in range(self.hparams.kid_full_chain_linear_estimator_repeats):
                kid = mmd_linear_time_metric(X_imp=feat_seq,
                                            X_ref=self.ref_feats_fid,
                                            kernel=self.hparams.kid_estimator_kernel)
                chain_kids.append(kid.detach().cpu().numpy())
            chain_kid = np.mean(chain_kids)
            chain_kid_stderr = np.std(chain_kids, ddof=1) / len(chain_kids)**0.5
            metrics['kid_chain'].append(chain_kid)
            metrics['kid_chain_stderr'].append(chain_kid_stderr)

        if self.hparams.estimate_iwelbo:
            # ref_samples2 = self.model.sample(num_samples=X.shape[0])
            ref_samples2 = X_true

            if self.hparams.iwelbo_batchsize > 0:
                # TODO: rewrite to use iwelbo_batchsize
                iwelbo = torch.zeros(ref_samples2.shape[0], device=X.device)
                for i in range(X.shape[0]):
                    X_i = ref_samples2[i].unsqueeze(0)
                    M_i = torch.ones_like(X_i)
                    iwelbo_i = compute_iwelbo(self.model, X_i, M_i,
                                              num_importance_samples=self.hparams.iwelbo_num_importance_samples, return_weights=False)
                    iwelbo[i] = iwelbo_i.squeeze(0)
                iwelbo = reduce(iwelbo, 'b -> ', 'mean')
                metrics['iwelbo_target'].append(iwelbo.detach().cpu().numpy())
            else:
                iwelbo = compute_iwelbo(self.model, ref_samples2, torch.ones_like(ref_samples2),
                                        num_importance_samples=self.hparams.iwelbo_num_importance_samples, return_weights=False)
                iwelbo = reduce(iwelbo, 'b k -> ', 'mean')

                metrics['iwelbo_target'].append(iwelbo.detach().cpu().numpy())

            del ref_samples2

        if self.hparams.estimate_kl_div_for_full_chain:
            sequence_for_kl = torch.tensor(rearrange(sequence, 't b k ... -> b (t k) ...'))
            kl_div = compute_KL_div(self.model, sequence_for_kl, M.cpu(),
                                    kde_kernel=self.hparams.kl_div_kde_kernel,
                                    num_importance_samples=self.hparams.kl_div_iwelbo_num_importance_samples)

            metrics['kl_div'].append(kl_div.detach().cpu().numpy())

            del sequence_for_kl

        if self.hparams.estimate_mog_kldivs:
            sequence_mog_kldivs = sequence
            if self.hparams.mog_kldivs_skip_first_num_imps is not None:
                sequence_mog_kldivs = sequence[self.hparams.mog_kldivs_skip_first_num_imps:]
            sequence_mog_kldivs = sequence_mog_kldivs[::self.hparams.mog_kldivs_use_every_nth]
            sequence_mog_kldivs = rearrange(sequence_mog_kldivs, 't b k ... -> b (t k) ...')

            self.mog_imputations_for_fitting_mogs.append(sequence_mog_kldivs)
            self.mog_imputation_masks_for_fitting_mogs.append(M.cpu().bool())

            true_cond_comp_log_probs, true_cond_means, true_cond_covs = \
                compute_conditional_mog_parameters(X[:, 0].cpu(), M.squeeze(1).cpu(),
                                                   torch.log(self.true_comp_probs),
                                                   self.true_means,
                                                   self.true_covs)

            self.mog_true_conditional_comp_log_probs.append(true_cond_comp_log_probs)
            self.mog_true_conditional_means.append(true_cond_means)
            self.mog_true_conditional_covs.append(true_cond_covs)

            # Compute metrics on p(z | xo) and \hat p(z | xo) = \sum_k p(z | xo, xm_k), where k is the number of imputations
            K_temp = sequence_mog_kldivs.shape[1]
            log_prob_c_given_ximps_b = compute_joint_p_c_given_xo_xm(torch.tensor(rearrange(sequence_mog_kldivs, 'b k d -> (b k) d')),
                                                                 torch.log(self.true_comp_probs),
                                                                 self.true_means,
                                                                 self.true_covs)
            log_prob_c_given_ximps_b = rearrange(log_prob_c_given_ximps_b, '(b k) c -> b k c', k=K_temp)
            mog_cond_log_probs_from_imps_b = torch.logsumexp(log_prob_c_given_ximps_b, dim=1) - np.log(K_temp, dtype=np.float32)
            condpost_kl_fow, condpost_kl_rev, condpost_jsd = compute_kldivs_between_categoricals(true_cond_comp_log_probs, mog_cond_log_probs_from_imps_b)

            self.mog_condpost_kl_fow.append(condpost_kl_fow)
            self.mog_condpost_kl_rev.append(condpost_kl_rev)
            self.mog_condpost_jsd.append(condpost_jsd)
            self.mog_condpost_logprobs.append(mog_cond_log_probs_from_imps_b)
            self.mog_condpost_logprobs_true.append(true_cond_comp_log_probs)

            # Find diverged chains
            log_probs = batched_compute_joint_log_probs_sparse_mogs(torch.tensor(sequence_mog_kldivs), M.squeeze(1).cpu(),
                                                                    comp_log_probs=true_cond_comp_log_probs,
                                                                    means=true_cond_means,
                                                                    covs=true_cond_covs,use_solver=self.hparams.mog_em_use_solver)
            log_probs = torch.logsumexp(log_probs, dim=-1)

            # NOTE: For numerical reasons replace the diverged chains with another imputation from the prior
            diverged = log_probs < -1e8
            idx = torch.where(diverged)
            samples = self.model.sample(diverged.sum())
            sequence_mog_kldivs[idx] = samples.cpu()

            comp_log_probs, means, covs = batched_fit_mogs_sparse(torch.tensor(sequence_mog_kldivs),
                                                                  M.squeeze(1).cpu(),
                                                                  num_components=self.true_comp_probs.shape[-1],
                                                                  num_iterations=self.hparams.num_mog_em_iterations,
                                                                  use_solver=self.hparams.mog_em_use_solver)

            self.mog_conditional_comp_log_probs.append(comp_log_probs)
            self.mog_conditional_means.append(means)
            self.mog_conditional_covs.append(covs)

            out1 = compute_kl_div_for_sparse_mogs(params1={
                                                         'comp_log_probs': true_cond_comp_log_probs,
                                                         'means': true_cond_means,
                                                         'covs': true_cond_covs,
                                                         },
                                                     params2={
                                                         'comp_log_probs': comp_log_probs.cpu(),
                                                         'means': means.cpu(),
                                                         'covs': covs.cpu(),
                                                         },
                                                     M=M.squeeze(1).cpu(),
                                                     N=self.hparams.num_mog_kl_samples,
                                                     use_solver=self.hparams.mog_em_use_solver,
                                                     return_jsd_midpoint=self.hparams.estimate_mog_jsd)

            out2 = compute_kl_div_for_sparse_mogs(params1={
                                                         'comp_log_probs': comp_log_probs.cpu(),
                                                         'means': means.cpu(),
                                                         'covs': covs.cpu(),
                                                         },
                                                     params2={
                                                         'comp_log_probs': true_cond_comp_log_probs,
                                                         'means': true_cond_means,
                                                         'covs': true_cond_covs,
                                                         },
                                                     M=M.squeeze(1).cpu(),
                                                     N=self.hparams.num_mog_kl_samples,
                                                     use_solver=self.hparams.mog_em_use_solver,
                                                     return_jsd_midpoint=self.hparams.estimate_mog_jsd)

            if self.hparams.estimate_mog_jsd:
                kldivs1, jsd_term1 = out1
                kldivs2, jsd_term2 = out2

                jsd = 0.5*(jsd_term1 + jsd_term2)
                self.mog_jsds.append(jsd)
                jsd_mean = jsd.mean()
                jsd_median = jsd.median()

                metrics['mog_jsd_mean'].append(jsd_mean.detach().cpu().numpy())
                metrics['mog_jsd_median'].append(jsd_median.detach().cpu().numpy())
            else:
                kldivs1 = out1
                kldivs2 = out2

            self.mog_kldivs.append(kldivs1)
            kldiv1_mean = kldivs1.mean()
            kldiv1_median = kldivs1.median()

            self.mog_kldivs_rev.append(kldivs2)
            kldiv2_mean = kldivs2.mean()
            kldiv2_median = kldivs2.median()

            skldivs = 0.5*(kldivs1 + kldivs2)
            skldiv_mean = skldivs.mean()
            skldiv_median = skldivs.median()

            metrics['mog_kldiv_mean'].append(kldiv1_mean.detach().cpu().numpy())
            metrics['mog_kldiv_median'].append(kldiv1_median.detach().cpu().numpy())
            metrics['mog_kldiv_rev_mean'].append(kldiv2_mean.detach().cpu().numpy())
            metrics['mog_kldiv_rev_median'].append(kldiv2_median.detach().cpu().numpy())
            metrics['mog_skldiv_mean'].append(skldiv_mean.detach().cpu().numpy())
            metrics['mog_skldiv_median'].append(skldiv_median.detach().cpu().numpy())

        if self.hparams.estimate_sampler_variance:
            M_not = asnumpy(~M.squeeze(-2))
            sequence_temp = rearrange(sequence, 't b k ... -> b (t k) ...')

            std = np.std(sequence_temp, axis=1, ddof=True)*M_not
            var = (std**2).sum(axis=-1) / M_not.sum(axis=-1)

            var = var[np.isfinite(var)]

            self.sampler_var.append(var)


        if self.hparams.estimate_sampler_ess:
            M_not = asnumpy(~M.squeeze(-2))

            sequence = rearrange(sequence, 't ... -> t ...')
            ess = self.eval_sequence_ess(sequence)
            ess = (ess * M_not).sum(axis=-1) / M_not.sum(axis=-1)
            if self.hparams.normalise_trajectory_ess_by_is_ess:
                avg_norm_is_ess = reduce(rearrange(iss_sequence[self.hparams.sampler_ess_estimator_warmup:], 't b -> t b'), 't b -> b', 'mean')

                is_normalised_trajectory_ess = ess * avg_norm_is_ess
                is_normalised_trajectory_ess = is_normalised_trajectory_ess[~np.isnan(is_normalised_trajectory_ess)]

                # Normalise by trajectory length
                is_normalised_trajectory_ess = is_normalised_trajectory_ess / ((sequence.shape[0] - self.hparams.sampler_ess_estimator_warmup)*sequence.shape[2])

                self.is_normalised_ess.append(is_normalised_trajectory_ess)

            ess = ess[~np.isnan(ess)]

            # Normalise
            ess = ess / ((sequence.shape[0] - self.hparams.sampler_ess_estimator_warmup)*sequence.shape[2])

            self.ess.append(ess)

        # if self.hparams.store_is_norm_ess:
        #     assert self.hparams.normalise_trajectory_ess_by_is_ess
        #     avg_norm_is_ess = reduce(rearrange(iss_sequence[self.hparams.sampler_ess_estimator_warmup:], 't b -> t b'), 't b -> b', 'mean')

        #     self.avg_norm_is_ess.append(avg_norm_is_ess)

        imputations = rearrange(imputations, 't b k ... -> t b k ...')
        if self.hparams.store_latents:
            latents = rearrange(latents, 't b k ... -> t b k ...')
        if len(imputations) > 0 and not self.hparams.dont_store_imputations:
            if bool_data:
                self.log_batch_imputations(imputations, latents, asnumpy(X_true.bool()), asnumpy(M), asnumpy(I), batch_idx)
            else:
                self.log_batch_imputations(imputations, latents, asnumpy(X_true), asnumpy(M), asnumpy(I), batch_idx)

        if self.hparams.save_img_snapshot_in_tb:
            if bool_data:
                self.log_imputation_img_snapshot_in_tb(imputations, asnumpy(X_true.bool()), asnumpy(M), batch_idx)
            else:
                self.log_imputation_img_snapshot_in_tb(imputations, asnumpy(X_true), asnumpy(M), batch_idx)

        for key, value in metrics.items():
            self.metrics[key].append(value)
        self.metrics['num_X'].append(X_true.shape[0])

    def log_batch_imputations(self, imputations, latents, true_X, M, data_idx, eval_batch_idx):
        np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), f'imputations_{eval_batch_idx}.npz'),
                            imputations=imputations,
                            latents=latents,
                            true_X=true_X,
                            masks=M,
                            data_idx=data_idx)

    def log_imputation_img_snapshot_in_tb(self, imputations, true_X, M, eval_batch_idx):
        #NOTE: only store 10 from each batch as a snapshot in Tensorboard
        tensorboard_imputations = imputations[:, :10, 0, ...]
        if self.hparams.data_channel_dim is None:
            tensorboard_imputations = np.reshape(tensorboard_imputations, tensorboard_imputations.shape[:-1] + tuple(self.hparams.snapshot_image_dims))
            tensorboard_imputations = rearrange(tensorboard_imputations, 't n h w -> (n h) (t w)')
            true_X = true_X[:10, 0, ...]
            true_X = np.reshape(true_X, true_X.shape[:-1] + tuple(self.hparams.snapshot_image_dims))
            true_X = rearrange(true_X, 'n h w -> (n h) w')
            masks = M[:10, 0, ...]
            masks = np.reshape(masks, masks.shape[:-1] + tuple(self.hparams.snapshot_image_dims))
            masks = rearrange(masks, 'n h w -> (n h) w')
            masked_X = true_X * masks + 0.5*(~masks)

            tensorboard_imputations = np.concatenate([true_X, masked_X, tensorboard_imputations], axis=-1)
            self.logger.experiment.add_image(f'imputation_preview/test', tensorboard_imputations, eval_batch_idx, dataformats='HW')
        else:
            tensorboard_imputations = np.reshape(tensorboard_imputations, tensorboard_imputations.shape[:self.hparams.data_channel_dim] + tuple(self.hparams.snapshot_image_dims))
            tensorboard_imputations = rearrange(tensorboard_imputations, 't n c h w -> c (n h) (t w)')
            true_X = true_X[:10, 0, ...]
            true_X = np.reshape(true_X, true_X.shape[:self.hparams.data_channel_dim] + tuple(self.hparams.snapshot_image_dims))
            true_X = rearrange(true_X, 'n c h w -> c (n h) w')
            masks = M[:10, 0, ...]
            masks = np.reshape(masks, masks.shape[:-1] + (1,) + tuple(self.hparams.snapshot_image_dims)[1:])
            masks = rearrange(masks, 'n c h w -> c (n h) w')
            masked_X = true_X * masks + 0.5*(~masks)

            tensorboard_imputations = np.concatenate([true_X, masked_X, tensorboard_imputations], axis=-1)
            self.logger.experiment.add_image(f'imputation_preview/test', tensorboard_imputations, eval_batch_idx, dataformats='CHW')

    def test_epoch_end(self, outputs):
        num_X = np.array(self.metrics['num_X'])

        for key, values in self.metrics.items():
            if key == 'num_X':
                continue
            elif key.startswith('imp_metric/'):
                metric_name = key.split('imp_metric/')[1]
                values = np.concatenate([np.stack(i, axis=0) for i in values], axis=1)
                values = rearrange(values, 't n k -> n k t')

                metric_mean_over_data = values.mean(axis=0)
                metric_mean_over_imps = metric_mean_over_data.mean(axis=0)
                if metric_mean_over_data.shape[0] != 1:
                    metric_std_over_imps = metric_mean_over_data.std(axis=0, ddof=1)
                else:
                    metric_std_over_imps = np.ones_like(metric_mean_over_imps)

                # No Array logging in tensorboard so log in a loop.
                for t, v in enumerate(metric_mean_over_imps):
                    self.logger.experiment.add_scalar(f'{metric_name}/mean/test', v, t)
                for t, v in enumerate(metric_std_over_imps):
                    self.logger.experiment.add_scalar(f'{metric_name}/std/test', v, t)

                metric_quantiles = np.quantile(rearrange(values, 'n k t -> (n k) t'), axis=0, q=self.hparams.imputation_metric_quantiles)
                for i, q in enumerate(self.hparams.imputation_metric_quantiles):
                    for t, v in enumerate(metric_quantiles[i]):
                        self.logger.experiment.add_scalar(f'{metric_name}/q{q}/test', v, t)
                continue
            else:
                metric = np.array(values)
                metric = (metric * num_X[:, None]).sum(axis=0) / np.sum(num_X)

            # No Array logging in tensorboard so log in a loop.
            for t, v in enumerate(metric):
                # self.log(f'{key}/test', v, on_epoch=True, prog_bar=True, logger=True)
                self.logger.experiment.add_scalar(f'{key}/test', v, t)

        if self.hparams.estimate_sampler_ess:
            import matplotlib.pyplot as plt

            ess = np.concatenate(self.ess)

            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'ess.npz'), ess=ess)

            fig, ax = plt.subplots()
            ax.hist(ess)

            self.logger.experiment.add_figure('ess/test', fig, close=True)

            # IS-normalised ESS for IRWG
            if self.hparams.normalise_trajectory_ess_by_is_ess:
                is_normalised_ess = np.concatenate(self.is_normalised_ess)

                np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'is_normalised_ess.npz'), is_normalised_ess=is_normalised_ess)

                fig, ax = plt.subplots()
                ax.hist(is_normalised_ess)

                self.logger.experiment.add_figure('is_normalised_ess/test', fig, close=True)

        if self.hparams.estimate_sampler_variance:
            import matplotlib.pyplot as plt

            sampler_var = np.concatenate(self.sampler_var)

            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'sampler_var.npz'), sampler_var=sampler_var)

            fig, ax = plt.subplots()
            ax.hist(sampler_var)

            self.logger.experiment.add_figure('sampler_var/test', fig, close=True)

        # if self.hparams.store_is_norm_ess:
        #     avg_norm_is_ess = np.concatenate(self.avg_norm_is_ess)

        #     np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'avg_norm_is_ess.npz'), avg_norm_is_ess=avg_norm_is_ess)

        #     fig, ax = plt.subplots()
        #     ax.hist(avg_norm_is_ess)

        #     self.logger.experiment.add_figure('avg_norm_is_ess/test', fig, close=True)

        if self.hparams.estimate_mog_kldivs:
            import matplotlib.pyplot as plt

            # Save p(c | xo) stats
            mog_condpost_kl_fow = np.concatenate(self.mog_condpost_kl_fow)
            mog_condpost_kl_rev = np.concatenate(self.mog_condpost_kl_rev)
            mog_condpost_jsd = np.concatenate(self.mog_condpost_jsd)
            mog_condpost_logprobs= np.concatenate(self.mog_condpost_logprobs)
            mog_condpost_logprobs_true = np.concatenate(self.mog_condpost_logprobs_true)
            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'mog_comp_log_probs.npz'),
                    mog_true_cond_comp_log_probs=asnumpy(mog_condpost_logprobs_true),
                    mog_cond_log_probs_from_imps=asnumpy(mog_condpost_logprobs),
                    # log_prob_c_given_ximps=asnumpy(log_prob_c_given_ximps),
                    kl_fow=asnumpy(mog_condpost_kl_fow),
                    kl_rev=asnumpy(mog_condpost_kl_rev),
                    jsd=asnumpy(mog_condpost_jsd)
                    )

            # Save mog conditional parameters
            mog_true_cond_comp_log_probs = np.concatenate(self.mog_true_conditional_comp_log_probs)
            mog_true_cond_means = np.concatenate(self.mog_true_conditional_means)
            mog_true_cond_covs = np.concatenate(self.mog_true_conditional_covs)
            mog_cond_comp_log_probs = np.concatenate(self.mog_conditional_comp_log_probs)
            mog_cond_means = np.concatenate(self.mog_conditional_means)
            mog_cond_covs = np.concatenate(self.mog_conditional_covs)
            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'cond_mog_params.npz'),
                                mog_true_cond_comp_log_probs=mog_true_cond_comp_log_probs,
                                mog_true_cond_means=mog_true_cond_means,
                                mog_true_cond_covs=mog_true_cond_covs,
                                mog_cond_comp_log_probs=mog_cond_comp_log_probs,
                                mog_cond_means=mog_cond_means,
                                mog_cond_covs=mog_cond_covs
                                )

            # Save mog imputations (used for fitting the mogs)
            mog_imputations_for_fitting_mogs = np.concatenate(self.mog_imputations_for_fitting_mogs)
            mog_imputation_masks_for_fitting_mogs = np.concatenate(self.mog_imputation_masks_for_fitting_mogs)
            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(),
                                             'mog_imputations_for_fitting.npz'),
                                mog_imputations_for_fitting_mogs=mog_imputations_for_fitting_mogs,
                                mog_imputation_masks_for_fitting_mogs=mog_imputation_masks_for_fitting_mogs)

            # Forward KL

            mog_kldivs = np.concatenate(self.mog_kldivs)

            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'mog_kldivs.npz'), mog_kldivs=mog_kldivs)

            fig, ax = plt.subplots()
            ax.hist(mog_kldivs)

            self.logger.experiment.add_figure('mog_kldivs_hist/test', fig, close=True)

            # Rev KL

            mog_kldivs_rev = np.concatenate(self.mog_kldivs_rev)

            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'mog_kldivs_rev.npz'), mog_kldivs_rev=mog_kldivs_rev)

            fig, ax = plt.subplots()
            ax.hist(mog_kldivs_rev)

            self.logger.experiment.add_figure('mog_kldivs_rev_hist/test', fig, close=True)

            # JSD
            if self.hparams.estimate_mog_jsd:
                mog_jsds = np.concatenate(self.mog_jsds)

                np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), 'mog_jsds.npz'), mog_jsds=mog_jsds)

                fig, ax = plt.subplots()
                ax.hist(mog_jsds)

                self.logger.experiment.add_figure('mog_jsds_hist/test', fig, close=True)

    def eval_sequence_ess(self, sequence):
        import tensorflow_probability as tfp

        sequence = sequence[self.hparams.sampler_ess_estimator_warmup:]

        if self.hparams.ess_evaluation_batchsize == -1:
            cv = tfp.mcmc.effective_sample_size(sequence.astype(np.float), cross_chain_dims=2).numpy()
            # cv = tfp.mcmc.effective_sample_size(sequence.astype(np.float), cross_chain_dims=None).numpy().sum(1)
        else:
            batchsize = self.hparams.ess_evaluation_batchsize
            cvs = []
            for b in tqdm(range(math.ceil(sequence.shape[1] / batchsize)), desc='Estimating ESS'):
                sequence_b = sequence[:, b*batchsize:min((b+1)*batchsize, sequence.shape[1]), ...]
                cv = tfp.mcmc.effective_sample_size(sequence_b.astype(np.float), cross_chain_dims=2).numpy()
                # cv = tfp.mcmc.effective_sample_size(sequence_b.astype(np.float), cross_chain_dims=None).numpy().sum(1)
                cvs.append(cv)
            cv = np.concatenate(cvs, axis=0)
        cv[np.isnan(cv)] = 1.

        return cv

class TestVAEPseudoGibbs(TestVAESamplerBase):
    """
    Pseudo-Gibbs sampling

    Args:
        num_iterations:                 Number of pseudo-Gibbs iterations
    """
    def __init__(self,
                 num_iterations: int,
                 *args,
                 clip_imputations: bool = False,
                 clipping_mode: str = 'batch_data',
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_iterations

    def sampler(self, X, M):
        if self.hparams.clip_imputations:
            if self.hparams.clipping_mode == 'batch_data':
                X_temp = X[:, 0].clone()
                X_temp *= M.squeeze(1)
                max_values = torch.max(X_temp, dim=0)[0]*2
                min_values = torch.min(X_temp, dim=0)[0]*2
            elif self.hparams.clipping_mode == 'dataset':
                max_values = torch.tensor(self.datamodule.test_data_core.data_max*2, device=X.device)
                min_values = torch.tensor(self.datamodule.test_data_core.data_min*2, device=X.device)
            else:
                raise NotImplementedError(f'Invalid {self.hparams.clipping_mode=}')

        for t in tqdm(range(self.hparams.num_iterations), desc='Running Pseudo-Gibbs'):
            if self.hparams.clip_imputations:
                X, Z_imp = pseudo_gibbs_iteration(self.model, X, M, clip_values_min=min_values, clip_values_max=max_values, data_channel_dim=self.hparams.data_channel_dim)
            else:
                X, Z_imp = pseudo_gibbs_iteration(self.model, X, M, data_channel_dim=self.hparams.data_channel_dim)
            outputs = {'X': X, 'Z': Z_imp}
            yield outputs


class TestVAEMetropolisWithinGibbs(TestVAESamplerBase):
    """
    Metropolis-within-Gibbs sampling

    Args:
        num_pseudo_warmup_iterations:                 Number of pseudo-Gibbs iterations
        num_mwg_iterations:                           Number of MWG iterations after PG
        clip_imputations_during_pg_warmup:            Clip PG imputations
        var_proposal_temperature:                     Temperature for variational proposal
        var_proposal_anneal_type_to_prior:            Annealing type for variational proposal
    """
    def __init__(self,
                 num_pseudo_warmup_iterations: int,
                 num_mwg_iterations: int,
                 *args,
                 clip_imputations_during_pg_warmup: bool = False,
                 clipping_mode: str = 'batch_data',

                 var_proposal_temperature: float = None,
                 var_proposal_anneal_type_to_prior: str = None,
                 warmup_method: str = 'pseudo_gibbs',

                 # IRWG warmup
                 num_k_per_cluster: int = None,
                 num_imp_samples: int = 1,
                 weighting_scheme: str = 'dmis_within_groups',
                 resampling_scheme: str = 'glr',
                 resampling_method: str = 'multinomial',
                 glr_cluster_size: int = None,
                 num_prior_replenish_proposals: int = 0,

                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        if self.hparams.warmup_method == 'irwg':
            assert (self.hparams.num_k_per_cluster + self.hparams.num_prior_replenish_proposals/self.hparams.glr_cluster_size) == self.hparams.glr_cluster_size

    @property
    def max_iterations(self):
        if self.hparams.warmup_method != 'irwg':
            return self.hparams.num_pseudo_warmup_iterations + self.hparams.num_mwg_iterations
        else:
            return (self.hparams.num_pseudo_warmup_iterations*self.hparams.num_k_per_cluster) + self.hparams.num_mwg_iterations

    def sampler(self, X, M):
        model = self.model
        M_not = ~M
        M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
        M_not_expanded = ~M_expanded

        clip_values_max, clip_values_min = None, None
        if self.hparams.clip_imputations_during_pg_warmup:
            if self.hparams.clipping_mode == 'batch_data':
                X_temp = X[:, 0].clone()
                X_temp *= M.squeeze(1)
                clip_values_max = torch.max(X_temp, dim=0)[0]*2
                clip_values_min = torch.min(X_temp, dim=0)[0]*2
            elif self.hparams.clipping_mode == 'dataset':
                clip_values_max = torch.tensor(self.datamodule.test_data_core.data_max*2, device=X.device)
                clip_values_min = torch.tensor(self.datamodule.test_data_core.data_min*2, device=X.device)
            else:
                raise NotImplementedError(f'Invalid {self.hparams.clipping_mode=}')

        if self.hparams.warmup_method == 'none':
            var_latent_params = model.predict_var_latent_params(X, M)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)
        elif self.hparams.warmup_method == 'pseudo_gibbs':
            for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running Pseudo-Gibbs warmup'):
                # Create latent distribution and sample
                var_latent_params = model.predict_var_latent_params(X, M)

                # Sample latent variables
                var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
                Z_new = var_latent_distr.sample()

                # Create the distribution of the missingness model
                mis_params = model.generator_network(Z_new)
                mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

                # Sample missing values
                X_m = mis_distr.sample()

                if self.hparams.clip_imputations_during_pg_warmup:
                    # Workaround for unstable Pseudo-Gibbs sampler
                    if clip_values_min is not None:
                        X_m = torch.max(X_m, clip_values_min)
                    if clip_values_max is not None:
                        X_m = torch.min(X_m, clip_values_max)
                    # Another safety
                    is_nan = ~(X_m.isfinite())
                    X_m[is_nan] = X[is_nan]

                # Set imputed missing values
                X_new =  X*M_expanded + X_m*M_not_expanded
                X = X_new

                outputs = {
                    'X': X,
                    'Z': Z_new,
                    'acceptance': 1.
                }

                yield outputs
        elif self.hparams.warmup_method == 'irwg':
            K = X.shape[1] * self.hparams.num_k_per_cluster
            X_irwg_warmup = repeat(X[:, 0], 'b ... -> b k ...', k=K)
            X_irwg_warmup = self.basic_imputation(X_irwg_warmup, M)

            for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running IRWG warmup'):
                X_irwg_warmup, _, _, _, _, _, _, _, _ = importance_resampling_gibbs_iteration(self.model, X_irwg_warmup, M,
                                                                                        self.hparams.num_imp_samples,
                                                            glr_cluster_size=self.hparams.glr_cluster_size,
                                                            weighting=self.hparams.weighting_scheme,
                                                            resampling_method=self.hparams.resampling_method,
                                                            resampling_scheme=self.hparams.resampling_scheme,
                                                            num_prior_replenish_proposals=self.hparams.num_prior_replenish_proposals,
                                                            # num_historical_proposals=num_historical_proposals,
                                                            # history=history,
                                                            data_channel_dim=self.hparams.data_channel_dim,
                                                            var_proposal_temperature=self.hparams.var_proposal_temperature,
                                                            var_proposal_anneal_type_to_prior=self.hparams.var_proposal_anneal_type_to_prior)

                X_temp = rearrange(X_irwg_warmup, 'b (c k) ... -> b k c ...', c=self.hparams.glr_cluster_size)
                X_temp = X_temp.contiguous()
                if self.hparams.num_prior_replenish_proposals > 0:
                    needed_imps = X.shape[1] - X_temp.shape[1]
                    X_temp = torch.concat([X_temp, X_temp[:, :needed_imps]], dim=1)

                for i in range(self.hparams.num_k_per_cluster):
                    X_temp_i = X_temp[:, :, i]

                    outputs = {'X': X_temp_i,
                               'acceptance': 1.0}
                    yield outputs
                if self.hparams.num_prior_replenish_proposals > 0:
                    if t == self.hparams.num_pseudo_warmup_iterations-1:
                        X_irwg_warmup = rearrange(X_temp, 'b k c ... -> b (c k) ...', c=self.hparams.glr_cluster_size)

                del X_temp
                del X_temp_i

            # Only select one from each "cluster" as current X
            X = rearrange(X_irwg_warmup, 'b (c k) ... -> b k c ...', c=self.hparams.glr_cluster_size)
            X = X[:, :, 0]
            del X_irwg_warmup

            X = X.contiguous()
            var_latent_params = model.predict_var_latent_params(X, M)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # NOTE: Assuming standard prior
        # prior_distr = torch.distributions.Normal(0, 1)
        prior_distr = model.get_prior()

        Z_old = Z_new
        mis_params_old = mis_params
        Z_old_prior_logprob = reduce(prior_distr.log_prob(Z_old), '... d -> ...', 'sum')
        X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

        for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations, self.hparams.num_pseudo_warmup_iterations+self.hparams.num_mwg_iterations), desc='Running MWG'):
            # Create latent distribution and sample
            var_latent_params = model.predict_var_latent_params(X, M)
            if self.hparams.var_proposal_temperature is not None:
                var_latent_params = model.distribution_param_tempering(var_latent_params,
                                                                       model.hparams.var_latent_distribution,
                                                                       temperature=self.hparams.var_proposal_temperature)
            if self.hparams.var_proposal_anneal_type_to_prior is not None:
                var_latent_params = model.anneal_var_distribution_to_prior_based_on_missingness(
                    var_latent_params, M,
                    var_distribution=model.hparams.var_latent_distribution,
                    anneal_type=self.hparams.var_proposal_anneal_type_to_prior)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()
            Z_new_var_logprob = reduce(var_latent_distr.log_prob(Z_new), '... d -> ...', 'sum')
            Z_old_var_logprob = reduce(var_latent_distr.log_prob(Z_old), '... d -> ...', 'sum')

            # Eval prior
            Z_new_prior_logprob = reduce(prior_distr.log_prob(Z_new), '... d -> ...', 'sum')

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

            X_new_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

            log_accept = (X_new_logprob + Z_new_prior_logprob) - (X_old_logprob + Z_old_prior_logprob) + (Z_old_var_logprob - Z_new_var_logprob)

            # acceptance_prob = torch.exp(torch.clamp(log_accept, -25, 25))
            acceptance_prob = torch.exp(log_accept)

            acceptance_samples = torch.rand_like(acceptance_prob)
            accepted = acceptance_samples < acceptance_prob

            accepted_not = ~accepted
            Z_old = Z_old*accepted_not.unsqueeze(-1) + Z_new*accepted.unsqueeze(-1)
            Z_old_prior_logprob = Z_old_prior_logprob*accepted_not + Z_new_prior_logprob*accepted

            mis_params_old = mis_params_old*accepted_not.unsqueeze(-1) + mis_params*accepted.unsqueeze(-1)
            mis_distr = model.create_distribution(mis_params_old, model.hparams.generator_distribution)

            # Sample missing values
            X_m = mis_distr.sample()

            # Set imputed missing values
            X_new =  X*M_expanded + X_m*M_not_expanded
            X = X_new

            X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

            outputs = {
                'X': X,
                'Z': Z_old,
                'acceptance': accepted.float().mean()
            }

            # if isinstance(var_latent_distr, torch.distributions.Normal):
            #     avg_var_std = (var_latent_distr.stddev).mean(-1)
            #     outputs['var_std_median'] = avg_var_std.median()
            #     outputs['var_std_mean'] = avg_var_std.mean()

            yield outputs

class TestVAEMetropolisWithinGibbsMNAR(TestVAESamplerBase):
    """
    Metropolis-within-Gibbs sampling for MNAR data

    Args:
        num_pseudo_warmup_iterations:                 Number of pseudo-Gibbs iterations
        num_mwg_iterations:                           Number of MWG iterations after PG
        clip_imputations_during_pg_warmup:            Clip PG imputations
        var_proposal_temperature:                     Temperature for the proposal distribution
        var_proposal_anneal_type_to_prior:            Anneal the proposal distribution to the prior
    """
    def __init__(self,
                 num_pseudo_warmup_iterations: int,
                 num_mwg_iterations: int,
                 *args,
                 clip_imputations_during_pg_warmup: bool = False,
                 var_proposal_temperature: float = None,
                 var_proposal_anneal_type_to_prior: str = None,
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_pseudo_warmup_iterations + self.hparams.num_mwg_iterations

    def sampler(self, X, M):
        model = self.model
        M_not = ~M
        M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
        M_not_expanded = ~M_expanded

        clip_values_max, clip_values_min = None, None
        if self.hparams.clip_imputations_during_pg_warmup:
            X_temp = X[:, 0].clone()
            X_temp *= M.squeeze(1)
            clip_values_max = torch.max(X_temp, dim=0)[0]*2
            clip_values_min = torch.min(X_temp, dim=0)[0]*2

        for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running Pseudo-Gibbs warmup'):
            # Create latent distribution and sample
            var_latent_params = model.predict_var_latent_params(X, M)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

            # Sample missing values
            X_m = mis_distr.sample()

            if self.hparams.clip_imputations_during_pg_warmup:
                # Workaround for unstable Pseudo-Gibbs sampler
                if clip_values_min is not None:
                    X_m = torch.max(X_m, clip_values_min)
                if clip_values_max is not None:
                    X_m = torch.min(X_m, clip_values_max)
                # Another safety
                is_nan = ~(X_m.isfinite())
                X_m[is_nan] = X[is_nan]

            # Set imputed missing values
            X_new =  X*M_expanded + X_m*M_not_expanded
            X = X_new

            outputs = {
                'X': X,
                'acceptance': 1.
            }

            # if isinstance(var_latent_distr, torch.distributions.Normal):
            #     avg_var_std = (var_latent_distr.stddev).mean(-1)
            #     outputs['var_std_median'] = avg_var_std.median()
            #     outputs['var_std_mean'] = avg_var_std.mean()

            yield outputs

        # NOTE: Assuming standard prior
        # prior_distr = torch.distributions.Normal(0, 1)
        prior_distr = model.get_prior()
        del mis_params, mis_distr, var_latent_distr, X_new, X_m, var_latent_params

        Z = Z_new
        # mis_params_old = mis_params
        # Z_old_prior_logprob = reduce(prior_distr.log_prob(Z_old), '... d -> ...', 'sum')
        # X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

        for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations, self.hparams.num_pseudo_warmup_iterations+self.hparams.num_mwg_iterations), desc='Running MWG'):
            # Create latent distribution and sample
            var_latent_params = model.predict_var_latent_params(X, M)
            if self.hparams.var_proposal_temperature is not None:
                var_latent_params = model.distribution_param_tempering(var_latent_params,
                                                                       model.hparams.var_latent_distribution,
                                                                       temperature=self.hparams.var_proposal_temperature)
            if self.hparams.var_proposal_anneal_type_to_prior is not None:
                var_latent_params = model.anneal_var_distribution_to_prior_based_on_missingness(
                    var_latent_params, M,
                    var_distribution=model.hparams.var_latent_distribution,
                    anneal_type=self.hparams.var_proposal_anneal_type_to_prior)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()
            Z_new_var_logprob = reduce(var_latent_distr.log_prob(Z_new), '... d -> ...', 'sum')
            Z_old_var_logprob = reduce(var_latent_distr.log_prob(Z), '... d -> ...', 'sum')

            # Eval prior
            Z_new_prior_logprob = reduce(prior_distr.log_prob(Z_new), '... d -> ...', 'sum')
            Z_old_prior_logprob = reduce(prior_distr.log_prob(Z), '... d -> ...', 'sum')

            # Create the distribution of the missingness model
            mis_params_new = model.generator_network(Z_new)
            mis_params_old = model.generator_network(Z)
            mis_distr_new = model.create_distribution(mis_params_new, model.hparams.generator_distribution)
            mis_distr_old = model.create_distribution(mis_params_old, model.hparams.generator_distribution)

            X_new_logprob = reduce(mis_distr_new.log_prob(X), '... d -> ...', 'sum')
            X_old_logprob = reduce(mis_distr_old.log_prob(X), '... d -> ...', 'sum')

            log_accept = (X_new_logprob + Z_new_prior_logprob) - (X_old_logprob + Z_old_prior_logprob) + (Z_old_var_logprob - Z_new_var_logprob)

            acceptance_prob = torch.exp(log_accept)

            acceptance_samples = torch.rand_like(acceptance_prob)
            accepted_z = acceptance_samples < acceptance_prob

            accepted_z_not = ~accepted_z
            Z = Z*accepted_z_not.unsqueeze(-1) + Z_new*accepted_z.unsqueeze(-1)
            # Z_old_prior_logprob = Z_old_prior_logprob*accepted_z_not + Z_new_prior_logprob*accepted_z

            #
            # Sample XM with MH acceptance using the miss_model
            #

            mis_params = mis_params_old*accepted_z_not.unsqueeze(-1) + mis_params_new*accepted_z.unsqueeze(-1)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

            # Sample missing values
            X_m = mis_distr.sample()

            # Set imputed missing values
            X_new =  X*M_expanded + X_m*M_not_expanded

            mis_log_prob_old = self.datamodule.test_miss_model.log_prob(X, M)
            mis_log_prob_new = self.datamodule.test_miss_model.log_prob(X_new, M)

            log_accept = mis_log_prob_new - mis_log_prob_old

            acceptance_prob = torch.exp(log_accept)

            acceptance_samples = torch.rand_like(acceptance_prob)
            accepted_x = acceptance_samples < acceptance_prob

            accepted_x_not = ~accepted_x
            X = X*accepted_x_not.unsqueeze(-1) + X_new*accepted_x.unsqueeze(-1)

            outputs = {
                'X': X,
                'acceptance_z': accepted_z.float().mean(),
                'acceptance_x': accepted_x.float().mean(),
            }

            # if isinstance(var_latent_distr, torch.distributions.Normal):
            #     avg_var_std = (var_latent_distr.stddev).mean(-1)
            #     outputs['var_std_median'] = avg_var_std.median()
            #     outputs['var_std_mean'] = avg_var_std.mean()

            yield outputs

class TestVAEAdaptiveCollapsedMetropolisWithinGibbs(TestVAESamplerBase):
    """
    Metropolis-within-Gibbs sampling marginalised missing values
    called AC-MWG (adaptive collapsed-Metropolis-within-Gibbs) in the paper.

    Args:
        num_pseudo_warmup_iterations:                 Number of pseudo-Gibbs iterations
        num_mwg_iterations:                           Number of MWG iterations after PG
        prior_mixture_probability:                    Probability for the proposals to mix with prior

        clip_imputations_during_pg_warmup:            Clip PG imputations
        var_proposal_temperature:                     Temperature for the variational proposal
        var_proposal_anneal_type_to_prior:            Anneal the variational proposal to the prior

        ablation_use_mwg_target:                      Only used for ablation studies: uses p(z|xm, xo) as a target instead of p(z|xo)
    """
    def __init__(self,
                 num_pseudo_warmup_iterations: int,
                 num_mwg_iterations: int,
                 prior_mixture_probability: float,

                 *args,
                 enqueue_allwarmup_samples_to_history: bool = False,
                 warmup_method: str = 'pseudo_gibbs',
                 clip_imputations_during_pg_warmup: bool = False,
                 clipping_mode: str = 'batch_data',
                 var_proposal_temperature: float = None,
                 var_proposal_anneal_type_to_prior: str = None,
                 num_prior_replenish_proposals: int = 0,

                 # IRWG warmup
                 num_k_per_cluster: int = None,
                 num_imp_samples: int = 1,
                 weighting_scheme: str = 'dmis_within_groups',
                 resampling_scheme: str = 'glr',
                 resampling_method: str = 'multinomial',
                 glr_cluster_size: int = None,

                 ablation_use_mwg_target: bool = False,

                 **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if self.hparams.warmup_method == 'irwg':
            assert (self.hparams.num_k_per_cluster + self.hparams.num_prior_replenish_proposals/self.hparams.glr_cluster_size) == self.hparams.glr_cluster_size

    @property
    def max_iterations(self):
        if self.hparams.warmup_method != 'irwg':
            return self.hparams.num_pseudo_warmup_iterations + self.hparams.num_mwg_iterations
        else:
            return (self.hparams.num_pseudo_warmup_iterations*self.hparams.num_k_per_cluster) + self.hparams.num_mwg_iterations

    def sampler(self, X, M):
        model = self.model
        M_not = ~M
        M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
        M_not_expanded = ~M_expanded

        clip_values_max, clip_values_min = None, None
        if self.hparams.clip_imputations_during_pg_warmup:
            if self.hparams.clipping_mode == 'batch_data':
                X_temp = X[:, 0].clone()
                X_temp *= M.squeeze(1)
                clip_values_max = torch.max(X_temp, dim=0)[0]*2
                clip_values_min = torch.min(X_temp, dim=0)[0]*2
            elif self.hparams.clipping_mode == 'dataset':
                clip_values_max = torch.tensor(self.datamodule.test_data_core.data_max*2, device=X.device)
                clip_values_min = torch.tensor(self.datamodule.test_data_core.data_min*2, device=X.device)
            else:
                raise NotImplementedError(f'Invalid {self.hparams.clipping_mode=}')

        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X)

        history = ImputationHistoryQueue_with_restrictedavailability(
                                        max_history_length=self.max_iterations,
                                        batch_shape=X.shape,
                                        dtype=X.dtype if not bool_data else torch.bool)

        if self.hparams.warmup_method == 'none':
            history.enqueue_batch((X.cpu().bool() if bool_data else X.cpu()))

            var_latent_params = model.predict_var_latent_params(X, M)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)
        elif self.hparams.warmup_method == 'pseudo_gibbs':
            for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running Pseudo-Gibbs warmup'):
                # Create latent distribution and sample
                var_latent_params = model.predict_var_latent_params(X, M)

                # Sample latent variables
                var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
                Z_new = var_latent_distr.sample()

                # Create the distribution of the missingness model
                mis_params = model.generator_network(Z_new)
                mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

                # Sample missing values
                X_m = mis_distr.sample()

                if self.hparams.clip_imputations_during_pg_warmup:
                    # Workaround for unstable Pseudo-Gibbs sampler
                    if clip_values_min is not None:
                        X_m = torch.max(X_m, clip_values_min)
                    if clip_values_max is not None:
                        X_m = torch.min(X_m, clip_values_max)
                    # Another safety
                    is_nan = ~(X_m.isfinite())
                    X_m[is_nan] = X[is_nan]

                # Set imputed missing values
                X_new =  X*M_expanded + X_m*M_not_expanded
                X = X_new

                if self.hparams.enqueue_allwarmup_samples_to_history or t == self.hparams.num_pseudo_warmup_iterations - 1:
                    history.enqueue_batch((X.cpu().bool() if bool_data else X.cpu()))

                outputs = {
                    'X': X,
                    'Z': Z_new,
                    'acceptance': 1.
                }

                yield outputs
        elif self.hparams.warmup_method == 'irwg':
            K = X.shape[1] * self.hparams.num_k_per_cluster
            X_irwg_warmup = repeat(X[:, 0], 'b ... -> b k ...', k=K)
            X_irwg_warmup = self.basic_imputation(X_irwg_warmup, M)

            for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running IRWG warmup'):
                X_irwg_warmup, _, _, _, _, _, _, _, _ = importance_resampling_gibbs_iteration(self.model, X_irwg_warmup, M,
                                                                                        self.hparams.num_imp_samples,
                                                            glr_cluster_size=self.hparams.glr_cluster_size,
                                                            weighting=self.hparams.weighting_scheme,
                                                            resampling_method=self.hparams.resampling_method,
                                                            resampling_scheme=self.hparams.resampling_scheme,
                                                            num_prior_replenish_proposals=self.hparams.num_prior_replenish_proposals,
                                                            # num_historical_proposals=num_historical_proposals,
                                                            # history=history,
                                                            data_channel_dim=self.hparams.data_channel_dim,
                                                            var_proposal_temperature=self.hparams.var_proposal_temperature,
                                                            var_proposal_anneal_type_to_prior=self.hparams.var_proposal_anneal_type_to_prior)

                X_temp = rearrange(X_irwg_warmup, 'b (c k) ... -> b k c ...', c=self.hparams.glr_cluster_size)
                X_temp = X_temp.contiguous()
                if self.hparams.num_prior_replenish_proposals > 0:
                    needed_imps = X.shape[1] - X_temp.shape[1]
                    X_temp = torch.concat([X_temp, X_temp[:, :needed_imps]], dim=1)

                for i in range(self.hparams.num_k_per_cluster):
                    X_temp_i = X_temp[:, :, i]

                    if self.hparams.enqueue_allwarmup_samples_to_history or t == self.hparams.num_pseudo_warmup_iterations - 1:
                        history.enqueue_batch((X_temp_i.cpu().bool() if bool_data else X_temp_i.cpu()))

                    outputs = {'X': X_temp_i,
                               'acceptance': 1.0}
                    yield outputs
                if self.hparams.num_prior_replenish_proposals > 0:
                    if t == self.hparams.num_pseudo_warmup_iterations-1:
                        X_irwg_warmup = rearrange(X_temp, 'b k c ... -> b (c k) ...', c=self.hparams.glr_cluster_size)

                del X_temp
                del X_temp_i

            # Only select one from each "cluster" as current X
            X = rearrange(X_irwg_warmup, 'b (c k) ... -> b k c ...', c=self.hparams.glr_cluster_size)
            X = X[:, :, 0]
            del X_irwg_warmup

            X = X.contiguous()
            var_latent_params = model.predict_var_latent_params(X, M)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # NOTE: Assuming standard prior
        # prior_distr = torch.distributions.Normal(0, 1)
        prior_distr = model.get_prior()

        Z_old = Z_new
        mis_params_old = mis_params
        Z_old_prior_logprob = reduce(prior_distr.log_prob(Z_old), '... d -> ...', 'sum')

        if self.hparams.ablation_use_mwg_target:
            X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')
        else:
            X_old_logprob = reduce(mis_distr.log_prob(X)*M_expanded, '... d -> ...', 'sum')

        history.update_available_timesteps(torch.ones(X.shape[:2], dtype=torch.bool))

        for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations, self.hparams.num_pseudo_warmup_iterations+self.hparams.num_mwg_iterations),
                      desc='Running MWG w/ Marginalisation'):
            X_hist = history.sample_history_nonstratified_for_each_chain(num_samples_per_chain=1)
            X_hist = rearrange(X_hist, 'k b d -> b k d').to(X.device)
            X_hist = X_hist.to(X.dtype).contiguous()

            # Create latent distribution and sample
            var_latent_params = model.predict_var_latent_params(X_hist, M)
            if self.hparams.var_proposal_temperature is not None:
                var_latent_params = model.distribution_param_tempering(var_latent_params,
                                                                       model.hparams.var_latent_distribution,
                                                                       temperature=self.hparams.var_proposal_temperature)
            if self.hparams.var_proposal_anneal_type_to_prior is not None:
                var_latent_params = model.anneal_var_distribution_to_prior_based_on_missingness(
                    var_latent_params, M,
                    var_distribution=model.hparams.var_latent_distribution,
                    anneal_type=self.hparams.var_proposal_anneal_type_to_prior)

            # Sample latent variables
            var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
            Z_new = var_latent_distr.sample()

            # Sample the prior and use prior sample with prior_mixture_probability
            if self.hparams.prior_mixture_probability > 0.:
                Z_new_fromprior = prior_distr.sample(Z_new.shape)

                use_prior_sample = torch.bernoulli(torch.ones(Z_new.shape[:2])*self.hparams.prior_mixture_probability).bool()
                Z_new[use_prior_sample] = Z_new_fromprior[use_prior_sample].to(Z_new.device)

            Z_new_var_logprob = reduce(var_latent_distr.log_prob(Z_new), '... d -> ...', 'sum')
            Z_old_var_logprob = reduce(var_latent_distr.log_prob(Z_old), '... d -> ...', 'sum')
            # Compute the proposal-mixture probability
            if self.hparams.prior_mixture_probability > 0.:
                Z_new_var_logprob_2 = reduce(prior_distr.log_prob(Z_new), '... d -> ...', 'sum')
                Z_old_var_logprob_2 = reduce(prior_distr.log_prob(Z_old), '... d -> ...', 'sum')

                log_prior_mixture_probability = torch.log(torch.tensor(self.hparams.prior_mixture_probability))
                log_var_mixture_probability = torch.log(torch.tensor(1-self.hparams.prior_mixture_probability))

                Z_new_var_logprob = torch.logsumexp(
                    rearrange([Z_new_var_logprob+log_var_mixture_probability,
                               Z_new_var_logprob_2+log_prior_mixture_probability],
                              'vp ... -> vp ...'),
                    dim=0)
                Z_old_var_logprob = torch.logsumexp(
                    rearrange([Z_old_var_logprob+log_var_mixture_probability,
                               Z_old_var_logprob_2+log_prior_mixture_probability],
                              'vp ... -> vp ...'),
                    dim=0)

            # Eval prior
            Z_new_prior_logprob = reduce(prior_distr.log_prob(Z_new), '... d -> ...', 'sum')

            # Create the distribution of the missingness model
            mis_params = model.generator_network(Z_new)
            mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)


            if self.hparams.ablation_use_mwg_target:
                X_new_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')
            else:
                X_new_logprob = reduce(mis_distr.log_prob(X)*M_expanded, '... d -> ...', 'sum')

            log_accept = (X_new_logprob + Z_new_prior_logprob) - (X_old_logprob + Z_old_prior_logprob) \
                + (Z_old_var_logprob - Z_new_var_logprob)

            # acceptance_prob = torch.exp(torch.clamp(log_accept, -25, 25))
            acceptance_prob = torch.exp(log_accept)

            acceptance_samples = torch.rand_like(acceptance_prob)
            accepted = acceptance_samples < acceptance_prob

            # Update available timesteps for accepted samples
            history.update_available_timesteps(accepted.cpu())

            accepted_not = ~accepted
            Z_old = Z_old*accepted_not.unsqueeze(-1) + Z_new*accepted.unsqueeze(-1)
            Z_old_prior_logprob = Z_old_prior_logprob*accepted_not + Z_new_prior_logprob*accepted

            mis_params_old = mis_params_old*accepted_not.unsqueeze(-1) + mis_params*accepted.unsqueeze(-1)
            mis_distr = model.create_distribution(mis_params_old, model.hparams.generator_distribution)

            # Sample missing values
            X_m = mis_distr.sample()

            # Set imputed missing values
            X_new = X*M_expanded + X_m*M_not_expanded
            X = X_new

            history.enqueue_batch((X.cpu().bool() if bool_data else X.cpu()))

            X_old_logprob = reduce(mis_distr.log_prob(X)*M_expanded, '... d -> ...', 'sum')

            outputs = {
                'X': X,
                'Z': Z_old,
                'acceptance': accepted.float().mean()
            }

            # if isinstance(var_latent_distr, torch.distributions.Normal):
            #     avg_var_std = (var_latent_distr.stddev).mean(-1)
            #     outputs['var_std_median'] = avg_var_std.median()
            #     outputs['var_std_mean'] = avg_var_std.mean()

            yield outputs

class TestVAELatentAdaptiveImportanceResampling(TestVAESamplerBase):
    """
    IRWG sampling, called LAIR (latent-adaptive importance resampling) in the paper

    Args:
        num_iterations:                 Number of IRWG iterations
        num_imp_samples:                Number of importance samples
        weighting_scheme:               Which weighting scheme to use (e.g. dmis, smis)
        resampling_scheme:              Which resampling scheme to use (e.g. gr, lr, ir)
        resampling_method:              Which resampling method to use (e.g. multinomial, systematic)
        glr_cluster_size:               Cluster size for use with global-local resampling
        num_prior_replenish_proposals:  Number of prior replenish proposals
        prior_replenish_iteration_frequency: How often to replenish the proposals from the prior

        var_proposal_temperature:       Temperature for the variational proposal
        var_proposal_anneal_type_to_prior:  How to anneal the variational proposal to the prior

        weight_transform:               What (if any) non-linear transformation to apply to the weights before resampling
        clip_imputations:               Whether to clip imputations to the data range

        num_historical_proposals:           How many proposals to sample from historical imputations
        no_historic_proposals_iterations: warmup iterations when history is not used
        imputation_history_length:      Size of history

        use_miss_model:         Whether miss_model should be used (for MNAR data)
        accept_reject_xm:       Whether to MH-accept-reject imputations using the miss model

        store_norm_log_weights: Whether to store normalized log weights for debugging

        resample_final_imps: Whether it should resample all the latents across iterations at the end and sample new imputations
        resample_final_imps_num_samples: The number of samples to use for the final resampling
    """
    def __init__(self,
                 num_iterations: int,
                 num_imp_samples: int,
                 weighting_scheme: str,
                 resampling_scheme: str,
                 resampling_method: str,
                 *args,
                 glr_cluster_size: int = None,
                 num_pseudo_warmup_iterations: int = 0,
                 num_prior_replenish_proposals: int = 0,
                 prior_replenish_iteration_frequency: int = 1,

                 var_proposal_temperature: float = None,
                 var_proposal_anneal_type_to_prior: str = None,

                 weight_transform: str = None,
                 clip_imputations: bool = False,

                 num_historical_proposals: int = None,
                 no_historic_proposals_iterations: int = None,
                 imputation_history_length: int = None,

                 use_miss_model: bool = False,
                 accept_reject_xm: bool = False,

                 store_norm_log_weights: bool = False,

                 resample_final_imps: bool = False,
                 resample_final_imps_num_samples: int = None,
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_iterations

    def on_test_epoch_start(self):
        super().on_test_epoch_start()

        if self.hparams.store_norm_log_weights:
            self.norm_log_weights = []

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_start(batch, batch_idx, dataloader_idx)

        if self.hparams.resample_final_imps:
            self.latent_proposals = []
            self.latent_proposal_log_weights = []

    def sampler(self, X, M):
        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X)

        if self.hparams.clip_imputations:
            X_temp = X[:, 0].clone()
            X_temp *= M.squeeze(1)
            clip_max_values = torch.max(X_temp, dim=0)[0]*2
            clip_min_values = torch.min(X_temp, dim=0)[0]*2
            del X_temp

        # Initialise history
        history = None
        if self.hparams.num_historical_proposals is not None:
            history = ImputationHistoryQueue(max_history_length=self.hparams.imputation_history_length,
                                             batch_shape=X.shape,
                                             dtype=X.dtype if not bool_data else torch.bool)

        if self.hparams.num_pseudo_warmup_iterations > 0:
            for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running Pseudo-Gibbs warmup'):
                X, Z_imp = pseudo_gibbs_iteration(self.model, X, M, data_channel_dim=self.hparams.data_channel_dim)
                outputs = {'X': X,
                           'Z': Z_imp,
                        'is_ess': 0.,
                        'is_ess_not_reduced': torch.zeros(X.shape[0]),
                        'total_num_props': 1.}

                if history is not None:
                    history.enqueue_batch((X.cpu().bool() if bool_data else X.cpu()))

                yield outputs

        all_norm_log_weights = []
        for t in tqdm(range(self.hparams.num_iterations), desc='Running IRWG'):
            weight_transform=None
            if self.hparams.weight_transform == 'poly':
                weight_transform = lambda log_weights: self.polynomial_transform(log_weights, iter=t, max_iter=self.max_iterations)
            elif self.hparams.weight_transform is not None:
                raise NotImplementedError()

            use_historical_proposals = (self.hparams.no_historic_proposals_iterations is not None
                                        and (self.hparams.num_pseudo_warmup_iterations + t) > self.hparams.no_historic_proposals_iterations)

            num_prior_replenish_proposals = 0
            if t % self.hparams.prior_replenish_iteration_frequency == 0:
                num_prior_replenish_proposals = self.hparams.num_prior_replenish_proposals

            if self.hparams.store_latents:
                X_m, ess, transformed_ess, perplexity, total_num_props, norm_log_weights, Z_prop, log_weights, acceptance_x, latents = importance_resampling_gibbs_iteration(self.model, X, M, self.hparams.num_imp_samples,
                                                            glr_cluster_size=self.hparams.glr_cluster_size,
                                                            weighting=self.hparams.weighting_scheme,
                                                            resampling_method=self.hparams.resampling_method,
                                                            resampling_scheme=self.hparams.resampling_scheme,
                                                            num_prior_replenish_proposals=num_prior_replenish_proposals,
                                                            weight_transform=weight_transform,
                                                            use_historical_proposals=use_historical_proposals,
                                                            num_historical_proposals=self.hparams.num_historical_proposals,
                                                            history=history,
                                                            miss_model=self.datamodule.test_miss_model if self.hparams.use_miss_model else None,
                                                            accept_reject_xm=self.hparams.accept_reject_xm,
                                                            data_channel_dim=self.hparams.data_channel_dim,
                                                            var_proposal_temperature=self.hparams.var_proposal_temperature,
                                                            var_proposal_anneal_type_to_prior=self.hparams.var_proposal_anneal_type_to_prior,
                                                            return_resampled_latents=True
                                                            )
            else:
                X_m, ess, transformed_ess, perplexity, total_num_props, norm_log_weights, Z_prop, log_weights, acceptance_x = importance_resampling_gibbs_iteration(self.model, X, M, self.hparams.num_imp_samples,
                                                            glr_cluster_size=self.hparams.glr_cluster_size,
                                                            weighting=self.hparams.weighting_scheme,
                                                            resampling_method=self.hparams.resampling_method,
                                                            resampling_scheme=self.hparams.resampling_scheme,
                                                            num_prior_replenish_proposals=num_prior_replenish_proposals,
                                                            weight_transform=weight_transform,
                                                            use_historical_proposals=use_historical_proposals,
                                                            num_historical_proposals=self.hparams.num_historical_proposals,
                                                            history=history,
                                                            miss_model=self.datamodule.test_miss_model if self.hparams.use_miss_model else None,
                                                            accept_reject_xm=self.hparams.accept_reject_xm,
                                                            data_channel_dim=self.hparams.data_channel_dim,
                                                            var_proposal_temperature=self.hparams.var_proposal_temperature,
                                                            var_proposal_anneal_type_to_prior=self.hparams.var_proposal_anneal_type_to_prior,
                                                            return_resampled_latents=False
                                                            )

            if self.hparams.resample_final_imps:
                self.latent_proposals.append(asnumpy(Z_prop))
                self.latent_proposal_log_weights.append(asnumpy(log_weights))

            if self.hparams.clip_imputations:
                # Workaround for unstable Pseudo-Gibbs sampler
                if clip_min_values is not None:
                    X_m = torch.max(X_m, clip_min_values)
                if clip_max_values is not None:
                    X_m = torch.min(X_m, clip_max_values)
                # Another safety
                is_nan = ~(X_m.isfinite())
                X_m[is_nan] = X[is_nan]
            X = X_m

            if self.hparams.store_norm_log_weights:
                all_norm_log_weights.append(norm_log_weights.detach().cpu().numpy())

            ess_not_reduced = ess

            ess = ess.mean(dim=0)
            transformed_ess.mean(dim=0)

            if history is not None:
                history.enqueue_batch((X.cpu().bool() if bool_data else X.cpu()))

            outputs = {'X': X,
                       'Z': latents,
                       'is_ess': torch.mean(ess).detach().cpu().numpy(),
                       'is_ess_not_reduced': ess_not_reduced.detach().cpu().numpy(),
                       'transformed_ess': torch.mean(transformed_ess).detach().cpu().numpy(),
                       'perplexity': torch.mean(perplexity).detach().cpu().numpy(),
                       'total_num_props': total_num_props,
                       'acceptance_x': acceptance_x.detach().cpu().numpy(),
                      }
            yield outputs

        if self.hparams.store_norm_log_weights:
            all_norm_log_weights = rearrange(all_norm_log_weights, 't b i -> b t i')
            self.norm_log_weights.append(all_norm_log_weights)

    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

        X_true, M = batch[:2]
        I = batch[-1]

        M = M.unsqueeze(1)

        X_true_zeroed = X_true.clone()
        X_true_zeroed = X_true_zeroed.unsqueeze(1)
        X_true_zeroed[~M] = 0.

        if self.hparams.resample_final_imps:
            T, (L, B, K) = len(self.latent_proposals), self.latent_proposals[0].shape[0:3]

            def check_if_bool(X):
                return ((X == 0).sum() + (X == 1).sum()) == X.size
            bool_data = check_if_bool(self.latent_proposals[0])

            latent_proposals = torch.tensor(rearrange(self.latent_proposals, 't i b k ... -> b (t i k) ...'))
            latent_proposal_log_weights = torch.tensor(rearrange(self.latent_proposal_log_weights, 't i b k -> b (t i k)'))

            importance_distr = torch.distributions.Categorical(logits=latent_proposal_log_weights)
            resample_final_imps_num_samples = T*K
            if self.hparams.resample_final_imps_num_samples is not None:
                resample_final_imps_num_samples = self.hparams.resample_final_imps_num_samples
            idx = importance_distr.sample(sample_shape=(resample_final_imps_num_samples,))

            # Resample the latents across all iterations
            Z = latent_proposals[torch.arange(B), idx]
            Z = rearrange(Z, 'i b ... -> b i ...')

            batch_size = K

            imps = []
            for b in trange(math.ceil(resample_final_imps_num_samples/K), desc='Decoding the final imputations'):
                Z_b = Z[:, b*batch_size:min((b+1)*batch_size, Z.shape[1])]
                Z_b = Z_b.to(self.device)

                # Create the distribution
                generator_params = self.model.generator_network(Z_b)
                generator_distr = self.model.create_distribution(generator_params, self.model.hparams.generator_distribution)

                # Sample missing values
                X_m = generator_distr.sample()

                # Set true observed values
                X_m = X_true_zeroed*M + X_m*(~M)
                if bool_data:
                    imps.append(X_m.cpu().bool().numpy())
                else:
                    imps.append(asnumpy(X_m))

            imps = np.concatenate(imps, axis=1)[None, :]
            true_X = asnumpy(X_true.unsqueeze(1))
            M = asnumpy(M.bool())
            I = asnumpy(I)
            latents = []
            if self.hparams.store_latents:
                latents = asnumpy(Z)
            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(),
                                            f'irwg_imputations_after_final_resampling_{batch_idx}.npz'),
                                imputations=imps,
                                latents=latents,
                                true_X=true_X,
                                masks=M,
                                data_idx=I)


    def test_epoch_end(self, outputs):
        out = super().test_epoch_end(outputs)

        if self.hparams.store_norm_log_weights:
            norm_log_weights = np.concatenate(self.norm_log_weights)
            np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(),
                                                'is_norm_log_weights.npz'),
                                is_norm_log_weights=norm_log_weights)

        return out

    def polynomial_transform(self, log_weights, iter, max_iter):
        alpha = min((iter+10)/max_iter, 1)
        return log_weights*alpha

class TestVAEImportanceResampling(TestVAESamplerBase):
    """
    IR sampling

    Args:
        num_iterations:                 Number of IR iterations
        num_imp_samples:                Number of importance samples
        weighting_scheme:               Which weighting scheme to use (e.g. dmis, smis)
        resampling_scheme:              Which resampling scheme to use (e.g. gr, lr, ir)
        resampling_method:              Which resampling method to use (e.g. multinomial, systematic)
        glr_cluster_size:               Cluster size for use with global-local resampling

        resample_final_imps_num_samples: The number of samples to use for the final resampling
    """
    def __init__(self,
                 num_iterations: int,
                 num_imp_samples: int,
                 weighting_scheme: str,
                 resampling_scheme: str,
                 resampling_method: str,
                 *args,
                 glr_cluster_size: int = None,

                 resample_final_imps_num_samples: int = None,
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_iterations

    def on_test_epoch_start(self):
        super().on_test_epoch_start()

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_start(batch, batch_idx, dataloader_idx)

        self.latent_proposals = []
        self.latent_proposal_log_weights = []

    def sampler(self, X, M):
        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X)

        for t in tqdm(range(self.hparams.num_iterations), desc='Running IR'):
            X, ess, transformed_ess, perplexity, total_num_props, _, Z_prop, log_weights = standard_importance_resampling(self.model, X, M, self.hparams.num_imp_samples,
                                                           glr_cluster_size=self.hparams.glr_cluster_size,
                                                           weighting=self.hparams.weighting_scheme,
                                                           resampling_method=self.hparams.resampling_method,
                                                           resampling_scheme=self.hparams.resampling_scheme,
                                                           data_channel_dim=self.hparams.data_channel_dim)

            self.latent_proposals.append(asnumpy(Z_prop))
            self.latent_proposal_log_weights.append(asnumpy(log_weights))

            ess_not_reduced = ess

            ess = ess.mean(dim=0)
            transformed_ess.mean(dim=0)

            outputs = {'X': X,
                       'is_ess_standard': torch.mean(ess).detach().cpu().numpy(),
                       'is_ess_not_reduced': ess_not_reduced.detach().cpu().numpy(),
                       'perplexity_standard': torch.mean(perplexity).detach().cpu().numpy()
                      }
            yield outputs

    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

        X_true, M = batch[:2]
        I = batch[-1]

        M = M.unsqueeze(1)

        X_true_zeroed = X_true.clone()
        X_true_zeroed = X_true_zeroed.unsqueeze(1)
        X_true_zeroed[~M] = 0.

        T, (L, B, K) = len(self.latent_proposals), self.latent_proposals[0].shape[0:3]

        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.size
        bool_data = check_if_bool(self.latent_proposals[0])

        latent_proposals = torch.tensor(rearrange(self.latent_proposals, 't i b k ... -> b (t i k) ...'))
        latent_proposal_log_weights = torch.tensor(rearrange(self.latent_proposal_log_weights, 't i b k -> b (t i k)'))

        importance_distr = torch.distributions.Categorical(logits=latent_proposal_log_weights)
        resample_final_imps_num_samples = T*K
        if self.hparams.resample_final_imps_num_samples is not None:
            resample_final_imps_num_samples = self.hparams.resample_final_imps_num_samples
        idx = importance_distr.sample(sample_shape=(resample_final_imps_num_samples,))

        # Resample the latents across all iterations
        Z = latent_proposals[torch.arange(B), idx]
        Z = rearrange(Z, 'i b ... -> b i ...')

        batch_size = K

        imps = []
        for b in trange(math.ceil(resample_final_imps_num_samples/K), desc='Decoding the final imputations'):
            Z_b = Z[:, b*batch_size:min((b+1)*batch_size, Z.shape[1])]
            Z_b = Z_b.to(self.device)

            # Create the distribution
            generator_params = self.model.generator_network(Z_b)
            generator_distr = self.model.create_distribution(generator_params, self.model.hparams.generator_distribution)

            # Sample missing values
            X_m = generator_distr.sample()

            # Set true observed values
            X_m = X_true_zeroed*M + X_m*(~M)
            if bool_data:
                imps.append(X_m.cpu().bool().numpy())
            else:
                imps.append(asnumpy(X_m))

        imps = np.concatenate(imps, axis=1)[None, :]
        true_X = asnumpy(X_true.unsqueeze(1))
        M = asnumpy(M.bool())
        I = asnumpy(I)
        np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(),
                                        f'imputations_{batch_idx}.npz'),
                            imputations=imps,
                            true_X=true_X,
                            masks=M,
                            data_idx=I)
