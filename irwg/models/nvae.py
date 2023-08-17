from collections import defaultdict
from typing import List, Tuple, Union
import os.path

import numpy as np
import pytorch_lightning as pl
import torch
from einops import asnumpy, rearrange, repeat
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from irwg.dependencies.NVAE.model import AutoEncoder
from irwg.dependencies.NVAE.utils import (count_parameters_in_M,
                                          get_arch_cells, kl_balancer,
                                          kl_balancer_coeff, kl_coeff,
                                          reconstruction_loss)
from irwg.sampling.vae_imputation import expand_M_dim, gr_resampling
from irwg.utils.basic_imputation import imputation_fn
from irwg.utils.test_step_base import TestBase
from irwg.sampling.imputation_metrics import imputation_ssim_metric
from irwg.sampling.utils import ImputationHistoryQueue


def reconstruction_loss_with_M(decoder, x, M, crop=False):
    from irwg.dependencies.NVAE.distributions import DiscMixLogistic, Normal

    recon = decoder.log_prob(x)*M
    if crop:
        recon = recon[:, :, 2:30, 2:30]

    if isinstance(decoder, DiscMixLogistic):
        return - torch.sum(recon, dim=[1, 2])    # summation over RGB is done.
    else:
        return - torch.sum(recon, dim=[1, 2, 3])

def load_nvae(eval_args):
    # load a checkpoint
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        print('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        print('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        print('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    # if eval_args.batch_size > 0:
    #     args.batch_size = eval_args.batch_size

    print('loaded the model at epoch %d', checkpoint['epoch'])
    arch_instance = get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # model = model.cuda()

    print('args = %s', args)
    print('num conv layers: %d', len(model.all_conv_layers))
    print('param size = %fM ', count_parameters_in_M(model))

    return model


class NVAELightning(TestBase):
    def __init__(self,
                 nvae_checkpoint: str,

                 temp_p: float,
                 temp_q: float,

                 store_last_n_iterations: int,
                 store_every_n_iterations: int,

                 num_copies: int,
                 imputation_fn: str,

                 imputation_metric_quantiles: List[float],
                 compute_ssim_metric: bool = False,

                 data_channel_dim: int = None,
                 img_shape: List[int] = [32,32],

                 save_img_snapshot_in_tb: bool = False,

                 expand_x_channel_dim: bool = False,
                 ):
        super().__init__()

        class NVAEArgs():
            def __init__(self, checkpoint):
                self.checkpoint = checkpoint

        self.vae = load_nvae(NVAEArgs(nvae_checkpoint))

    def set_datamodule(self, datamodule):
        self.datamodule = datamodule

    def on_test_start(self):
        if hasattr(self, 'datamodule') and hasattr(self.datamodule, 'test_miss_model'):
            self.datamodule.test_miss_model = self.datamodule.test_miss_model.to(self.device)
        if hasattr(self, 'datamodule') and hasattr(self.datamodule, 'test_miss_model') and hasattr(self.datamodule.test_miss_model, 'set_classifier'):
            self.datamodule.test_miss_model.set_classifier(self.classifier_model.to(self.device))

    def on_test_epoch_start(self):
        self.metrics = defaultdict(list)

    def basic_imputation(self, X, M):
        if self.hparams.imputation_fn in imputation_fn.keys():
            return imputation_fn[self.hparams.imputation_fn](X, M)
        elif self.hparams.imputation_fn == 'vae_samples':
            X_imp = rearrange(self.sample_vae(X.shape[0]*X.shape[1]), '(b k) ... -> b k ...', b=X.shape[0], k=X.shape[1])
            M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
            X_imp = X_imp.reshape(*X_imp.shape[:self.hparams.data_channel_dim], -1)
            return X*M_expanded + X_imp*(~M_expanded)
        else:
            raise NotImplementedError()

    def sample_vae(self, num_samples, rng=None):
        logits = self.vae.sample(num_samples, t=self.hparams.temp_p)
        output = self.vae.decoder_output(logits)
        return output.sample()

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        X_true, M = batch[:2]
        if self.hparams.expand_x_channel_dim:
            X_true = expand_M_dim(X_true, data_channel_dim=self.hparams.data_channel_dim)
        I = batch[-1]
        X = X_true.clone()

        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X_true)

        X = repeat(X, 'b ... -> b k ...', k=self.hparams.num_copies)
        M = rearrange(M, 'b ... -> b 1 ...')
        X_true = rearrange(X_true, 'b ... -> b 1 ...')

        with torch.inference_mode():
            # Impute values
            X = self.basic_imputation(X, M)

            imputations = []
            # Log initial imputations
            if bool_data:
                imputations.append(X.cpu().bool().numpy())
            else:
                imputations.append(X.cpu().numpy())

            metrics = defaultdict(list)
            for t, outputs in enumerate(self.sampler(X, M)):
                X = outputs['X']

                if ((self.hparams.store_last_n_iterations == -1 or (self.max_iterations - t) > self.hparams.store_last_n_iterations)
                    and t % self.hparams.store_every_n_iterations == 0):
                    if bool_data:
                        imputations.append(X.cpu().bool().numpy())
                    else:
                        imputations.append(X.cpu().numpy())

                for key in outputs.keys():
                    if key in ('X', 'is_ess_not_reduced', 'total_num_props'):
                        continue
                    if isinstance(outputs[key], torch.Tensor):
                        metrics[key].append(outputs[key].cpu().numpy())
                    else:
                        metrics[key].append(outputs[key])

                if self.hparams.compute_ssim_metric:
                    ssim = imputation_ssim_metric(X_imp=X, X_true=X_true, img_shape=[X.shape[self.hparams.data_channel_dim]] + self.hparams.img_shape)
                    metrics['imp_metric/ssim'].append(asnumpy(ssim))

            imputations = rearrange(imputations, 't b k ... -> t b k ...')
            if len(imputations) > 0:
                if bool_data:
                    self.log_batch_imputations(imputations, asnumpy(X_true.bool()), asnumpy(M), asnumpy(I), batch_idx)
                else:
                    self.log_batch_imputations(imputations, asnumpy(X_true), asnumpy(M), asnumpy(I), batch_idx)

            if self.hparams.save_img_snapshot_in_tb:
                if bool_data:
                    self.log_imputation_img_snapshot_in_tb(imputations, asnumpy(X_true.bool()), asnumpy(M), batch_idx)
                else:
                    self.log_imputation_img_snapshot_in_tb(imputations, asnumpy(X_true), asnumpy(M), batch_idx)

            for key, value in metrics.items():
                self.metrics[key].append(value)
            self.metrics['num_X'].append(X_true.shape[0])

    def log_batch_imputations(self, imputations, true_X, M, data_idx, eval_batch_idx):
        np.savez_compressed(os.path.join(self.logger.experiment.get_logdir(), f'imputations_{eval_batch_idx}.npz'),
                            imputations=imputations,
                            true_X=true_X,
                            masks=M,
                            data_idx=data_idx)

    def log_imputation_img_snapshot_in_tb(self, imputations, true_X, M, eval_batch_idx):
        #NOTE: only store 10 from each batch as a snapshot in Tensorboard
        tensorboard_imputations = imputations[:, :10, 0, ...]
        if self.hparams.data_channel_dim is None:
            # tensorboard_imputations = np.reshape(tensorboard_imputations, tensorboard_imputations.shape[:-1] + tuple(self.hparams.snapshot_image_dims))
            # tensorboard_imputations = rearrange(tensorboard_imputations, 't n h w -> (n h) (t w)')
            # true_X = true_X[:10, 0, ...]
            # true_X = np.reshape(true_X, true_X.shape[:-1] + tuple(self.hparams.snapshot_image_dims))
            # true_X = rearrange(true_X, 'n h w -> (n h) w')
            # masks = M[:10, 0, ...]
            # masks = np.reshape(masks, masks.shape[:-1] + tuple(self.hparams.snapshot_image_dims))
            # masks = rearrange(masks, 'n h w -> (n h) w')
            # masked_X = true_X * masks + 0.5*(~masks)

            # tensorboard_imputations = np.concatenate([true_X, masked_X, tensorboard_imputations], axis=-1)
            # self.logger.experiment.add_image(f'imputation_preview/test', tensorboard_imputations, eval_batch_idx, dataformats='HW')
            raise NotImplementedError()
        else:
            tensorboard_imputations = np.reshape(tensorboard_imputations, tensorboard_imputations.shape[:self.hparams.data_channel_dim+1] + tuple(self.hparams.img_shape))
            tensorboard_imputations = rearrange(tensorboard_imputations, 't n c h w -> c (n h) (t w)')
            true_X = true_X[:10, 0, ...]
            true_X = np.reshape(true_X, true_X.shape[:self.hparams.data_channel_dim+1] + tuple(self.hparams.img_shape))
            true_X = rearrange(true_X, 'n c h w -> c (n h) w')
            masks = M[:10, 0, ...]
            masks = np.reshape(masks, masks.shape[:-1] + (1,) + tuple(self.hparams.img_shape))
            masks = rearrange(masks, 'n c h w -> c (n h) w')
            masked_X = true_X * masks + 0.5*(~masks)

            tensorboard_imputations = np.concatenate([true_X, masked_X, tensorboard_imputations], axis=-1)
            self.logger.experiment.add_image(f'imputation_preview/test', tensorboard_imputations, eval_batch_idx, dataformats='CHW')
        self.logger.save()

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


    def sampler(self, X, M):
        # To be implemented by a subclass
        raise NotImplementedError()

    def compute_log_ratio(self, X, M):
        X_shape = X.shape
        X = X.reshape(np.prod(X_shape[:self.hparams.data_channel_dim]), X_shape[self.hparams.data_channel_dim], *self.hparams.img_shape)
        logits, log_q, log_p, _, _, _, _, _ = self.vae(X, q_t=self.hparams.temp_q, p_t=self.hparams.temp_p)
        distr = self.vae.decoder_output(logits=logits)

        # log_pz_minus_log_qz = -torch.stack(kl_all, dim=-1)
        # log_pz_minus_log_qz = log_pz_minus_log_qz.sum(-1)
        log_pz_minus_log_qz = log_p - log_q

        crop = self.vae.crop_output
        log_p = distr.log_prob(X)
        if crop:
            log_p = log_p[:, :, 2:30, 2:30]
            M = M.reshape(M.shape[:-1] + tuple(self.hparams.img_shape))
            M = M[:, :, 2:30, 2:30]
            M = M.reshape(M.shape[:-len(self.hparams.img_shape)] + (-1,))

            shape = list(X_shape)
            shape[-1] = 28*28
            X_shape = shape

        log_p = log_p.reshape(*X_shape[:self.hparams.data_channel_dim], -1)*M
        log_p = torch.sum(log_p, dim=[-1])

        log_pz_minus_log_qz = log_pz_minus_log_qz.reshape(*X_shape[:self.hparams.data_channel_dim])
        log_ratio = log_p + log_pz_minus_log_qz

        # TODO: make this more general.
        logits = logits.reshape(*X_shape[:self.hparams.data_channel_dim], *logits.shape[-3:])

        return log_ratio, logits

    def compute_dmis_log_weights(self, X, M):
        X_shape = X.shape
        X = X.reshape(np.prod(X_shape[:self.hparams.data_channel_dim]), X_shape[self.hparams.data_channel_dim], *self.hparams.img_shape)
        logits, _, log_p, log_qz0 = self.vae.forward_dmis(X, batch_shape=X_shape[:self.hparams.data_channel_dim], q_t=self.hparams.temp_q, p_t=self.hparams.temp_p)
        distr = self.vae.decoder_output(logits=logits)

        # log_pz_minus_log_qz = -torch.stack(kl_all, dim=-1)
        # log_pz_minus_log_qz = log_pz_minus_log_qz.sum(-1)
        log_pz_minus_log_qz = log_p - log_qz0

        crop = self.vae.crop_output
        log_p = distr.log_prob(X)
        if crop:
            log_p = log_p[:, :, 2:30, 2:30]
            M = M.reshape(M.shape[:-1] + tuple(self.hparams.img_shape))
            M = M[:, :, 2:30, 2:30]
            M = M.reshape(M.shape[:-len(self.hparams.img_shape)] + (-1,))

            shape = list(X_shape)
            shape[-1] = 28*28
            X_shape = shape

        log_p = log_p.reshape(*X_shape[:self.hparams.data_channel_dim], -1)*M
        log_p = torch.sum(log_p, dim=[-1])

        log_pz_minus_log_qz = log_pz_minus_log_qz.reshape(*X_shape[:self.hparams.data_channel_dim])
        log_ratio = log_p + log_pz_minus_log_qz

        # TODO: make this more general.
        logits = logits.reshape(*X_shape[:self.hparams.data_channel_dim], *logits.shape[-3:])

        return log_ratio, logits


class NVAELightningPseudoGibbs(NVAELightning):
    def __init__(self,
                 num_iterations: int,
                 *args,
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_iterations

    def sampler(self, X, M):

        for t in tqdm(range(self.hparams.num_iterations), desc='Running Pseudo-Gibbs'):
            X_shape = X.shape
            X = X.reshape(np.prod(X_shape[:self.hparams.data_channel_dim]), X_shape[self.hparams.data_channel_dim], *self.hparams.img_shape)
            logits = self.vae(X, q_t=self.hparams.temp_q, p_t=self.hparams.temp_p)[0]
            mis_distr = self.vae.decoder_output(logits)

            # Sample missing values
            X_m = mis_distr.sample()
            X_m = X_m.reshape(*X_shape)

            # Set imputed missing values
            M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
            X = X.reshape(*X_shape)
            X = X*M_expanded + X_m*(~M_expanded)

            outputs = {'X': X}
            yield outputs

class NVAELightningImportanceResamplingWithinGibbs(NVAELightning):
    def __init__(self,
                 num_iterations: int,
                 *args,
                 num_historical_proposals: int = None,
                 no_historic_proposals_iterations: int = None,
                 imputation_history_length: int = None,
                 importance_weights: str = 'dmis',
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_iterations

    def sampler(self, X, M):
        K = X.shape[1]

        M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
        M_not_expanded = ~M_expanded

        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X)

        # Initialise history
        history = None
        if self.hparams.num_historical_proposals is not None and self.hparams.num_historical_proposals > 0:
            history = ImputationHistoryQueue(max_history_length=self.hparams.imputation_history_length,
                                             batch_shape=X.shape,
                                             dtype=X.dtype if not bool_data else torch.bool)

        for t in tqdm(range(self.hparams.num_iterations), desc='Running IRWG'):
            if history is not None and t > self.hparams.no_historic_proposals_iterations:
                # H B D
                hist_proposals = history.sample_history(self.hparams.num_historical_proposals)
                hist_proposals = rearrange(hist_proposals, 'h b ... -> b h ...').to(X.device)
                X = torch.concat([X, hist_proposals], dim=1)
                del hist_proposals

            X_shape = X.shape

            if self.hparams.importance_weights == 'smis':
                log_weights, logits = self.compute_log_ratio(X, M)
            elif self.hparams.importance_weights == 'dmis':
                log_weights, logits = self.compute_dmis_log_weights(X, M)
            else:
                raise NotImplementedError()


            log_weights = log_weights.unsqueeze(0)
            logits = logits.unsqueeze(0)
            logits, _ = gr_resampling(log_weights, generator_params=logits, resampling_method='multinomial')
            logits_shape = logits.shape
            logits = logits.reshape(-1, *logits_shape[self.hparams.data_channel_dim-1:])
            mis_distr = self.vae.decoder_output(logits)

            # Sample missing values
            X_m = mis_distr.sample()
            X_m = X_m.reshape(*X_shape)

            # Remove dimensions of augmentations
            X_m = X_m[:, :K, ...] # Ideally this should be random but with Multinomial sampler it won't matter

            # Set imputed missing values
            X = X.reshape(*X_shape)
            X = X[:, :K, ...]
            X = X*M_expanded + X_m*(~M_expanded)

            if history is not None:
                history.enqueue_batch((X.cpu().bool() if bool_data else X.cpu()))

            outputs = {'X': X}
            yield outputs

class NVAELightningMetropolisWithinGibbs(NVAELightning):
    def __init__(self,
                 num_pseudo_warmup_iterations: int,
                 num_mwg_iterations: int,
                 *args,
                 **kwargs
                ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @property
    def max_iterations(self):
        return self.hparams.num_iterations

    def sampler(self, X, M):
        K = X.shape[1]

        M_expanded = expand_M_dim(M, data_channel_dim=self.hparams.data_channel_dim)
        M_not_expanded = ~M_expanded

        def check_if_bool(X):
            return ((X == 0).sum() + (X == 1).sum()) == X.numel()
        bool_data = check_if_bool(X)

        X_shape = X.shape
        for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations), desc='Running Pseudo-Gibbs warmup'):
            X = X.reshape(np.prod(X_shape[:self.hparams.data_channel_dim]), X_shape[self.hparams.data_channel_dim], *self.hparams.img_shape)
            logits, _, log_p, _, _, _, all_z0, _ = self.vae(X, q_t=self.hparams.temp_q, p_t=self.hparams.temp_p)
            mis_distr = self.vae.decoder_output(logits)

            # Sample missing values
            X_m = mis_distr.sample()
            X_m = X_m.reshape(*X_shape)

            # Set imputed missing values
            X = X.reshape(*X_shape)
            X = X*M_expanded + X_m*M_not_expanded

            outputs = {'X': X,
                       'acceptance': 1.}
            yield outputs

        Z0 = all_z0
        # Z = all_z
        mis_params_old = logits
        Z_old_prior_logprob = log_p
        X_old_logprob = torch.sum(mis_distr.log_prob(X.reshape((np.prod(X_shape[:self.hparams.data_channel_dim]),) + (X_shape[self.hparams.data_channel_dim],) + tuple(self.hparams.img_shape))), dim=[-3,-2,-1])

        for t in tqdm(range(self.hparams.num_pseudo_warmup_iterations, self.hparams.num_pseudo_warmup_iterations+self.hparams.num_mwg_iterations), desc='Running MWG'):
            X = X.reshape(np.prod(X_shape[:self.hparams.data_channel_dim]), X_shape[self.hparams.data_channel_dim], *self.hparams.img_shape)

            logits_new, log_q_new, log_p_new, _, _, all_q, all_z0_new, _ = self.vae(X, q_t=self.hparams.temp_q, p_t=self.hparams.temp_p)

            # Eval log-prob of previous Z0 on new q
            log_q_old = 0.
            for q, z0 in zip(all_q, Z0):
                # NOTE: this assumes additive flows (the jacobian of the transformation is 0)
                log_q_old += torch.sum(q.log_prob(z0), dim=[1, 2, 3])

            mis_distr = self.vae.decoder_output(logits_new)
            X_new_logprob = torch.sum(mis_distr.log_prob(X.reshape((np.prod(X_shape[:self.hparams.data_channel_dim]),) + (X_shape[self.hparams.data_channel_dim],) + tuple(self.hparams.img_shape))), dim=[-3,-2,-1])

            # Compute MH acceptance criterion
            log_accept = (X_new_logprob + log_p_new) - (X_old_logprob + Z_old_prior_logprob) + (log_q_old - log_q_new)

            acceptance_prob = torch.exp(log_accept)

            acceptance_samples = torch.rand_like(acceptance_prob)
            accepted = acceptance_samples < acceptance_prob

            # Update accepted
            accepted_not = ~accepted
            new_Z0 = []
            for z0_old, z0_new in zip(Z0, all_z0_new):
                new_Z0.append(z0_old*accepted_not.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + z0_new*accepted.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            Z0 = new_Z0

            mis_params_old = mis_params_old*accepted_not.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + logits_new*accepted.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mis_distr = self.vae.decoder_output(mis_params_old)

            # Sample missing values
            X_m = mis_distr.sample()
            X_m = X_m.reshape(*X_shape)

            # Set imputed missing values
            X = X.reshape(*X_shape)
            X = X*M_expanded + X_m*M_not_expanded

            X_old_logprob = torch.sum(mis_distr.log_prob(X.reshape((np.prod(X_shape[:self.hparams.data_channel_dim]),) + (X_shape[self.hparams.data_channel_dim],) + tuple(self.hparams.img_shape))), dim=[-3,-2,-1])

            outputs = {
                'X': X,
                'acceptance': accepted.float().mean()
            }

            yield outputs






