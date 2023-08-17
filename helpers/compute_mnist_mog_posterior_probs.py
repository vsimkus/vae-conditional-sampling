import math
import os
import os.path
from collections import defaultdict

import numpy as np
import scipy
import torch
from einops import asnumpy, rearrange, repeat
from tqdm import trange

from fit_mogs_on_imputations import eval_mogs, eval_mogs_diag
from irwg.models.resnet_classifier import ResNetClassifier
from irwg.models.vae_resnet import ResNetEncoder
from irwg.models.vae import VAE
from irwg.data.mnist_gmm import MNIST_GMM, MNIST_GMM_customtest, sample_mog
from irwg.utils.mog_utils import (compute_conditional_mog_parameters,
                                  compute_joint_p_c_given_xo_xm,
                                  compute_kl_div_for_sparse_mogs,
                                  compute_kl_div_for_sparse_mogs_perdatapoint,
                                  compute_kldivs_between_categoricals,
                                  sample_sparse_mog,
                                  compute_p_z_given_xo_from_imps,
                                  supervised_mog_diag_fit_using_groundtruth_z,
                                  supervised_mog_fit_using_groundtruth_z)

exp_dir = './logs/mnist_gmm/imputation/samples/{}/seed_m{}_d{}/lightning_logs/version_{}/'
# exp_dir = './logs/mnist_gmm/imputation/samples1/{}/seed_m{}_d{}/lightning_logs/version_{}/'

def load_last_nstep_imputations_from_experiment(path, last_n_steps, load_every_nth_step):
    imp_files = [f for f in os.listdir(path) if 'imputations_' in f and f.endswith('.npz')]
    batch_idxs = [int(f.split('imputations_')[1].split('.npz')[0]) for f in imp_files]
    imp_files = [f for _, f in sorted(zip(batch_idxs, imp_files))]

    if len(imp_files) == 0:
        raise FileNotFoundError()

    data = defaultdict(list)
    if last_n_steps == -1:
        last_n_steps = 0
    for f in imp_files:
        f = np.load(os.path.join(path, f))
        keys = list(f.keys())
        for k in keys:
            if k == 'imputations':
                data[k].append(f[k][-last_n_steps::load_every_nth_step])
            else:
                data[k].append(f[k])

    for k in list(data.keys()):
        if k == 'imputations':
            data[k] = np.concatenate(data[k], axis=1)
            data[k] = rearrange(data[k], 't b k ... -> b k t ...')
        else:
            data[k] = np.concatenate(data[k], axis=0)

    return data

def load_last_nstep_imputations_from_experiment_irwg_resampled(path, last_n_steps, load_every_nth_step):
    imp_files = [f for f in os.listdir(path) if 'irwg_imputations_after_final_resampling_' in f and f.endswith('.npz')]
    batch_idxs = [int(f.split('irwg_imputations_after_final_resampling_')[1].split('.npz')[0]) for f in imp_files]
    imp_files = [f for _, f in sorted(zip(batch_idxs, imp_files))]

    if len(imp_files) == 0:
        raise FileNotFoundError()

    data = defaultdict(list)
    if last_n_steps == -1:
        last_n_steps = 0
    for f in imp_files:
        f = np.load(os.path.join(path, f))
        keys = list(f.keys())
        for k in keys:
            if k == 'imputations':
                data[k].append(f[k][-last_n_steps::load_every_nth_step])
            else:
                data[k].append(f[k])

    for k in list(data.keys()):
        if k == 'imputations':
            data[k] = np.concatenate(data[k], axis=1)
            data[k] = rearrange(data[k], 't b k ... -> b k t ...')
        else:
            data[k] = np.concatenate(data[k], axis=0)

    return data

def process_dir(log_dir, params, batch_size, *, fit_iterations, irwg_use_resampled_imps=False, true_datapoints=None):
    # cond_mog_params = load_cond_mog_params_from_experiment(log_dir)
    if irwg_use_resampled_imps:
        imps_ = load_last_nstep_imputations_from_experiment_irwg_resampled(log_dir, -1, 1)
    else:
        imps_ = load_last_nstep_imputations_from_experiment(log_dir, -1, 1)

    print(f'Processing {log_dir}')
    # mog_true_cond_comp_log_probs = cond_mog_params['mog_true_cond_comp_log_probs'].astype(np.float32)

    # fit_iterations=[100, 500, 1000, 5000, 10000]

    all_klfow = []
    all_klrev = []
    all_jsd = []

    # all_mog_klfow = []
    # all_mog_klrev = []
    # all_mog_jsd = []
    for final_iter in fit_iterations:
        masks = imps_['masks']
        imps = imps_['imputations']

        if true_datapoints is not None:
            true_mask = ~true_datapoints.isnan()
            assert torch.all(torch.tensor(masks).squeeze(1) == true_mask).item()
            imps_temp = rearrange(imps, 'b k t d -> b (t k) d')
            imps_temp = torch.tensor(imps_temp)
            true_mask_temp = repeat(true_mask, 'b d -> b k d', k=imps_temp.shape[1])
            imps_temp[~true_mask_temp] = 0.
            true_datapoints_temp = true_datapoints.clone()
            true_datapoints_temp[~true_mask] = 0.
            true_datapoints_temp = true_datapoints_temp.unsqueeze(1)
            assert torch.all(imps_temp == true_datapoints_temp).item()
            del imps_temp
            del true_datapoints_temp
            del true_mask_temp

        # Clip imputations
        # imps = data.clip_unconstrained_samples_to_unconstraineddatarange(imps)
        imps = imps[:, :, :final_iter]
        imps = rearrange(imps, 'b k t d -> b (t k) d')
        B = imps.shape[0]
        K = imps.shape[1]

        log_prob_c_given_ximps = []
        mog_cond_log_probs_from_imps = []
        # mog_cond_means_from_imps = []
        # mog_cond_covs_from_imps = []

        mog_true_cond_comp_log_probs = []
        # mog_true_cond_means = []
        # mog_true_cond_covs = []
        for b in trange(math.ceil(B/batch_size)):
            imps_b = imps[b*batch_size:min((b+1)*batch_size, B)]
            masks_b = masks[b*batch_size:min((b+1)*batch_size, B)]

            # out_b = supervised_mog_fit_using_groundtruth_z(imps_b, masks_b, params, K=K)
            # mog_true_cond_comp_log_probs.append(out_b['mog_true_cond_comp_log_probs_b'])
            # mog_true_cond_means.append(out_b['mog_true_cond_means_b'])
            # mog_true_cond_covs.append(out_b['mog_true_cond_covs_b'])

            # log_prob_c_given_ximps.append(out_b['log_prob_c_given_ximps_b'])
            # mog_cond_log_probs_from_imps.append(out_b['mog_cond_log_probs_from_imps_b'])

            # mog_cond_means_from_imps.append(out_b['cond_means_from_imps_b'])
            # mog_cond_covs_from_imps.append(out_b['cond_covs_from_imps_b'])

            mog_true_cond_comp_log_probs_b, _, _ = compute_conditional_mog_parameters(
                torch.tensor(imps_b[:, 0, :]), torch.tensor(masks_b).squeeze(1),
                params['comp_logits'],
                params['means'],
                params['covs'])
            mog_true_cond_comp_log_probs.append(mog_true_cond_comp_log_probs_b)

            out_b = compute_p_z_given_xo_from_imps(imps_b, params, K=K)
            mog_cond_log_probs_from_imps.append(out_b['mog_cond_log_probs_from_imps_b'])
            log_prob_c_given_ximps.append(out_b['log_prob_c_given_ximps_b'])

        log_prob_c_given_ximps = torch.cat(log_prob_c_given_ximps, dim=0)
        mog_cond_log_probs_from_imps = torch.cat(mog_cond_log_probs_from_imps, dim=0)
        # mog_cond_means_from_imps = torch.cat(mog_cond_means_from_imps, dim=0)
        # mog_cond_covs_from_imps = torch.cat(mog_cond_covs_from_imps, dim=0)

        mog_true_cond_comp_log_probs = torch.cat(mog_true_cond_comp_log_probs, dim=0)
        # mog_true_cond_means = torch.cat(mog_true_cond_means, dim=0)
        # mog_true_cond_covs = torch.cat(mog_true_cond_covs, dim=0)

        # Compute Divergences between p(z | xo) and q(z | xo)
        kl_fow, kl_rev, jsd = compute_kldivs_between_categoricals(torch.tensor(mog_true_cond_comp_log_probs), mog_cond_log_probs_from_imps)

        all_klfow.append(kl_fow)
        all_klrev.append(kl_rev)
        all_jsd.append(jsd)

        # # Compute Divergences between p(xm | xo) and q(xm | xo)
        # class HPARAMS:
        #     def __init__(self, **entries):
        #         self.__dict__.update(entries)
        # hparams = HPARAMS(**{
        #     'use_batched_per_datapoint_computation': True,
        #     'num_kl_samples': 10000,
        # })
        # # print(final_iter)
        # mog_klfow, mog_klrev, mog_jsd = eval_mogs(torch.tensor(masks).squeeze(1),
        #                                           hparams=hparams,
        #       true_cond_comp_log_probs=mog_true_cond_comp_log_probs,
        #       true_cond_means=mog_true_cond_means,
        #       true_cond_covs=mog_true_cond_covs,
        #       comp_log_probs=mog_cond_log_probs_from_imps,
        #       means=mog_cond_means_from_imps,
        #       covs=mog_cond_covs_from_imps)
        # all_mog_klfow.append(mog_klfow)
        # all_mog_klrev.append(mog_klrev)
        # all_mog_jsd.append(mog_jsd)

    all_kldivs_fow = torch.stack(all_klfow, dim=1)
    all_kldivs_rev = torch.stack(all_klrev, dim=1)
    all_jsds = torch.stack(all_jsd, dim=1)

    # all_mog_kldivs_fow = torch.stack(all_mog_klfow, dim=1)
    # all_mog_kldivs_rev = torch.stack(all_mog_klrev, dim=1)
    # all_mog_jsds = torch.stack(all_mog_jsd, dim=1)

    if irwg_use_resampled_imps:
        np.savez_compressed(os.path.join(log_dir, 'mog_comp_log_probs_irwg_resampled.npz'),
                    mog_true_cond_comp_log_probs=asnumpy(mog_true_cond_comp_log_probs),
                    mog_cond_log_probs_from_imps=asnumpy(mog_cond_log_probs_from_imps),
                    kl_fow=asnumpy(all_kldivs_fow),
                    kl_rev=asnumpy(all_kldivs_rev),
                    jsd=asnumpy(all_jsds),

                    # # Added later
                    # mog_kl_fow=asnumpy(all_mog_kldivs_fow),
                    # mog_kl_rev=asnumpy(all_mog_kldivs_rev),
                    # mog_jsd=asnumpy(all_mog_jsds),
                    )

    else:
        np.savez_compressed(os.path.join(log_dir, 'mog_comp_log_probs.npz'),
                        mog_true_cond_comp_log_probs=asnumpy(mog_true_cond_comp_log_probs),
                        mog_cond_log_probs_from_imps=asnumpy(mog_cond_log_probs_from_imps),
                        kl_fow=asnumpy(all_kldivs_fow),
                        kl_rev=asnumpy(all_kldivs_rev),
                        jsd=asnumpy(all_jsds),

                        # # Added later
                        # mog_kl_fow=asnumpy(all_mog_kldivs_fow),
                        # mog_kl_rev=asnumpy(all_mog_kldivs_rev),
                        # mog_jsd=asnumpy(all_mog_jsds),
                        )

def process_dir_diag_gauss(log_dir, params, batch_size, *, fit_iterations, irwg_use_resampled_imps=False, true_datapoints=None, dataset=None,
                           min_max_vals=(None, None), clamp_imps_effectivesamples=-1):
    # cond_mog_params = load_cond_mog_params_from_experiment(log_dir)
    if irwg_use_resampled_imps:
        imps_ = load_last_nstep_imputations_from_experiment_irwg_resampled(log_dir, -1, 1)
    else:
        imps_ = load_last_nstep_imputations_from_experiment(log_dir, -1, 1)

    print(f'Processing {log_dir}')
    # mog_true_cond_comp_log_probs = cond_mog_params['mog_true_cond_comp_log_probs'].astype(np.float32)

    # p(c | xo)
    all_klfow = []
    all_klrev = []
    all_jsd = []

    # p(xm | xo)
    all_mog_klfow = []
    all_mog_klrev = []
    all_mog_jsd = []

    # p(xm, c | xo)
    all_joint_klfow = []
    all_joint_klrev = []

    all_wass_d = []

    for final_iter in fit_iterations:
        masks = imps_['masks']
        imps = imps_['imputations']

        if true_datapoints is not None:
            true_mask = ~true_datapoints.isnan()
            assert torch.all(torch.tensor(masks).squeeze(1) == true_mask).item()
            imps_temp = rearrange(imps, 'b k t d -> b (t k) d')
            imps_temp = torch.tensor(imps_temp)
            true_mask_temp = repeat(true_mask, 'b d -> b k d', k=imps_temp.shape[1])
            imps_temp[~true_mask_temp] = 0.
            true_datapoints_temp = true_datapoints.clone()
            true_datapoints_temp[~true_mask] = 0.
            true_datapoints_temp = true_datapoints_temp.unsqueeze(1)
            assert torch.all(imps_temp == true_datapoints_temp).item()
            del imps_temp
            del true_datapoints_temp
            del true_mask_temp

        # Clip imputations
        # imps = data.clip_unconstrained_samples_to_unconstraineddatarange(imps)
        imps = imps[:, :, :final_iter]
        imps = rearrange(imps, 'b k t d -> b (t k) d')
        # TEMP
        imps = torch.clamp(torch.tensor(imps), min=min_max_vals[0], max=min_max_vals[1])


        B = imps.shape[0]
        K = imps.shape[1]

        log_prob_c_given_ximps = []
        mog_cond_log_probs_from_imps = []
        mog_cond_means_from_imps = []
        mog_cond_vars_from_imps = []

        mog_true_cond_comp_log_probs = []
        mog_true_cond_means = []
        mog_true_cond_vars = []
        for b in trange(math.ceil(B/batch_size)):
            imps_b = imps[b*batch_size:min((b+1)*batch_size, B)]
            masks_b = masks[b*batch_size:min((b+1)*batch_size, B)]

            out_b = supervised_mog_diag_fit_using_groundtruth_z(imps_b, masks_b, params, K=K, min_max_vals=min_max_vals,
                                                                clamp_imps_effectivesamples=clamp_imps_effectivesamples)
            mog_true_cond_comp_log_probs.append(out_b['mog_true_cond_comp_log_probs_b'])
            mog_true_cond_means.append(out_b['mog_true_cond_means_b'])
            mog_true_cond_vars.append(out_b['mog_true_cond_vars_b'])

            log_prob_c_given_ximps.append(out_b['log_prob_c_given_ximps_b'])
            mog_cond_log_probs_from_imps.append(out_b['mog_cond_log_probs_from_imps_b'])

            mog_cond_means_from_imps.append(out_b['cond_means_from_imps_b'])
            mog_cond_vars_from_imps.append(out_b['cond_vars_from_imps_b'])

        log_prob_c_given_ximps = torch.cat(log_prob_c_given_ximps, dim=0)
        mog_cond_log_probs_from_imps = torch.cat(mog_cond_log_probs_from_imps, dim=0)
        mog_cond_means_from_imps = torch.cat(mog_cond_means_from_imps, dim=0)
        mog_cond_vars_from_imps = torch.cat(mog_cond_vars_from_imps, dim=0)

        mog_true_cond_comp_log_probs = torch.cat(mog_true_cond_comp_log_probs, dim=0)
        mog_true_cond_means = torch.cat(mog_true_cond_means, dim=0)
        mog_true_cond_vars = torch.cat(mog_true_cond_vars, dim=0)

        # Compute Divergences between p(z | xo) and q(z | xo)
        kl_fow, kl_rev, jsd = compute_kldivs_between_categoricals(torch.tensor(mog_true_cond_comp_log_probs), mog_cond_log_probs_from_imps)

        all_klfow.append(kl_fow)
        all_klrev.append(kl_rev)
        all_jsd.append(jsd)

        # Compute Divergences between p(xm, c | xo) and q(xm, c | xo)
        individual_gauss_fowkl, individual_gauss_revkl = compare_diag_Gaussians_per_component(
            torch.tensor(masks).squeeze(1),
                true_cond_means=mog_true_cond_means,
                true_cond_vars=mog_true_cond_vars,
                means=mog_cond_means_from_imps,
                vars=mog_cond_vars_from_imps,)
        joint_klfow = (individual_gauss_fowkl*mog_true_cond_comp_log_probs.exp()).sum(-1) + kl_fow
        joint_klrev = (individual_gauss_revkl*mog_cond_log_probs_from_imps.exp()).sum(-1) + kl_rev
        all_joint_klfow.append(joint_klfow)
        all_joint_klrev.append(joint_klrev)

        wasserstein_d = compare_diag_Gaussian_per_component_using_Wasserstein(torch.tensor(masks).squeeze(1),
                                                                          true_cond_means=mog_true_cond_means,
                                                                          true_cond_vars=mog_true_cond_vars,
                                                                          means=mog_cond_means_from_imps,
                                                                          vars=mog_cond_vars_from_imps,)
        wasserstein_d_compweightavg = (wasserstein_d*mog_true_cond_comp_log_probs.exp()).sum(-1)
        all_wass_d.append(wasserstein_d_compweightavg)

        # Compute Divergences between p(xm | xo) and q(xm | xo)
        class HPARAMS:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        hparams = HPARAMS(**{
            'use_batched_per_datapoint_computation': True,
            'num_kl_samples': 10000,
        })
        mog_klfow, mog_klrev, mog_jsd = eval_mogs_diag(torch.tensor(masks).squeeze(1),
                                                       hparams=hparams,
              true_cond_comp_log_probs=mog_true_cond_comp_log_probs,
              true_cond_means=mog_true_cond_means,
              true_cond_vars=mog_true_cond_vars,
              comp_log_probs=mog_cond_log_probs_from_imps,
              means=mog_cond_means_from_imps,
              vars=mog_cond_vars_from_imps)
        all_mog_klfow.append(mog_klfow)
        all_mog_klrev.append(mog_klrev)
        all_mog_jsd.append(mog_jsd)

    all_kldivs_fow = torch.stack(all_klfow, dim=1)
    all_kldivs_rev = torch.stack(all_klrev, dim=1)
    all_jsds = torch.stack(all_jsd, dim=1)

    all_mog_kldivs_fow = torch.stack(all_mog_klfow, dim=1)
    all_mog_kldivs_rev = torch.stack(all_mog_klrev, dim=1)
    all_mog_jsds = torch.stack(all_mog_jsd, dim=1)

    all_joint_klfow = torch.stack(all_joint_klfow, dim=1)
    all_joint_klrev = torch.stack(all_joint_klrev, dim=1)

    all_wass_d = torch.stack(all_wass_d, dim=1)

    if irwg_use_resampled_imps:
       np.savez_compressed(os.path.join(log_dir, 'diag_mog_comp_log_probs_irwg_resampled.npz'),
                    mog_true_cond_comp_log_probs=asnumpy(mog_true_cond_comp_log_probs),
                    mog_cond_log_probs_from_imps=asnumpy(mog_cond_log_probs_from_imps),
                    kl_fow=asnumpy(all_kldivs_fow),
                    kl_rev=asnumpy(all_kldivs_rev),
                    jsd=asnumpy(all_jsds),

                    # Added later
                    mog_kl_fow=asnumpy(all_mog_kldivs_fow),
                    mog_kl_rev=asnumpy(all_mog_kldivs_rev),
                    mog_jsd=asnumpy(all_mog_jsds),

                    joint_kl_fow=asnumpy(all_joint_klfow),
                    joint_kl_rev=asnumpy(all_joint_klrev),

                    compweightavg_wasserstein_d=asnumpy(all_wass_d),
                    )
    else:
        np.savez_compressed(os.path.join(log_dir, 'diag_mog_comp_log_probs.npz'),
                        mog_true_cond_comp_log_probs=asnumpy(mog_true_cond_comp_log_probs),
                        mog_cond_log_probs_from_imps=asnumpy(mog_cond_log_probs_from_imps),
                        kl_fow=asnumpy(all_kldivs_fow),
                        kl_rev=asnumpy(all_kldivs_rev),
                        jsd=asnumpy(all_jsds),

                        # Added later
                        mog_kl_fow=asnumpy(all_mog_kldivs_fow),
                        mog_kl_rev=asnumpy(all_mog_kldivs_rev),
                        mog_jsd=asnumpy(all_mog_jsds),

                        joint_kl_fow=asnumpy(all_joint_klfow),
                        joint_kl_rev=asnumpy(all_joint_klrev),

                        compweightavg_wasserstein_d=asnumpy(all_wass_d),
                        )

def process_dir_fid(log_dir, params, batch_size, *, filename, fit_iterations, inception_model, irwg_use_resampled_imps=False, true_datapoints=None, dataset=None,
                    min_max_vals=(None, None)):
    if irwg_use_resampled_imps:
        imps_ = load_last_nstep_imputations_from_experiment_irwg_resampled(log_dir, -1, 1)
    else:
        imps_ = load_last_nstep_imputations_from_experiment(log_dir, -1, 1)

    print(f'Processing {log_dir}')

    num_ref_samples = imps_['imputations'].shape[1]*imps_['imputations'].shape[2]

    all_fid_scores = []
    for final_iter in fit_iterations:
        masks = imps_['masks']
        imps = imps_['imputations']

        if true_datapoints is not None:
            true_mask = ~true_datapoints.isnan()
            assert torch.all(torch.tensor(masks).squeeze(1) == true_mask).item()
            imps_temp = rearrange(imps, 'b k t d -> b (t k) d')
            imps_temp = torch.tensor(imps_temp)
            true_mask_temp = repeat(true_mask, 'b d -> b k d', k=imps_temp.shape[1])
            imps_temp[~true_mask_temp] = 0.
            true_datapoints_temp = true_datapoints.clone()
            true_datapoints_temp[~true_mask] = 0.
            true_datapoints_temp = true_datapoints_temp.unsqueeze(1)
            assert torch.all(imps_temp == true_datapoints_temp).item()
            del imps_temp
            del true_datapoints_temp
            del true_mask_temp

        # Clip imputations
        # imps = data.clip_unconstrained_samples_to_unconstraineddatarange(imps)
        imps = imps[:, :, :final_iter]
        imps = rearrange(imps, 'b k t d -> b (t k) d')
        imps = torch.tensor(imps)
        # TEMP
        imps = torch.clamp(imps, min=min_max_vals[0], max=min_max_vals[1])

        B = imps.shape[0]
        K = imps.shape[1]

        fid_scores = []
        for b in trange(math.ceil(B/batch_size)):
            imps_b = imps[b*batch_size:min((b+1)*batch_size, B)]
            masks_b = masks[b*batch_size:min((b+1)*batch_size, B)]
            masks_b = torch.tensor(masks_b)

            mog_true_cond_comp_log_probs_b, mog_true_cond_means_b, mog_true_cond_covs_b = compute_conditional_mog_parameters(
                                                imps_b[:, 0, :], masks_b.squeeze(1),
                                                params['comp_logits'],
                                                params['means'],
                                                params['covs'])

            X = sample_sparse_mog(num_samples=num_ref_samples, M=masks_b.squeeze(1),
                          comp_log_probs=mog_true_cond_comp_log_probs_b,
                          cond_means=mog_true_cond_means_b,
                          cond_covs=mog_true_cond_covs_b,
                          sampling_batch_size=10000,
                          )
            X_ref = X*(~masks_b) + imps_b[:, 0][:, None]*masks_b
            refs_b = X_ref
            # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
            #     ref_b = dataset.clip_unconstrained_samples_to_unconstraineddatarange(ref_b)

            # Compute FID scores
            fid_scores_b = compute_fid_score(refs_b, imps_b, inception_model=inception_model)

            fid_scores.append(fid_scores_b)
        fid_scores = torch.cat(fid_scores, dim=0)

        all_fid_scores.append(fid_scores)
    all_fid_scores = torch.stack(all_fid_scores, dim=1)

    # masks = torch.tensor(imps_['masks'][:, 0, :])
    # imps = torch.tensor(imps_['imputations'][:, 0, 0, :])
    # imps = imps.clone()
    # imps[~masks] = float('nan')
    # ref_fid = eval_fid_on_ground_truth_samples(imps, masks, params, batch_size, inception_model=inception_model)

    if irwg_use_resampled_imps:
       np.savez_compressed(os.path.join(log_dir, f'{filename}_irwg_resampled.npz'),
                        fid_scores=asnumpy(all_fid_scores)
                        )
    else:
        np.savez_compressed(os.path.join(log_dir, f'{filename}.npz'),
                        fid_scores=asnumpy(all_fid_scores)
                        )

def fit_and_eval_on_ground_truth_samples(datapoints, masks, params, batch_size):
    num_samples = 10000
    datapoints = datapoints.unsqueeze(1)
    masks = masks.unsqueeze(1)
    datapoints[~masks] = 0.

    # Compute true conditional MoG parameters
    mog_true_cond_comp_log_probs_all, mog_true_cond_means_all, mog_true_cond_covs_all = compute_conditional_mog_parameters(
        torch.tensor(datapoints[:, 0, :]), torch.tensor(masks).squeeze(1),
        params['comp_logits'],
        params['means'],
        params['covs'])

    B = datapoints.shape[0]
    K = num_samples

    log_prob_c_given_ximps = []
    mog_cond_log_probs_from_imps = []
    mog_cond_means_from_imps = []
    mog_cond_covs_from_imps = []

    mog_true_cond_comp_log_probs = []
    mog_true_cond_means = []
    mog_true_cond_covs = []
    for b in trange(math.ceil(B/batch_size)):
        masks_b = masks[b*batch_size:min((b+1)*batch_size, B)]
        datapoints_b = datapoints[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_comp_log_probs_b = mog_true_cond_comp_log_probs_all[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_means_b = mog_true_cond_means_all[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_covs_b = mog_true_cond_covs_all[b*batch_size:min((b+1)*batch_size, B)]

        X = sample_sparse_mog(num_samples=num_samples, M=masks_b.squeeze(1),
                          comp_log_probs=mog_true_cond_comp_log_probs_b,
                          cond_means=mog_true_cond_means_b,
                          cond_covs=mog_true_cond_covs_b,
                          sampling_batch_size=10000,
                          )
        X_imp = X*(~masks_b) + datapoints_b*masks_b

        imps_b = X_imp
        # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
        #     imps_b = dataset.clip_unconstrained_samples_to_unconstraineddatarange(imps_b)

        # Full rank covariance
        out_b = supervised_mog_fit_using_groundtruth_z(imps_b, masks_b, params, K=K)
        mog_true_cond_comp_log_probs.append(out_b['mog_true_cond_comp_log_probs_b'])
        mog_true_cond_means.append(out_b['mog_true_cond_means_b'])
        mog_true_cond_covs.append(out_b['mog_true_cond_covs_b'])

        log_prob_c_given_ximps.append(out_b['log_prob_c_given_ximps_b'])
        mog_cond_log_probs_from_imps.append(out_b['mog_cond_log_probs_from_imps_b'])

        mog_cond_means_from_imps.append(out_b['cond_means_from_imps_b'])
        mog_cond_covs_from_imps.append(out_b['cond_covs_from_imps_b'])

    log_prob_c_given_ximps = torch.cat(log_prob_c_given_ximps, dim=0)
    mog_cond_log_probs_from_imps = torch.cat(mog_cond_log_probs_from_imps, dim=0)
    mog_cond_means_from_imps = torch.cat(mog_cond_means_from_imps, dim=0)
    mog_cond_covs_from_imps = torch.cat(mog_cond_covs_from_imps, dim=0)

    mog_true_cond_comp_log_probs = torch.cat(mog_true_cond_comp_log_probs, dim=0)
    mog_true_cond_means = torch.cat(mog_true_cond_means, dim=0)
    mog_true_cond_covs = torch.cat(mog_true_cond_covs, dim=0)

    # Compute Divergences between p(z | xo) and q(z | xo)
    kl_fow, kl_rev, jsd = compute_kldivs_between_categoricals(torch.tensor(mog_true_cond_comp_log_probs), mog_cond_log_probs_from_imps)
    print(f'--------p(c| xo)--------\nkl_fow: {kl_fow}, \nkl_rev: {kl_rev}, \njsd: {jsd}')

    # Compute Divergences between p(xm | xo) and q(xm | xo)
    individual_gauss_fowkl, individual_gauss_revkl = compare_Gaussians_per_component(
        torch.tensor(masks).squeeze(1),
            true_cond_means=mog_true_cond_means,
            true_cond_covs=mog_true_cond_covs,
            means=mog_cond_means_from_imps,
            covs=mog_cond_covs_from_imps,)
    print(f'--------p(xm| xo, c)--------\nkl_fow: {individual_gauss_fowkl}\nkl_rev: {individual_gauss_revkl}')
    joint_klfow = (individual_gauss_fowkl*mog_true_cond_comp_log_probs.exp()).sum(-1) + kl_fow
    joint_klrev = (individual_gauss_revkl*mog_cond_log_probs_from_imps.exp()).sum(-1) + kl_rev
    print(f'--------p(xm, c| xo)--------\nkl_fow: {joint_klfow}\nkl_rev: {joint_klrev}')

    class HPARAMS:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    hparams = HPARAMS(**{
        'use_batched_per_datapoint_computation': True,
        'num_kl_samples': 10000,
    })
    mog_klfow, mog_klrev, mog_jsd = eval_mogs(torch.tensor(masks).squeeze(1),
                                                hparams=hparams,
            true_cond_comp_log_probs=mog_true_cond_comp_log_probs,
            true_cond_means=mog_true_cond_means,
            true_cond_covs=mog_true_cond_covs,
            comp_log_probs=mog_cond_log_probs_from_imps,
            means=mog_cond_means_from_imps,
            covs=mog_cond_covs_from_imps,
            validate_args=False)
    print(f'--------p(xm | xo)--------\nkl_fow: {mog_klfow}\nkl_rev: {mog_klrev}\njsd: {mog_jsd}')

def fit_diag_and_eval_on_ground_truth_samples(datapoints, masks, params, batch_size):
    num_samples = 10000
    datapoints = datapoints.unsqueeze(1)
    masks = masks.unsqueeze(1)
    datapoints[~masks] = 0.

    # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
    #     datapoints = dataset.clip_unconstrained_samples_to_unconstraineddatarange(datapoints)

    # Compute true conditional MoG parameters
    mog_true_cond_comp_log_probs_all, mog_true_cond_means_all, mog_true_cond_covs_all = compute_conditional_mog_parameters(
        torch.tensor(datapoints[:, 0, :]), torch.tensor(masks).squeeze(1),
        params['comp_logits'],
        params['means'],
        params['covs'])

    B = datapoints.shape[0]
    K = num_samples

    log_prob_c_given_ximps = []
    mog_cond_log_probs_from_imps = []
    mog_cond_means_from_imps = []
    mog_cond_vars_from_imps = []

    mog_true_cond_comp_log_probs = []
    mog_true_cond_means = []
    mog_true_cond_vars = []
    for b in trange(math.ceil(B/batch_size)):
        masks_b = masks[b*batch_size:min((b+1)*batch_size, B)]
        datapoints_b = datapoints[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_comp_log_probs_b = mog_true_cond_comp_log_probs_all[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_means_b = mog_true_cond_means_all[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_covs_b = mog_true_cond_covs_all[b*batch_size:min((b+1)*batch_size, B)]

        X = sample_sparse_mog(num_samples=num_samples, M=masks_b.squeeze(1),
                          comp_log_probs=mog_true_cond_comp_log_probs_b,
                          cond_means=mog_true_cond_means_b,
                          cond_covs=mog_true_cond_covs_b,
                          sampling_batch_size=10000,
                          )
        X_imp = X*(~masks_b) + datapoints_b*masks_b

        imps_b = X_imp
        # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
        #     imps_b = dataset.clip_unconstrained_samples_to_unconstraineddatarange(imps_b)

        # Full rank covariance
        out_b = supervised_mog_diag_fit_using_groundtruth_z(imps_b, masks_b, params, K=K)
        mog_true_cond_comp_log_probs.append(out_b['mog_true_cond_comp_log_probs_b'])
        mog_true_cond_means.append(out_b['mog_true_cond_means_b'])
        mog_true_cond_vars.append(out_b['mog_true_cond_vars_b'])

        log_prob_c_given_ximps.append(out_b['log_prob_c_given_ximps_b'])
        mog_cond_log_probs_from_imps.append(out_b['mog_cond_log_probs_from_imps_b'])

        mog_cond_means_from_imps.append(out_b['cond_means_from_imps_b'])
        mog_cond_vars_from_imps.append(out_b['cond_vars_from_imps_b'])

    log_prob_c_given_ximps = torch.cat(log_prob_c_given_ximps, dim=0)
    mog_cond_log_probs_from_imps = torch.cat(mog_cond_log_probs_from_imps, dim=0)
    mog_cond_means_from_imps = torch.cat(mog_cond_means_from_imps, dim=0)
    mog_cond_vars_from_imps = torch.cat(mog_cond_vars_from_imps, dim=0)

    mog_true_cond_comp_log_probs = torch.cat(mog_true_cond_comp_log_probs, dim=0)
    mog_true_cond_means = torch.cat(mog_true_cond_means, dim=0)
    mog_true_cond_vars = torch.cat(mog_true_cond_vars, dim=0)

    # Compute Divergences between p(z | xo) and q(z | xo)
    kl_fow, kl_rev, jsd = compute_kldivs_between_categoricals(torch.tensor(mog_true_cond_comp_log_probs), mog_cond_log_probs_from_imps)
    print(f'--------p(c| xo)--------\nkl_fow: {kl_fow}, \nkl_rev: {kl_rev}, \njsd: {jsd}')

    # Compute Divergences between p(xm | xo) and q(xm | xo)
    individual_gauss_fowkl, individual_gauss_revkl = compare_diag_Gaussians_per_component(
        torch.tensor(masks).squeeze(1),
            true_cond_means=mog_true_cond_means,
            true_cond_vars=mog_true_cond_vars,
            means=mog_cond_means_from_imps,
            vars=mog_cond_vars_from_imps,)
    print(f'--------p(xm| xo, c)--------\nkl_fow: {individual_gauss_fowkl}\nkl_rev: {individual_gauss_revkl}')
    joint_klfow = (individual_gauss_fowkl*mog_true_cond_comp_log_probs.exp()).sum(-1) + kl_fow
    joint_klrev = (individual_gauss_revkl*mog_cond_log_probs_from_imps.exp()).sum(-1) + kl_rev
    print(f'--------p(xm, c| xo)--------\nkl_fow: {joint_klfow}\nkl_rev: {joint_klrev}')

    wasserstein_d = compare_diag_Gaussian_per_component_using_Wasserstein(torch.tensor(masks).squeeze(1),
                                                                          true_cond_means=mog_true_cond_means,
                                                                          true_cond_vars=mog_true_cond_vars,
                                                                          means=mog_cond_means_from_imps,
                                                                          vars=mog_cond_vars_from_imps,)
    print('--------Wasserstein Distance p(xm | xo, c)--------\n', wasserstein_d)
    wasserstein_d_compweightavg = (wasserstein_d*mog_true_cond_comp_log_probs.exp()).sum(-1)
    print('--------Wasserstein Distance p(xm | xo, c) weighted average over c--------\n', wasserstein_d_compweightavg)


    class HPARAMS:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    hparams = HPARAMS(**{
        'use_batched_per_datapoint_computation': True,
        'num_kl_samples': 10000,
    })
    mog_klfow, mog_klrev, mog_jsd = eval_mogs_diag(torch.tensor(masks).squeeze(1),
                                                hparams=hparams,
            true_cond_comp_log_probs=mog_true_cond_comp_log_probs,
            true_cond_means=mog_true_cond_means,
            true_cond_vars=mog_true_cond_vars,
            comp_log_probs=mog_cond_log_probs_from_imps,
            means=mog_cond_means_from_imps,
            vars=mog_cond_vars_from_imps,
            validate_args=False)
    print(f'--------p(xm | xo)--------\nkl_fow: {mog_klfow}\nkl_rev: {mog_klrev}\njsd: {mog_jsd}')

def eval_fid_on_ground_truth_samples(datapoints, masks, params, batch_size, *, inception_model):
    inception_model = inception_model.eval()

    num_samples = 10000
    num_ref_samples = 10000
    datapoints = datapoints.unsqueeze(1)
    masks = masks.unsqueeze(1)
    datapoints[~masks] = 0.

    # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
    #     datapoints = dataset.clip_unconstrained_samples_to_unconstraineddatarange(datapoints)

    # Compute true conditional MoG parameters
    mog_true_cond_comp_log_probs_all, mog_true_cond_means_all, mog_true_cond_covs_all = compute_conditional_mog_parameters(
        torch.tensor(datapoints[:, 0, :]), torch.tensor(masks).squeeze(1),
        params['comp_logits'],
        params['means'],
        params['covs'])

    B = datapoints.shape[0]
    K = num_samples

    fid_scores = []
    for b in trange(math.ceil(B/batch_size)):
        masks_b = masks[b*batch_size:min((b+1)*batch_size, B)]
        datapoints_b = datapoints[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_comp_log_probs_b = mog_true_cond_comp_log_probs_all[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_means_b = mog_true_cond_means_all[b*batch_size:min((b+1)*batch_size, B)]
        mog_true_cond_covs_b = mog_true_cond_covs_all[b*batch_size:min((b+1)*batch_size, B)]

        X = sample_sparse_mog(num_samples=num_ref_samples, M=masks_b.squeeze(1),
                          comp_log_probs=mog_true_cond_comp_log_probs_b,
                          cond_means=mog_true_cond_means_b,
                          cond_covs=mog_true_cond_covs_b,
                          sampling_batch_size=10000,
                          )
        X_ref = X*(~masks_b) + datapoints_b*masks_b
        refs_b = X_ref
        # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
        #     ref_b = dataset.clip_unconstrained_samples_to_unconstraineddatarange(ref_b)

        X = sample_sparse_mog(num_samples=num_samples, M=masks_b.squeeze(1),
                          comp_log_probs=mog_true_cond_comp_log_probs_b,
                          cond_means=mog_true_cond_means_b,
                          cond_covs=mog_true_cond_covs_b,
                          sampling_batch_size=10000,
                          )
        X_imp = X*(~masks_b) + datapoints_b*masks_b
        imps_b = X_imp
        # if hasattr(dataset, 'clip_unconstrained_samples_to_unconstraineddatarange'):
        #     imps_b = dataset.clip_unconstrained_samples_to_unconstraineddatarange(imps_b)

        # Compute FID scores
        fid_scores_b = compute_fid_score(refs_b, imps_b, inception_model=inception_model)

        fid_scores.append(fid_scores_b)

    fid_scores = torch.cat(fid_scores, dim=0)

    print(f'FID scores\n{fid_scores}')

    return fid_scores

def compute_fid_score(refs, imps, *, inception_model):
    inception_model = inception_model.eval()

    B = refs.shape[0]
    fid_scores = []
    with torch.inference_mode():
        for b in range(B):
            refs_b = refs[b]
            imps_b = imps[b]

            if isinstance(inception_model, ResNetClassifier):
                _, feats_refs_b = inception_model(refs_b)
                _, feats_imps_b = inception_model(imps_b)
            elif isinstance(inception_model, ResNetEncoder):
                feats_refs_b = inception_model(refs_b)
                feats_imps_b = inception_model(imps_b)
            else:
                raise NotImplementedError(f'Unknown inception model type: {type(inception_model)}')

            ref_feat_mean, ref_feat_cov = torch.mean(feats_refs_b, dim=0), torch.cov(feats_refs_b.T)
            imps_feat_mean, imps_feat_cov = torch.mean(feats_imps_b, dim=0), torch.cov(feats_imps_b.T)

            # FID is equivalent to squared Wasserstein distance between Gaussians
            sqrt = scipy.linalg.sqrtm(imps_feat_cov @ ref_feat_cov)
            if sqrt.dtype in (np.complex64, np.complex128):
                sqrt = sqrt.real
            sqrt = torch.tensor(sqrt)
            fid = torch.norm(ref_feat_mean - imps_feat_mean, p=2)**2 + torch.trace(ref_feat_cov + imps_feat_cov - 2*sqrt)
            fid_scores_b = fid

            fid_scores.append(fid_scores_b)

    fid_scores = torch.stack(fid_scores, dim=0)

    return fid_scores


def compute_gaussian_kl_div(mean1, cov1, mean2, cov2):
    slog_det1 = np.linalg.slogdet(cov1)
    log_det1 = slog_det1[0] * slog_det1[1]
    slog_det2 = np.linalg.slogdet(cov2)
    log_det2 = slog_det2[0] * slog_det2[1]

    kl_div = (1/2) * (log_det2 - log_det1
                    - mean1.shape[0] + np.trace(np.linalg.inv(cov2) @ cov1)
                    + np.transpose(mean2 - mean1) @ np.linalg.inv(cov2) @ (mean2 - mean1))
    return kl_div

def compute_diag_gaussian_kl_div(mean1, var1, mean2, var2):
    kl_div = 0.5*(np.log(var2) - np.log(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
    kl_div = np.sum(kl_div, axis=-1)

    return kl_div

def compute_diag_gaussian_wasserstein_distance(mean1, var1, mean2, var2):
    wasserstein_distance = (mean1 - mean2)**2 + var1 + var2 - 2*np.sqrt(var1*var2)
    wasserstein_distance = np.sum(wasserstein_distance, axis=-1)

    return wasserstein_distance**0.5

def compare_Gaussians_per_component(M,
                                    # true_cond_comp_log_probs,
                                    true_cond_means,
                                    true_cond_covs,
                                    # comp_log_probs,
                                    means,
                                    covs):

    fow_kl_divs = torch.empty(M.shape[0], true_cond_means.shape[1])
    rev_kl_divs = torch.empty(M.shape[0], true_cond_means.shape[1])
    # jsds = torch.empty(M.shape[0], true_cond_comp_log_probs.shape[-1])
    for i in range(M.shape[0]):
        M_i = M[i]
        # true_cond_comp_log_probs_i = true_cond_comp_log_probs[i]
        true_cond_means_i = true_cond_means[i][:, ~M_i]
        true_cond_covs_i = true_cond_covs[i][:, ~M_i, :][:, :, ~M_i]
        # comp_log_probs_i = comp_log_probs[i]
        means_i = means[i][:, ~M_i]
        covs_i = covs[i][:, ~M_i, :][:, :, ~M_i]

        for c in range(true_cond_means_i.shape[0]):
            # true_cond_comp_log_probs_ic = true_cond_comp_log_probs_i[c]
            true_cond_means_ic = true_cond_means_i[c]
            true_cond_covs_ic = true_cond_covs_i[c]
            # comp_log_probs_ic = comp_log_probs_i[c]
            means_ic = means_i[c]
            covs_ic = covs_i[c]

            fowkl = compute_gaussian_kl_div(asnumpy(true_cond_means_ic), asnumpy(true_cond_covs_ic),
                                            asnumpy(means_ic), asnumpy(covs_ic))
            revkl = compute_gaussian_kl_div(asnumpy(means_ic), asnumpy(covs_ic),
                                            asnumpy(true_cond_means_ic), asnumpy(true_cond_covs_ic))

            fow_kl_divs[i,c] = fowkl
            rev_kl_divs[i,c] = revkl

    return fow_kl_divs, rev_kl_divs

def compare_diag_Gaussians_per_component(M,
                                         # true_cond_comp_log_probs,
                                         true_cond_means,
                                         true_cond_vars,
                                         # comp_log_probs,
                                         means,
                                         vars):

    fow_kl_divs = torch.empty(M.shape[0], true_cond_means.shape[1])
    rev_kl_divs = torch.empty(M.shape[0], true_cond_means.shape[1])
    # jsds = torch.empty(M.shape[0], true_cond_comp_log_probs.shape[-1])
    for i in range(M.shape[0]):
        M_i = M[i]
        # true_cond_comp_log_probs_i = true_cond_comp_log_probs[i]
        true_cond_means_i = true_cond_means[i][:, ~M_i]
        true_cond_vars_i = true_cond_vars[i][:, ~M_i]
        # comp_log_probs_i = comp_log_probs[i]
        means_i = means[i][:, ~M_i]
        vars_i = vars[i][:, ~M_i]

        for c in range(true_cond_means_i.shape[0]):
            # true_cond_comp_log_probs_ic = true_cond_comp_log_probs_i[c]
            true_cond_means_ic = true_cond_means_i[c]
            true_cond_vars_ic = true_cond_vars_i[c]
            # comp_log_probs_ic = comp_log_probs_i[c]
            means_ic = means_i[c]
            vars_ic = vars_i[c]

            fowkl = compute_diag_gaussian_kl_div(asnumpy(true_cond_means_ic), asnumpy(true_cond_vars_ic),
                                                 asnumpy(means_ic), asnumpy(vars_ic))
            revkl = compute_diag_gaussian_kl_div(asnumpy(means_ic), asnumpy(vars_ic),
                                                 asnumpy(true_cond_means_ic), asnumpy(true_cond_vars_ic))
            fow_kl_divs[i,c] = torch.tensor(fowkl)
            rev_kl_divs[i,c] = torch.tensor(revkl)

    return fow_kl_divs, rev_kl_divs

def compare_diag_Gaussian_per_component_using_Wasserstein(M,
                                         true_cond_means,
                                         true_cond_vars,
                                         means,
                                         vars):
    W = torch.empty(M.shape[0], true_cond_means.shape[1])
    for i in range(M.shape[0]):
        M_i = M[i]
        true_cond_means_i = true_cond_means[i][:, ~M_i]
        true_cond_vars_i = true_cond_vars[i][:, ~M_i]
        means_i = means[i][:, ~M_i]
        vars_i = vars[i][:, ~M_i]

        for c in range(true_cond_means_i.shape[0]):
            true_cond_means_ic = true_cond_means_i[c]
            true_cond_vars_ic = true_cond_vars_i[c]
            means_ic = means_i[c]
            vars_ic = vars_i[c]

            wd = compute_diag_gaussian_wasserstein_distance(asnumpy(true_cond_means_ic), asnumpy(true_cond_vars_ic),
                                                            asnumpy(means_ic), asnumpy(vars_ic))

            W[i,c] = torch.tensor(wd)
    return W

def compare_vae_and_mog_using_fid(num_samples, *, vae, mog_parameters, inception_model):
    vae_samples = vae.sample(num_samples)

    comp_probs = torch.tensor(mog_parameters['comp_logits'].exp()).squeeze(0)
    means = torch.tensor(mog_parameters['means'])
    covs = torch.tensor(mog_parameters['covs'])
    mog_samples = sample_mog(num_samples, comp_probs, means, covs=covs)

    fid_scores = compute_fid_score(mog_samples[None, :], vae_samples[None, :], inception_model=inception_model)

    print('FID score: {}'.format(fid_scores))
    return fid_scores

if __name__ == '__main__':
    train_dataset = MNIST_GMM(root='./data', filename='mnist_gmm_data', split='train')
    min_vals = torch.tensor(train_dataset[:].min(axis=0))
    max_vals = torch.tensor(train_dataset[:].max(axis=0))

    data = MNIST_GMM_customtest(root='./data', split = 'test', version=0)
    params = {
        'comp_logits': torch.tensor(np.log(data.data_file['comp_probs'])),
        'covs': torch.tensor(data.data_file['covs']),
        'means': torch.tensor(data.data_file['means'])
    }
    datapoints = data[:]

    batch_size=20

    inception_model_path = './logs/mnist_gmm/imputation/mnist_gmm_resnet_classifier/seed_m20220118_d20220118/lightning_logs/version_3/checkpoints/epoch=39-step=3600.ckpt'
    inception_model = ResNetClassifier.load_from_checkpoint(checkpoint_path=inception_model_path)
    inception_model_classifier = inception_model.to('cpu')
    inception_model_path = './logs/mnist_gmm/imputation/vae_convresnet3/seed_m20220118_d20220118/lightning_logs/version_0/checkpoints/epoch=5999-step=540000.ckpt'
    inception_model = VAE.load_from_checkpoint(checkpoint_path=inception_model_path)
    inception_model = inception_model.to('cpu')
    inception_model_encoder = inception_model.var_latent_network

    seeds = [20220118,
             2022011811, 2022011822, 2022011833, 2022011844,
             2022011855, 2022011866, 2022011877, 2022011888, 2022011899,
             20230118, 2023011811, 2023011822, 2023011833, 2023011844,
             2023011855, 2023011866, 2023011877, 2023011888, 2023011899,]

    # irwg_use_resampled_imps = True
    irwg_use_resampled_imps = False
    fit_iterations = [int(1e6)] # So we use all iterations

    # only_allow_filter = ['vae_convresnet3_k5_mwgm_usehistrestricted_005prior_with_irwg_warmup_replenish1']
    # only_allow_filter = ['ablation_vae_convresnet3_k5_mwgm_mwgtarget_usehistrestricted_005prior_nowarm',
    #                      'ablation_vae_convresnet3_k5_mwgm_usehistrestricted_000prior_nowarm',
    #                      'ablation_vae_convresnet3_standard_ir']
    only_allow_filter = ['ablation_vae_convresnet3_k2_irwg_i1_dmis_gr_mult_replenish3_finalresample',
                         'ablation_vae_convresnet3_k5_irwg_i1_dmis_gr_mult_replenish0_finalresample']


    dir = exp_dir.split('{}')[0]
    for exp in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, exp)):
            for seed in seeds:
                if not os.path.exists(exp_dir.split('version_{}')[0].format(exp, seed, seed)):
                    continue
                if (not irwg_use_resampled_imps) and '_finalresample' in exp:
                    continue
                if len(only_allow_filter) > 0 and exp not in only_allow_filter:#
                    continue

                versions = os.listdir(exp_dir.split('version_{}')[0].format(exp, seed, seed))
                versions = sorted([int(v.split('version_')[1]) for v in versions ])
                for v in versions:
                    log_dir = exp_dir.format(exp, seed, seed, v)

                    try:
                        process_dir(log_dir, params, batch_size, fit_iterations=fit_iterations, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints)
                        # process_dir_diag_gauss(log_dir, params, batch_size, fit_iterations=fit_iterations, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints,
                                            #    min_max_vals=(min_vals, max_vals), clamp_imps_effectivesamples=5)
                        process_dir_fid(log_dir, params, batch_size, filename='mog_fid_scores', fit_iterations=fit_iterations, inception_model=inception_model_classifier, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints,
                                    min_max_vals=(min_vals, max_vals))
                        process_dir_fid(log_dir, params, batch_size, filename='mog_fid_scores2', fit_iterations=fit_iterations, inception_model=inception_model_encoder, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints,
                                    min_max_vals=(min_vals, max_vals))
                    except FileNotFoundError:
                        print(f'No imputations or fitted mog params found in experiment: {log_dir}. Skipping.')
                        continue

    # irwg_use_resampled_imps = False
    # fit_iterations = [int(1e6)] # So we use all iterations

    # log_dir = './logs/mnist_gmm/imputation/samples/vae_convresnet3_k5_mwgm_usehistrestricted_005prior_nowarm/seed_m20220118_d20220118/lightning_logs/version_0/'
    # try:
    #     process_dir(log_dir, params, batch_size, fit_iterations=fit_iterations, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints)
    #     # process_dir_diag_gauss(log_dir, params, batch_size, fit_iterations=fit_iterations, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints,
    #                         #    min_max_vals=(min_vals, max_vals), clamp_imps_effectivesamples=5)
    #     process_dir_fid(log_dir, params, batch_size, filename='mog_fid_scores', fit_iterations=fit_iterations, inception_model=inception_model_classifier, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints,
    #                 min_max_vals=(min_vals, max_vals))
    #     process_dir_fid(log_dir, params, batch_size, filename='mog_fid_scores2', fit_iterations=fit_iterations, inception_model=inception_model_encoder, irwg_use_resampled_imps=irwg_use_resampled_imps, true_datapoints=datapoints,
    #                 min_max_vals=(min_vals, max_vals))
    # except FileNotFoundError:
    #     print(f'No imputations or fitted mog params found in experiment: {log_dir}. Skipping.')
    #     # continue

    # vae_model_path = './logs/mnist_gmm/imputation/vae_convresnet3/seed_m20220118_d20220118/lightning_logs/version_0/checkpoints/epoch=5999-step=540000.ckpt'
    # vae = VAE.load_from_checkpoint(checkpoint_path=vae_model_path)
    # vae = vae.to('cpu')

    # num_samples = 10000
    # compare_vae_and_mog_using_fid(num_samples, vae=vae, mog_parameters=params, inception_model=inception_model)

    # batch_size=1
    # masks = ~(datapoints.isnan())
    # # fit_and_eval_on_ground_truth_samples(datapoints, masks, params, batch_size)
    # # fit_diag_and_eval_on_ground_truth_samples(datapoints, masks, params, batch_size)
    # eval_fid_on_ground_truth_samples(datapoints, masks, params, batch_size, inception_model=inception_model)
