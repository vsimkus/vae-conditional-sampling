from typing import List, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from irwg.dependencies.vdvae.hps import HPARAMS_REGISTRY
from irwg.dependencies.vdvae.vae import load_vae


def save_img(img, filename):
    img = img.reshape(-1, *img.shape[-2:])
    imageio.imwrite(filename, img)


class VDVAELightning(LightningModule):
    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

        assert dataset in HPARAMS_REGISTRY.keys(), \
            f'{dataset=} is not configured.'

        H = HPARAMS_REGISTRY[dataset]
        self.vae = load_vae(H)

    def sample_unconditional(self, N, t=1.0):
        return self.vae.forward_uncond_samples(N, t=t)

    def compute_log_ratios(self, X_input, X_target, M, use_dmmis=False, dmmis_num_comps=None):
        activations = self.vae.encoder.forward(X_input)
        px_z, stats = self.vae.decoder.forward_and_pq_ratio(activations, get_latents=True, use_dmmis=use_dmmis, dmmis_num_comps=dmmis_num_comps)

        # distortion_per_pixel = self.vae.decoder.out_net.nll(px_z, X_target)
        gen_logprob_marg = self.vae.decoder.out_net.log_prob(px_z, X_target, M)
        # gen_logprob_marg = self.vae.decoder.out_net.log_prob(px_z, X_target, torch.ones_like(M))
        # rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        pq_ratio = torch.zeros_like(gen_logprob_marg)
        breakpoint()
        num_z = 0
        # ndims = np.prod(X_input.shape[1:])
        for statdict in stats:
            # rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))

            # params, z = statdict['params'], statdict['z']
            # qm, qv, pm, pv = params['qm'], params['qv'], params['pm'], params['pv']

            # q_distr = torch.distributions.Normal(qm, torch.exp(qv))
            # p_distr = torch.distributions.Normal(pm, torch.exp(pv))

            # prior_logprob = p_distr.log_prob(z).sum(dim=(1,2,3))
            # var_logprob = q_distr.log_prob(z).sum(dim=(1,2,3))

            pq_ratio += statdict['pq_ratio']
            num_z += np.prod(statdict['z'].shape[1:])
        breakpoint()
        print(num_z)
        # NOTE: This is because the decoder log-probability (nll) is scaled by 1/ndims in the training code
        # rate_per_pixel /= ndims
        logratio = gen_logprob_marg + pq_ratio
        # return dict(elbo=elbo, distortion=distortion_per_pixel.mean(), rate=rate_per_pixel.mean())
        return logratio, px_z

    # def asd(self, X):
    #     activations = self.encoder.forward(x)
    #     px_z, stats = self.decoder.forward(activations)
    #     distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target)
    #     rate_per_pixel = torch.zeros_like(distortion_per_pixel)
    #     ndims = np.prod(x.shape[1:])
    #     for statdict in stats:
    #         rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
    #     rate_per_pixel /= ndims
    #     elbo = (distortion_per_pixel + rate_per_pixel).mean()

    def pseudo_gibbs_iteration(self, X, M):
        # # Create latent distribution and sample
        # var_latent_params = model.predict_var_latent_params(X, M)

        # # Sample latent variables
        # var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
        # Z_imp = var_latent_distr.sample()

        # # Create the distribution of the missingness model
        # mis_params = model.generator_network(Z_imp)
        # mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # # Sample missing values
        # X_m = mis_distr.sample()

        # # Set imputed missing values
        # return X*M + X_m*~M
        enc_activations = self.vae.encoder.forward(X)

        px_z, _ = self.vae.decoder.forward(enc_activations, get_latents=False)

        # X_m = self.vae.decoder.out_net.sample(px_z)
        # X_m = output_to_input(X_m, dataset=self.dataset)

        X_m = self.vae.decoder.out_net.sample_noquantization(px_z)
        X_m = output_to_input(X_m, dataset=self.dataset)

        return X*M + X_m*~M

    def pseudo_gibbs(self, X, M):
        K=5
        X = torch.repeat_interleave(X, repeats=K, dim=0)
        M = torch.repeat_interleave(M, repeats=K, dim=0)

        # (1) Impute with random observed pixel values from the same image
        X_imp = X.reshape(X.shape[0], -1, X.shape[-1])
        idx = torch.distributions.Categorical(probs=M.reshape(M.shape[0], -1)).sample(sample_shape=(X_imp.shape[-2],))
        X_imp = X_imp[torch.arange(X_imp.shape[0])[:, None], idx.T, :]
        X_imp = X_imp.reshape_as(X)

        # (2) Impute with random sample from the prior
        # with torch.inference_mode():
        #     X_imp = self.vae.forward_uncond_samples(n_batch=X.shape[0], t=1.0)

        X = X*M + X_imp*(~M)
        del X_imp

        # Preprocess for input
        X = image_to_input(X, dataset=self.dataset)

        imageio.imwrite('test0_pg.png', input_to_image(X, dataset=self.dataset).cpu().numpy().reshape(-1, X.shape[-2], 3))
        T=1000
        for t in tqdm(range(T)):
            X = self.pseudo_gibbs_iteration(X, M)

        X2 = input_to_image(X, dataset=self.dataset)
        imageio.imwrite(f'test{T}_pg.png', X2.cpu().numpy().reshape(-1, X.shape[-2], 3))
        # X2 = input_to_image(X, dataset=self.dataset)
        # imageio.imwrite(f'test1000_pg.png', X2.cpu().numpy().reshape(-1, X.shape[-2], 3))


    def importance_resampling_gibbs_gr_iteration(self, X, M, K,num_imp_samples=1, weighting='dmis', resampling_method='multinomial'):
        # # Create latent distribution and sample
        # var_latent_params = model.predict_var_latent_params(X, M)

        # # Sample latent variables
        # var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
        # Z_imp = var_latent_distr.sample(sample_shape=(num_imp_samples,))
        # # Shape (i b k d)

        # # Create the distribution of the missingness model
        # generator_params = model.generator_network(Z_imp)
        # generator_distr = model.create_distribution(generator_params, model.hparams.generator_distribution)

        # # Create prior distribution
        # prior_distr = torch.distributions.Normal(loc=0., scale=1.)

        # # Compute (unnormalised)-log-importance-weights
        # if weighting == 'dmis':
        #     log_weights = compute_mis_log_unnormalised_importance_weights(X, M, Z_imp,
        #                                                                 var_latent_distr=var_latent_distr,
        #                                                                 var_comp_neg_idx=-2,
        #                                                                 prior_distr=prior_distr,
        #                                                                 generator_distr=generator_distr)
        # elif weighting == 'smis':
        #     log_weights = compute_log_unnormalised_importance_weights(X, M, Z_imp,
        #                                                                     var_latent_distr,
        #                                                                     prior_distr,
        #                                                                     generator_distr)
        # else:
        #     raise NotImplementedError()

        # # Shape (i b k)
        # # NOTE: here I treat i*k as the number of importance samples
        # log_weights = rearrange(log_weights, 'i b k -> b (i k)')

        # # Sample from the importance-weighted distribution
        # if resampling_method == 'multinomial':
        #     importance_distr = torch.distributions.Categorical(logits=log_weights)
        #     idx = importance_distr.sample(sample_shape=(X.shape[1],))
        # elif resampling_method == 'systematic':
        #     idx = _systematic_sample(log_weights, num_dependent_samples=X.shape[1])
        #     idx = rearrange(idx, 'b k -> k b')
        # else:
        #     raise NotImplementedError()

        # # Get generator params for the corresponding Z's
        # mis_params = rearrange(generator_params, 'i b k p -> b (i k) p')
        # mis_params = mis_params[torch.arange(X.shape[0], device=generator_params.device),
        #                         idx,
        #                         ...]
        # mis_params = rearrange(mis_params, 'k b d -> b k d')
        # mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # # Sample missing values
        # X_m = mis_distr.sample()

        # # Compute average ESS
        # log_norm_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        # ess = torch.exp(-torch.logsumexp(2 * log_norm_weights, -1))
        # avg_ess = ess.mean(dim=0).cpu()

        # # Set imputed missing values
        # return X*M + X_m*~M, avg_ess

        # enc_activations = self.vae.encoder.forward(X)

        # px_z, stats = self.vae.decoder.forward_and_params(enc_activations, get_latents=True)

        # Z_imp = stats[...]['z']
        # params = stats[...]['params']

        # TODO: don't have to do this conversion each time, since only the observed X matter.
        X_target=image_to_target(input_to_image(X, dataset=self.dataset), dataset=self.dataset)
        log_weights, px_z = self.compute_log_ratios(X_input=X,
                                              X_target=X_target,
                                              M=M,
                                              use_dmmis=True,
                                              dmmis_num_comps=K)
        log_weights = rearrange(log_weights, '(b k) -> b k', k=K)

        # Try scaling by the number of pixels
        # log_weights /= np.prod(X.shape[1:])

        px_z = rearrange(px_z, '(b k) ... -> b k ...', k=K)

        importance_distr = torch.distributions.Categorical(logits=log_weights)
        idx = importance_distr.sample(sample_shape=(K,))

        px_z = px_z[torch.arange(px_z.shape[0]), idx.transpose(0, -1), ...]

        px_z = rearrange(px_z, 'b k ... -> (b k) ...', k=K)

        X_m = self.vae.decoder.out_net.sample_noquantization(px_z)
        # X_m = self.vae.decoder.out_net.sample(px_z)
        X_m = output_to_input(X_m, dataset=self.dataset)

        # Compute ESS
        log_norm_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        ess = torch.exp(-torch.logsumexp(2 * log_norm_weights, -1))
        avg_ess = ess.mean(dim=0).cpu()

        return X*M + X_m*~M, avg_ess

    def irwg(self, X, M):
        K=1
        X = torch.repeat_interleave(X, repeats=K, dim=0)
        M = torch.repeat_interleave(M, repeats=K, dim=0)

        # (1) Impute with random observed pixel values from the same image
        X_imp = X.reshape(X.shape[0], -1, X.shape[-1])
        idx = torch.distributions.Categorical(probs=M.reshape(M.shape[0], -1)).sample(sample_shape=(X_imp.shape[-2],))
        X_imp = X_imp[torch.arange(X_imp.shape[0])[:, None], idx.T, :]
        X_imp = X_imp.reshape_as(X)

        # (2) Impute with random sample from the prior
        # with torch.inference_mode():
        #     X_imp = self.vae.forward_uncond_samples(n_batch=X.shape[0], t=1.0)

        # X = X*M + X_imp*(~M)
        # del X_imp

        # Preprocess for input
        X = image_to_input(X, dataset=self.dataset)

        imageio.imwrite(f'test0_irwg_K{K}.png', input_to_image(X, dataset=self.dataset).cpu().numpy().reshape(-1, X.shape[-2], 3))
        T=100
        ess = []
        for t in tqdm(range(T)):
            X, avg_ess = self.importance_resampling_gibbs_gr_iteration(X, M, K=K)
            ess.append(avg_ess)

        X2 = input_to_image(X, dataset=self.dataset)
        imageio.imwrite(f'test{T}_irwg_K{K}.png', X2.cpu().numpy().reshape(-1, X.shape[-2], 3))

        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(ess)), np.array(ess))
        # fig.show()
        fig.savefig(f'ess{T}_irwg_K{K}.png')
        # X2 = input_to_image(X, dataset=self.dataset)
        # imageio.imwrite(f'test1000_irwg.png', X2.cpu().numpy().reshape(-1, X.shape[-2], 3))

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        X, M = batch[0], batch[1]
        # I = batch[2]
        M = M.unsqueeze(-1)

        imageio.imwrite('test_orig.png', X.cpu().numpy().reshape(-1, X.shape[-2], 3))
        imageio.imwrite('test_orig_incomp.png', (X*M).cpu().numpy().reshape(-1, X.shape[-2], 3))

        # TODO: Make multiple copies



        # TODO: use target pre-processing and post-processing if necessary
        # Different postprocessing for writing images

        with torch.inference_mode():
            breakpoint()
            # X = self.pseudo_gibbs(X, M)
            self.irwg(X, M)
            breakpoint()
            asd
        return

def output_to_input(X, dataset):
    if dataset == 'ffhq256':
        # Transform 5bit image output of the NN back to
        # input space
        shift = -108.86352494925086
        scale = 1. / 69.75047759851941
    elif dataset == 'cifar10':
        shift = -120.63838
        scale = 1. / 64.16736
    elif dataset == 'imagenet32':
        shift = -116.2373
        scale = 1. / 69.37404
    elif dataset == 'imagenet64':
        shift = -115.92961967
        scale = 1. / 69.37404
    else:
        raise ValueError('unknown dataset: ', dataset)

    X = X.to(torch.float)
    X += shift
    X *= scale

    return X

def image_to_input(X, dataset):
    # Follow input preprocessing from VDVAE paper
    if dataset == 'ffhq256':
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif dataset == 'ffhq1024':
        shift = -0.4387
        scale = 1.0 / 0.2743
    elif dataset == 'cifar10':
        shift = -120.63838
        scale = 1. / 64.16736
    elif dataset == 'imagenet32':
        shift = -116.2373
        scale = 1. / 69.37404
    elif dataset == 'imagenet64':
        shift = -115.92961967
        scale = 1. / 69.37404
    else:
        raise ValueError('unknown dataset: ', dataset)

    X = X.to(torch.float)
    X += shift
    X *= scale
    return X

def input_to_image(X, dataset):
    # Undo input preprocessing from VDVAE paper
    if dataset == 'ffhq256':
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif dataset == 'ffhq1024':
        shift = -0.4387
        scale = 1.0 / 0.2743
    elif dataset == 'cifar10':
        shift = -120.63838
        scale = 1. / 64.16736
    elif dataset == 'imagenet32':
        shift = -116.2373
        scale = 1. / 69.37404
    elif dataset == 'imagenet64':
        shift = -115.92961967
        scale = 1. / 69.37404
    else:
        raise ValueError('unknown dataset: ', dataset)

    X = X / scale
    X -= shift
    return X.clamp_(0, 255).to(torch.uint8)

def image_to_target(X, dataset):
    # Follow preprocessing from VDVAE paper
    if dataset == 'ffhq256':
        shift_loss = -127.5
        scale_loss = 1. / 127.5
        do_low_bit = True
    elif dataset == 'ffhq1024':
        shift_loss = -0.5
        scale_loss = 2.0
        do_low_bit = False
    elif dataset == 'cifar10':
        shift_loss = -127.5
        scale_loss = 1. / 127.5
        do_low_bit = False
    elif dataset == 'imagenet32':
        shift_loss = -127.5
        scale_loss = 1. / 127.5
        do_low_bit = False
    elif dataset == 'imagenet64':
        shift_loss = -127.5
        scale_loss = 1. / 127.5
        do_low_bit = False
    else:
        raise ValueError()

    X = X.clone().float()
    if do_low_bit:
        # 5 bits of precision
        X.mul_(1. / 8.).floor_().mul_(8.)
        # NOTE: this means that the X are now between 0 and 248!.
    X.add_(shift_loss).mul_(scale_loss)

    return X


