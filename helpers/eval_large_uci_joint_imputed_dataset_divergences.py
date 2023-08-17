import datetime
import math
import os
import os.path
import sys
import time
from collections import defaultdict

import numpy as np
import geomloss

import torch
from dateutil import relativedelta
from einops import asnumpy, rearrange, repeat
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from tqdm import trange

# from compute_scores_util import build_argparser
from pytorch_lightning.utilities.cli import LightningArgumentParser

from irwg.data import MissingDataModule
from irwg.dependencies.otimputation.otimputer import \
    pick_epsilon as pick_epsilon_sinkhorn
from irwg.dependencies.otimputation.otimputer import nanmean
from irwg.dependencies.otimputation.otimputer import quantile

def construct_experiment_dir(args):
    # return f'{args.experiment_subdir_base}/seed_m{args.seed_everything}_d{args.data.setup_seed}'
    versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(args.experiment_path.split('version_')[0])))
    if len(versions) > 1:
        print('More than one version is available:', versions, '. Stopping.')
        sys.exit()
    version = versions[-1]
    return args.experiment_dir.format(version)

def build_argparser():
    parser = LightningArgumentParser('Eval divergences on imputations',
                                     parse_as_dict=False)

    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)

    parser.add_lightning_class_args(MissingDataModule, 'data')
    # parser.add_argument('--vae_path', type=str, required=False, default=None,
    #                     help=('VAE for generating VAE-data.'))
    # parser.add_argument('--load_best_vae', type=bool, default=None,
    #                     help=('If model_path is directoy use this flag to load the `epoch=*step=*.ckpt`. Throws error if there are multiple.'))

    parser.add_argument('--eval_for_one_seed', type=bool, required=False, default=True,
                        help='If true only evaluate for one seed')

    parser.add_argument('--default_root_dir', type=str, default=None,
                        help=('Default path.'))
    parser.add_argument('--experiment_dir', type=str,
                        help=('Results path'))

    # Add experiment path
    parser.add_argument('--experiment_path', type=str, required=True,
                        help=('Path (or template) of the experiment directory'))

    # # Use samples from VAE marginal instead of the imputations from the file for assessment of the evaluations.
    # parser.add_argument('--use_marginal_vae_samples_instead_of_imputations', type=bool, default=False,
    #                     help=('Instead of using the samples from the file, it generates samples from the marginal of the VAE. Used to assess/debug the evaluation.'))

    # Imputation args
    parser.add_argument('--last_n_imputations', type=int, required=True,
                        help=('Number of last imputations'))
    parser.add_argument('--every_nth_step', type=int, required=True,
                        help=('Use every nth step imputations'))
    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size.')
    parser.add_argument('--process_irwg_finalresample', type=bool, required=False, default=False,
                        help='Also processes the result file containing IRWG final-resampled imputations.')

    parser.add_argument('--backend', type=str, required=False, default='auto',
                        help=('Backend for geomloss.'))

    parser.add_argument('--compute_metrics', type=str, required=True, nargs='+',
                        help=('Which metrics to compute.'))

    return parser


def load_last_nstep_imputations_from_experiment(path, last_n_steps, load_every_nth_step, filename_prefix='imputations_'):
    imp_files = [f for f in os.listdir(path) if f.startswith(filename_prefix) and f.endswith('.npz')]
    batch_idxs = [int(f.split(filename_prefix)[1].split('.npz')[0]) for f in imp_files]
    imp_files = [f for _, f in sorted(zip(batch_idxs, imp_files))]
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

def vae_marginal_samples_imputation(X_true, M, *, num_imps, vae_model, batch_size):
    X_imp = []
    for batch_idx in trange(math.ceil(X_true.shape[0]/batch_size), desc='Imputing data from marginal of VAE'):
        X_true_i = X_true[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_true.shape[0])]
        M_i = M[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_true.shape[0])]

        imps = rearrange(vae_model.sample(X_true_i.shape[0]*num_imps), '(b k) ... -> b k ...', b=X_true_i.shape[0], k=num_imps)

        # Replace the missing values with 0 before doing imputation
        X_true_i = torch.tensor(X_true_i).clone()
        X_true_i[~M_i] = 0.
        X_true_i = repeat(X_true_i, 'b ... -> b k ...', k=num_imps)
        M_i = M_i[:, None]

        X_imp_i = X_true_i*M_i + imps*(~M_i)
        X_imp.append(X_imp_i)
    return torch.concat(X_imp, dim=0)

def pick_blur(X, quant=0.5, max_points=2000):
    """
        Returns a quantile of the pairwise squared distances in X.
        Used to select a regularization parameter for MMD distances (Gaussian/Laplacian).

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten()
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item()

def sinkhorn_per_imputation_autoblur(X_imp, X_true, *, X_true_train=None, backend='auto', blur=None, scaling=0.9, p=2, batch_size=None, verbose=False):
    if isinstance(X_imp, np.ndarray):
        X_imp = torch.tensor(X_imp)
    if isinstance(X_true, np.ndarray):
        X_true = torch.tensor(X_true)
    if X_true_train is not None and isinstance(X_true_train, np.ndarray):
        X_true_train = torch.tensor(X_true_train)

    # X_true = rearrange(X_true, ' b 1 d -> b d')
    if X_true_train is not None:
        X_true = torch.concat([X_true, X_true_train], dim=0)

    if blur is None:
        blur = pick_epsilon_sinkhorn(X_true)
        print(f'Sinkhorn blur auto-picked: {blur}')

    print('Backend:', backend)

    loss = geomloss.SamplesLoss("sinkhorn", blur=blur, p=p, scaling=scaling, backend=backend, verbose=verbose)

    # t, k = X_imp.shape[2], X_imp.shape[1]
    # X_imp = rearrange(X_imp, 'b k t ... -> (t k) b ...')
    X_imp = rearrange(X_imp, 'b k ... -> k b ...')

    if batch_size is None:
        batch_size = X_imp.shape[0]
    print(f'Batch size: {batch_size}')

    losses = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size)):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_true_i = repeat(X_true, 'b d -> tk b d', tk=X_imp_i.shape[0])
        X_imp_i = X_imp_i.contiguous()
        X_true_i = X_true_i.contiguous()

        with torch.inference_mode():
            losses.append(asnumpy(loss(X_imp_i, X_true_i)))

    losses = np.concatenate(losses, axis=0)
    return losses

def energymmd_per_imputation(X_imp, X_true, *, X_true_train=None, backend='auto', batch_size=None, verbose=False):
    if isinstance(X_imp, np.ndarray):
        X_imp = torch.tensor(X_imp)
    if isinstance(X_true, np.ndarray):
        X_true = torch.tensor(X_true)
    if X_true_train is not None and isinstance(X_true_train, np.ndarray):
        X_true_train = torch.tensor(X_true_train)

    # X_true = rearrange(X_true, ' b 1 d -> b d')
    if X_true_train is not None:
        X_true = torch.concat([X_true, X_true_train], dim=0)

    print('Backend:', backend)

    loss = geomloss.SamplesLoss("energy", backend=backend, verbose=verbose)

    # t, k = X_imp.shape[2], X_imp.shape[1]
    # X_imp = rearrange(X_imp, 'b k t ... -> (t k) b ...')
    X_imp = rearrange(X_imp, 'b k ... -> k b ...')

    if batch_size is None:
        batch_size = X_imp.shape[0]
    print(f'Batch size: {batch_size}')

    losses = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size)):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_true_i = repeat(X_true, 'b d -> tk b d', tk=X_imp_i.shape[0])

        with torch.inference_mode():
            losses.append(asnumpy(loss(X_imp_i, X_true_i)))

    losses = np.concatenate(losses, axis=0)
    return losses

def gaussianmmd_per_imputation_autoblur(X_imp, X_true, *, X_true_train=None, backend='auto', batch_size=None, verbose=False):
    if isinstance(X_imp, np.ndarray):
        X_imp = torch.tensor(X_imp)
    if isinstance(X_true, np.ndarray):
        X_true = torch.tensor(X_true)
    if X_true_train is not None and isinstance(X_true_train, np.ndarray):
        X_true_train = torch.tensor(X_true_train)

    # X_true = rearrange(X_true, ' b 1 d -> b d')
    if X_true_train is not None:
        X_true = torch.concat([X_true, X_true_train], dim=0)

    blur = pick_blur(X_true)
    print(f'Gaussian blur auto-picked: {blur}')
    print('Backend:', backend)

    loss = geomloss.SamplesLoss("gaussian", blur=blur, backend=backend, verbose=verbose)

    # t, k = X_imp.shape[2], X_imp.shape[1]
    # X_imp = rearrange(X_imp, 'b k t ... -> (t k) b ...')
    X_imp = rearrange(X_imp, 'b k ... -> k b ...')


    if batch_size is None:
        batch_size = X_imp.shape[0]
    print(f'Batch size: {batch_size}')

    losses = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size)):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_true_i = repeat(X_true, 'b d -> tk b d', tk=X_imp_i.shape[0])

        with torch.inference_mode():
            losses.append(asnumpy(loss(X_imp_i, X_true_i)))

    losses = np.concatenate(losses, axis=0)
    return losses

def laplacianmmd_per_imputation_autoblur(X_imp, X_true, *, X_true_train=None, backend='auto', batch_size=None, verbose=False):
    if isinstance(X_imp, np.ndarray):
        X_imp = torch.tensor(X_imp)
    if isinstance(X_true, np.ndarray):
        X_true = torch.tensor(X_true)
    if X_true_train is not None and isinstance(X_true_train, np.ndarray):
        X_true_train = torch.tensor(X_true_train)

    # X_true = rearrange(X_true, ' b 1 d -> b d')
    if X_true_train is not None:
        X_true = torch.concat([X_true, X_true_train], dim=0)

    blur = pick_blur(X_true)
    print(f'Laplacian blur auto-picked: {blur}')
    print('Backend:', backend)

    loss = geomloss.SamplesLoss("laplacian", blur=blur, backend=backend, verbose=verbose)

    # t, k = X_imp.shape[2], X_imp.shape[1]
    # X_imp = rearrange(X_imp, 'b k t ... -> (t k) b ...')
    X_imp = rearrange(X_imp, 'b k ... -> k b ...')

    if batch_size is None:
        batch_size = X_imp.shape[0]
    print(f'Batch size: {batch_size}')

    losses = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size)):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_true_i = repeat(X_true, 'b d -> tk b d', tk=X_imp_i.shape[0])

        with torch.inference_mode():
            losses.append(asnumpy(loss(X_imp_i, X_true_i)))

    losses = np.concatenate(losses, axis=0)
    return losses

def run(hparams):
    # Set random seed
    # NOTE: this must be done before any class initialisation,
    # hence also before the call to parser.instantiate_classes()
    seed_everything(hparams.seed_everything, workers=True)

    # Get the instantiated data
    # datamodule = hparams.data
    datamodule = MissingDataModule(**hparams.data)

    # # In case we want to generate data from the model
    # if hparams.vae_path is not None:
    #     versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.vae_path.split('version_')[0])))
    #     if len(versions) > 1:
    #         print('More than one version is available:', versions, '. Loading ', versions[-1])
    #     version = versions[-1]
    #     model_path = hparams.vae_path.format(version)
    #     if os.path.isdir(model_path):
    #         models = os.listdir(model_path)
    #         if hparams.load_best_vae is not None and hparams.load_best_vae:
    #             models.remove('last.ckpt')
    #         if len(models) > 1:
    #             raise ValueError(f'Too many models in path: {model_path}')
    #         model_path = os.path.join(model_path, models[0])
    #     print('VAE model path:', model_path)
    #     model = VAE.load_from_checkpoint(checkpoint_path=model_path)
    #     datamodule.set_model(model)

    # The instantiation steps might be different for different models.
    # Hence we reset the seed before training such that the seed at the start of lightning setup is the same.
    reset_seed()

    # Begin testing
    datamodule.setup('fit')
    datamodule.setup('test')
    # train_data = datamodule.train_data[:]
    # X_train, M_train = train_data[0], train_data[1]
    # if isinstance(M_train, torch.Tensor):
    #     M_train = asnumpy(M_train)
    # if isinstance(X_train, torch.Tensor):
    #     X_train = asnumpy(X_train)
    # test_data = datamodule.test_data[:]
    # X_test, M_test = test_data[0], test_data[1]
    # if isinstance(M_test, torch.Tensor):
    #     M_test = asnumpy(M_test)
    # if isinstance(X_test, torch.Tensor):
    #     X_test = asnumpy(X_test)
    # if isinstance(X_train, torch.Tensor):
    #     X_train = asnumpy(X_train)

    # Construct the experiment directory
    if hparams.default_root_dir is None:
        experiment_dir = f'./{hparams.experiment_dir}'
    else:
        experiment_dir = f'{hparams.default_root_dir}/{hparams.experiment_dir}'

    # if hparams.eval_for_one_seed:
    # Load imputations
    versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.experiment_path.split('version_')[0].format(hparams.seed_everything))))
    if len(versions) > 1:
        print('More than one version is available:', versions, '. Stopping.')
        sys.exit()
    version = versions[-1]
    experiment_path = hparams.experiment_path.format(hparams.seed_everything, version)

    experiment_dir = experiment_dir.format(hparams.seed_everything, 0)
    print('Original experiment path:', experiment_path)
    print('Results experiment path:', experiment_dir)
    # print(construct_experiment_dir(hparams))

    data = load_last_nstep_imputations_from_experiment(experiment_path,
                                                    last_n_steps=hparams.last_n_imputations,
                                                    load_every_nth_step=hparams.every_nth_step,
                                                    filename_prefix='imputations_')

    # data['imputations'] = data['imputations'][:, :, -hparams.last_n_imputations::hparams.every_nth_step]
    # else:
    #     experiment_dir = experiment_dir.format(0)
    #     print('Results experiment path:', experiment_dir)

    #     data = {}
    #     data['imputations'] = []
    #     for seed in hparams.eval_seeds:
    #         # Load imputations
    #         versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.experiment_path.format(seed).split('version_')[0])))
    #         if len(versions) > 1:
    #             print('More than one version is available:', versions, '. Stopping.')
    #             sys.exit()
    #         version = versions[-1]
    #         experiment_path = hparams.experiment_path.format(seed).format(version)

    #         data_s = load_last_nstep_imputations_from_experiment(experiment_path,
    #                                                     last_n_steps=hparams.last_n_imputations,
    #                                                     load_every_nth_step=hparams.every_nth_step,
    #                                                     filename_prefix='imputations_')

    #         for key in data_s.keys():
    #             if key == 'imputations':
    #                 continue
    #             if key not in data:
    #                 data[key] = data_s[key]
    #             else:
    #                 assert np.allclose(data[key], data_s[key]), f'Loaded data is not the same for key {key}.'

    #         data['imputations'].append(data_s['imputations'])
    #     data['imputations'] = np.concatenate(data['imputations'], axis=2)
    #     breakpoint()
    X_imp = rearrange(data['imputations'], 'b k t ... -> b (t k) ...')
    X_test_true_dataset = datamodule.test_data[:][0]
    if len(data['true_X'].shape) == 3:
        X_test_true = data['true_X'].squeeze(1)
    else:
        X_test_true = data['true_X']
    if datamodule.hparams.num_first_datapoints_test is not None and datamodule.hparams.num_first_datapoints_test > 0:
        X_imp = X_imp[:datamodule.hparams.num_first_datapoints_test]
        X_test_true = X_test_true[:datamodule.hparams.num_first_datapoints_test]
    masks = data['masks'].squeeze(1)
    data_idx = data['data_idx']

    assert np.allclose(X_test_true, X_test_true_dataset),\
        'The true test data is not the same as the true test data in the imputation experiment.'
    # true_targets = datamodule.test_data[:][-2]
    # true_targets=torch.tensor(true_targets).float()

    # if hparams.use_marginal_vae_samples_instead_of_imputations:
    #     X_imp = vae_marginal_samples_imputation(X_test_true_dataset, masks, num_imps=X_imp.shape[1], vae_model=model, batch_size=hparams.batch_size)

    # Print duration
    start = datetime.datetime.fromtimestamp(time.time())

    for metric in hparams.compute_metrics:
        if metric == 'sinkhorn_per_imputation':
            sinkhorn_per_imputation = sinkhorn_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                X_true_train=None, backend=hparams.backend,
                                                                scaling=0.9,
                                                                batch_size=hparams.batch_size,
                                                                verbose=False)

            # Save scores
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            filename = f'uci_imp_dataset_divergences_{metric}.npz'
            np.savez(os.path.join(experiment_dir, filename),
                    sinkhorn_per_imputation=sinkhorn_per_imputation,
                    )
        elif metric.startswith('sinkhorn_per_imputation_'):
            args_part = metric.split('sinkhorn_per_imputation')[1]
            p = 2
            if '_p' in args_part:
                p = int(args_part.split('p')[1].split('_')[0])
            blur = None
            if '_blur' in args_part:
                blur = float(args_part.split('blur')[1].split('_')[0])
            scaling = 0.9
            if '_scaling' in args_part:
                scaling = float(args_part.split('scaling')[1].split('_')[0])

            print(f'Using {p=}, {blur=}, {scaling=}')
            sinkhorn_per_imputation = sinkhorn_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                X_true_train=None, backend=hparams.backend,
                                                                scaling=scaling,
                                                                blur=blur,
                                                                p=p,
                                                                batch_size=hparams.batch_size,
                                                                verbose=False)

            # Save scores
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            filename = f'uci_imp_dataset_divergences_{metric}.npz'
            np.savez(os.path.join(experiment_dir, filename),
                    sinkhorn_per_imputation=sinkhorn_per_imputation,
                    )
        elif metric == 'energymmd_per_imputation':
            energymmd_per_imputation_ = energymmd_per_imputation(X_imp, X_test_true_dataset,
                                                                X_true_train=None, backend=hparams.backend,
                                                                batch_size=hparams.batch_size,
                                                                verbose=False)

            # Save scores
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            filename = f'uci_imp_dataset_divergences_{metric}.npz'
            np.savez(os.path.join(experiment_dir, filename),
                    energymmd_per_imputation=energymmd_per_imputation_,
                    )
        elif metric == 'gaussianmmd_per_imputation':
            gaussianmmd_per_imputation = gaussianmmd_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                X_true_train=None, backend=hparams.backend,
                                                                batch_size=hparams.batch_size,
                                                                verbose=False)

            # Save scores
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            filename = f'uci_imp_dataset_divergences_{metric}.npz'
            np.savez(os.path.join(experiment_dir, filename),
                    gaussianmmd_per_imputation=gaussianmmd_per_imputation,
                    )
        elif metric == 'laplacianmmd_per_imputation':
            laplacianmmd_per_imputation = laplacianmmd_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                X_true_train=None, backend=hparams.backend,
                                                                batch_size=hparams.batch_size,
                                                                verbose=False)

            # Save scores
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            filename = f'uci_imp_dataset_divergences_{metric}.npz'
            np.savez(os.path.join(experiment_dir, filename),
                    laplacianmmd_per_imputation=laplacianmmd_per_imputation,
                    )
        else:
            raise NotImplementedError(f'Unknown metric {metric}.')

    if hparams.process_irwg_finalresample:
        data = load_last_nstep_imputations_from_experiment(experiment_path,
                                                       last_n_steps=hparams.last_n_imputations,
                                                       load_every_nth_step=hparams.every_nth_step,
                                                       filename_prefix='irwg_imputations_after_final_resampling_')

        # data['imputations'] = data['imputations'][:, :, -hparams.last_n_imputations::hparams.every_nth_step]
        X_imp = rearrange(data['imputations'], 'b k t ... -> b (t k) ...')
        X_test_true_dataset = datamodule.test_data[:][0]
        X_test_true = data['true_X'].squeeze(1)
        if datamodule.hparams.num_first_datapoints_test is not None and datamodule.hparams.num_first_datapoints_test > 0:
            X_imp = X_imp[:datamodule.hparams.num_first_datapoints_test]
            X_test_true = X_test_true[:datamodule.hparams.num_first_datapoints_test]
        masks = data['masks'].squeeze(1)

        assert np.allclose(X_test_true, X_test_true_dataset),\
            'The true test data is not the same as the true test data in the imputation experiment.'

        for metric in hparams.compute_metrics:
            if metric == 'sinkhorn_per_imputation':
                sinkhorn_per_imputation = sinkhorn_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                    X_true_train=None, backend=hparams.backend,
                                                                    scaling=0.9,
                                                                    batch_size=hparams.batch_size,
                                                                    verbose=False)

                # Save scores
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)

                filename = f'uci_imp_dataset_divergences_{metric}_irwg_resampled.npz'
                np.savez(os.path.join(experiment_dir, filename),
                        sinkhorn_per_imputation=sinkhorn_per_imputation,
                        )
            elif metric.startswith('sinkhorn_per_imputation_'):
                args_part = metric.split('sinkhorn_per_imputation')[1]
                p = 2
                if '_p' in args_part:
                    p = int(args_part.split('p')[1].split('_')[0])
                blur = None
                if '_blur' in args_part:
                    blur = float(args_part.split('blur')[1].split('_')[0])
                scaling = 0.9
                if '_scaling' in args_part:
                    scaling = float(args_part.split('scaling')[1].split('_')[0])

                print(f'Using {p=}, {blur=}, {scaling=}')
                sinkhorn_per_imputation = sinkhorn_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                    X_true_train=None, backend=hparams.backend,
                                                                    scaling=scaling,
                                                                    blur=blur,
                                                                    p=p,
                                                                    batch_size=hparams.batch_size,
                                                                    verbose=False)

                # Save scores
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)

                filename = f'uci_imp_dataset_divergences_{metric}_irwg_resampled.npz'
                np.savez(os.path.join(experiment_dir, filename),
                        sinkhorn_per_imputation=sinkhorn_per_imputation,
                        )
            elif metric == 'energymmd_per_imputation':
                energymmd_per_imputation_ = energymmd_per_imputation(X_imp, X_test_true_dataset,
                                                                    X_true_train=None, backend=hparams.backend,
                                                                    batch_size=hparams.batch_size,
                                                                    verbose=False)

                # Save scores
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)

                filename = f'uci_imp_dataset_divergences_{metric}_irwg_resampled.npz'
                np.savez(os.path.join(experiment_dir, filename),
                        energymmd_per_imputation=energymmd_per_imputation_,
                        )
            elif metric == 'gaussianmmd_per_imputation':
                gaussianmmd_per_imputation = gaussianmmd_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                    X_true_train=None, backend=hparams.backend,
                                                                    batch_size=hparams.batch_size,
                                                                    verbose=False)

                # Save scores
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)

                filename = f'uci_imp_dataset_divergences_{metric}_irwg_resampled.npz'
                np.savez(os.path.join(experiment_dir, filename),
                        gaussianmmd_per_imputation=gaussianmmd_per_imputation,
                        )
            elif metric == 'laplacianmmd_per_imputation':
                laplacianmmd_per_imputation = laplacianmmd_per_imputation_autoblur(X_imp, X_test_true_dataset,
                                                                    X_true_train=None, backend=hparams.backend,
                                                                    batch_size=hparams.batch_size,
                                                                    verbose=False)

                # Save scores
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)

                filename = f'uci_imp_dataset_divergences_{metric}_irwg_resampled.npz'
                np.savez(os.path.join(experiment_dir, filename),
                        laplacianmmd_per_imputation=laplacianmmd_per_imputation,
                        )
            else:
                raise NotImplementedError(f'Unknown metric {metric}.')

    end = datetime.datetime.fromtimestamp(time.time())
    total_time = relativedelta.relativedelta(end, start)
    print('Finished in', "%d hours, %d minutes and %d seconds" % (total_time.hours, total_time.minutes, total_time.seconds))

if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
