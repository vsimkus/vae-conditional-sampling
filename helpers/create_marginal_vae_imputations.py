import datetime
import math
import os
import os.path
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.linalg as sl
from sklearn.neighbors import NearestNeighbors

import scipy.stats
import torch
from dateutil import relativedelta
from einops import asnumpy, rearrange, repeat
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from tqdm import trange, tqdm

# from compute_scores_util import build_argparser
from pytorch_lightning.utilities.cli import LightningArgumentParser

from irwg.data import MissingDataModule
from irwg.models.vae import VAE
import irwg.models.uci_aux_models
from irwg.models.vae_resnet import ResNetEncoder
from irwg.models.resnet_classifier import ResNetClassifier
from irwg.models.neural_nets import ResidualFCNetwork

def construct_experiment_dir(args):
    # return f'{args.experiment_subdir_base}/seed_m{args.seed_everything}_d{args.data.setup_seed}'
    versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(args.experiment_path.split('version_')[0])))
    if len(versions) > 1:
        print('More than one version is available:', versions, '. Stopping.')
        sys.exit()
    version = versions[-1]
    return args.experiment_dir.format(version)

def build_argparser():
    parser = LightningArgumentParser('Generate imputations from VAE marginal distribution.',
                                     parse_as_dict=False)

    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)

    parser.add_lightning_class_args(MissingDataModule, 'data')
    parser.add_argument('--vae_path', type=str, required=False, default=None,
                        help=('VAE for generating VAE-data.'))
    parser.add_argument('--load_best_vae', type=bool, default=None,
                        help=('If model_path is directoy use this flag to load the `epoch=*step=*.ckpt`. Throws error if there are multiple.'))

    parser.add_argument('--use_gpu', type=bool, default=False,)

    parser.add_argument('--default_root_dir', type=str, default=None,
                        help=('Default path.'))
    parser.add_argument('--experiment_dir', type=str,
                        help=('Results path'))

    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size.')
    parser.add_argument('--num_imps', type=int, required=True,
                        help=('Number of imputations to generate.'))

    return parser

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

def run(hparams):
    # Set random seed
    # NOTE: this must be done before any class initialisation,
    # hence also before the call to parser.instantiate_classes()
    seed_everything(hparams.seed_everything, workers=True)

    # Get the instantiated data
    # datamodule = hparams.data
    datamodule = MissingDataModule(**hparams.data)

    # In case we want to generate data from the model
    if hparams.vae_path is not None:
        versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.vae_path.split('version_')[0])))
        if len(versions) > 1:
            print('More than one version is available:', versions, '. Loading ', versions[-1])
        version = versions[-1]
        model_path = hparams.vae_path.format(version)
        if os.path.isdir(model_path):
            models = os.listdir(model_path)
            if hparams.load_best_vae is not None and hparams.load_best_vae:
                models.remove('last.ckpt')
            if len(models) > 1:
                raise ValueError(f'Too many models in path: {model_path}')
            model_path = os.path.join(model_path, models[0])
        print('VAE model path:', model_path)
        model = VAE.load_from_checkpoint(checkpoint_path=model_path)
        datamodule.set_model(model)

        if hparams.use_gpu:
            model = model.cuda()

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


    experiment_dir = experiment_dir.format(hparams.seed_everything, 0)
    print('Results experiment path:', experiment_dir)

    all_data = datamodule.test_data[:]
    X_test_true_dataset, true_masks = all_data[:2]
    data_idx = all_data[-1]

    X_imp = vae_marginal_samples_imputation(X_test_true_dataset, true_masks, num_imps=hparams.num_imps,
                                            vae_model=model, batch_size=hparams.batch_size)

    # Save scores
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    filename = f'imputations_0.npz'
    np.savez(os.path.join(experiment_dir, filename),
            imputations=asnumpy(X_imp[None, ...]),
            true_X=asnumpy(X_test_true_dataset),
            masks=asnumpy(true_masks)[:, None],
            data_idx=asnumpy(data_idx),
            )

if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
