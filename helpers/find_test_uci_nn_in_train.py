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

def construct_experiment_dir(args):
    # return f'{args.experiment_subdir_base}/seed_m{args.seed_everything}_d{args.data.setup_seed}'
    versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(args.experiment_path.split('version_')[0])))
    if len(versions) > 1:
        print('More than one version is available:', versions, '. Stopping.')
        sys.exit()
    version = versions[-1]
    return args.experiment_dir.format(version)

def build_argparser():
    parser = LightningArgumentParser('Nearest-neighbour finder for test data',
                                     parse_as_dict=False)

    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)

    parser.add_lightning_class_args(MissingDataModule, 'data')
    # parser.add_argument('--vae_path', type=str, required=False, default=None,
    #                     help=('VAE for generating VAE-data.'))
    # parser.add_argument('--load_best_vae', type=bool, default=None,
    #                     help=('If model_path is directoy use this flag to load the `epoch=*step=*.ckpt`. Throws error if there are multiple.'))

    parser.add_argument('--default_root_dir', type=str, default=None,
                        help=('Default path.'))
    parser.add_argument('--experiment_dir', type=str,
                        help=('Results path'))

    # Add experiment path
    # parser.add_argument('--experiment_path', type=str, required=True,
    #                     help=('Path (or template) of the experiment directory'))

    # Use samples from VAE marginal instead of the imputations from the file for assessment of the evaluations.
    # parser.add_argument('--use_marginal_vae_samples_instead_of_imputations', type=bool, default=False,
    #                     help=('Instead of using the samples from the file, it generates samples from the marginal of the VAE. Used to assess/debug the evaluation.'))

    # Imputation args
    # parser.add_argument('--last_n_imputations', type=int, required=True,
    #                     help=('Number of last imputations'))
    # parser.add_argument('--every_nth_step', type=int, required=True,
    #                     help=('Use every nth step imputations'))
    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size.')
    # parser.add_argument('--process_irwg_finalresample', type=bool, required=False, default=False,
    #                     help='Also processes the result file containing IRWG final-resampled imputations.')

    # Computation args
    parser.add_argument('--num_nearest_neighbours', type=int, required=False,
                        help='Number of nearest neighbors to find.')
    # parser.add_argument('--cr_ci_level', type=float, required=True, default=0.95,
    #                     help=('Confidence interval level'))
    parser.add_argument('--quantile_levels', type=float, nargs='+', required=True,
                        help=('Quantile levels to estimate quantiles at.'))


    return parser

def find_nearest_neighbours(X_query, M_query, X_database, *, k):
    unique_patterns, rev_index = np.unique(M_query, axis=0, return_inverse=True)
    nn_indices = np.zeros((X_query.shape[0], k), dtype=int)
    nn_distances = np.zeros((X_query.shape[0], k), dtype=float)
    for p, pattern in tqdm(enumerate(unique_patterns), desc='Processing patterns'):
        X_database_p = X_database[:, pattern]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        nbrs = nbrs.fit(X_database_p)
        distances, indices = nbrs.kneighbors(X_query[rev_index == p, :][:, pattern])

        nn_indices[rev_index == p] = indices
        nn_distances[rev_index == p] = distances

    return nn_indices, nn_distances

def impute_using_nn_indices(X_query, M_query, X_database, nn_indices):
    M_query = M_query[:, None, :]
    X_imp = X_database[nn_indices]*~M_query + X_query[:, None, :]*M_query

    return X_imp

def compute_quantiles(X_imp, masks, *, quantile_levels, batch_size):
    quantile_levels = torch.tensor(np.array(quantile_levels, dtype=np.float32))

    all_quantiles = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size), desc='Eval quantiles for imputations'):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_imp_i = torch.tensor(X_imp_i).float().clone()

        masks_i = masks[batch_size*batch_idx:min(batch_size*(batch_idx+1), masks.shape[0])]
        masks_i = torch.tensor(masks_i).bool()

        X_imp_i[repeat(masks_i, 'b d -> b k d', k=X_imp_i.shape[1])] = float('nan')

        quantiles_i = torch.nanquantile(X_imp_i, q=quantile_levels, dim=1)

        all_quantiles.append(quantiles_i)

    all_quantiles = torch.cat(all_quantiles, dim=1)
    all_quantiles = rearrange(all_quantiles, 'q b d -> b d q')

    return all_quantiles

def run(hparams):
    # Set random seed
    # NOTE: this must be done before any class initialisation,
    # hence also before the call to parser.instantiate_classes()
    seed_everything(hparams.seed_everything, workers=True)

    # Get the instantiated data
    # datamodule = hparams.data
    datamodule = MissingDataModule(**hparams.data)

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

    experiment_dir = experiment_dir.format(0)
    print('Results experiment path:', experiment_dir)

    X_test_true_dataset, M_test_true_dataset = datamodule.test_data[:][:2]
    X_train_true_dataset = datamodule.train_data[:][0]

    # Print duration
    start = datetime.datetime.fromtimestamp(time.time())

    nn_indices, nn_distances = find_nearest_neighbours(X_query=X_test_true_dataset, M_query=M_test_true_dataset,
                                                        X_database=X_train_true_dataset, k=hparams.num_nearest_neighbours)
    X_imp = impute_using_nn_indices(X_query=X_test_true_dataset, M_query=M_test_true_dataset,
                                    X_database=X_train_true_dataset, nn_indices=nn_indices)
    all_quantiles = compute_quantiles(X_imp, M_test_true_dataset,
                                      quantile_levels=hparams.quantile_levels, batch_size=hparams.batch_size)

    # Save scores
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    np.savez(os.path.join(experiment_dir, f'nn_stats.npz'),
            nn_indices=nn_indices,
            nn_distances=nn_distances,
            X_imp=X_imp,
            quantiles_levels=hparams.quantile_levels,
            all_quantiles=asnumpy(all_quantiles),
            )

    end = datetime.datetime.fromtimestamp(time.time())
    total_time = relativedelta.relativedelta(end, start)
    print('Finished in', "%d hours, %d minutes and %d seconds" % (total_time.hours, total_time.minutes, total_time.seconds))

if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
