import datetime
import math
import os
import os.path
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.linalg as sl
import scipy.stats
import torch
from dateutil import relativedelta
from einops import asnumpy, rearrange, repeat
from pytorch_lightning.utilities.seed import reset_seed, seed_everything
from tqdm import trange

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
    parser = LightningArgumentParser('Imputation distribution assessment runner',
                                     parse_as_dict=False)

    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)

    parser.add_lightning_class_args(MissingDataModule, 'data')
    parser.add_argument('--vae_path', type=str, required=False, default=None,
                        help=('VAE for generating VAE-data.'))
    parser.add_argument('--load_best_vae', type=bool, default=None,
                        help=('If model_path is directoy use this flag to load the `epoch=*step=*.ckpt`. Throws error if there are multiple.'))

    parser.add_argument('--default_root_dir', type=str, default=None,
                        help=('Default path.'))
    parser.add_argument('--experiment_dir', type=str,
                        help=('Results path'))

    # Add experiment path
    parser.add_argument('--experiment_path', type=str, required=True,
                        help=('Path (or template) of the experiment directory'))

    # Use samples from VAE marginal instead of the imputations from the file for assessment of the evaluations.
    parser.add_argument('--use_marginal_vae_samples_instead_of_imputations', type=bool, default=False,
                        help=('Instead of using the samples from the file, it generates samples from the marginal of the VAE. Used to assess/debug the evaluation.'))

    # Imputation args
    parser.add_argument('--last_n_imputations', type=int, required=True,
                        help=('Number of last imputations'))
    parser.add_argument('--every_nth_step', type=int, required=True,
                        help=('Use every nth step imputations'))
    parser.add_argument('--batch_size', type=int, required=False,
                        help='Batch size.')
    parser.add_argument('--process_irwg_finalresample', type=bool, required=False, default=False,
                        help='Also processes the result file containing IRWG final-resampled imputations.')

    # Computation args
    # parser.add_argument('--cr_ci_level', type=float, required=True, default=0.95,
    #                     help=('Confidence interval level'))
    parser.add_argument('--quantile_levels', type=float, nargs='+', required=True,
                        help=('Quantile levels to estimate quantiles at.'))


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

def compute_centered_ci_cr(X_imp, X_test_true, masks, cr_ci_level, *, batch_size):
    all_imp_ci = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size), desc='Eval imputations (CI)'):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_imp_i = torch.tensor(X_imp_i).float().clone()

        masks_i = masks[batch_size*batch_idx:min(batch_size*(batch_idx+1), masks.shape[0])]
        masks_i = torch.tensor(masks_i).bool()

        X_imp_i[repeat(masks_i, 'b d -> b k d', k=X_imp_i.shape[1])] = float('nan')

        quantiles = torch.tensor([(1-cr_ci_level)/2, 1-(1-cr_ci_level)/2])
        ci = torch.nanquantile(X_imp_i, q=quantiles, dim=1)

        all_imp_ci.append(ci)

    all_imp_ci = rearrange(torch.cat(all_imp_ci, dim=1), 'ci b d -> b d ci')
    X_test_true_nan = torch.tensor(X_test_true).float().clone()
    X_test_true_nan[torch.tensor(masks).bool()] = float('nan')
    imputation_coverage = (all_imp_ci[..., 0] <= X_test_true_nan) * (X_test_true_nan <= all_imp_ci[..., 1])
    imputation_coverage = imputation_coverage.float()
    imputation_coverage[torch.tensor(masks).bool()] = float('nan')

    return all_imp_ci, imputation_coverage

def compute_expected_calibration_err(X_imp, X_test_true, masks, *, batch_size):
    X_test_true_nan = torch.tensor(X_test_true).float().clone()
    X_test_true_nan[torch.tensor(masks).bool()] = float('nan')

    quant_levels = np.linspace(0.05, 0.95, num=19, dtype=np.float32)

    all_coverage = []
    for batch_idx in trange(math.ceil(X_imp.shape[0]/batch_size), desc='Eval imputations (ECE)'):
        X_imp_i = X_imp[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]
        X_imp_i = torch.tensor(X_imp_i).float().clone()

        masks_i = masks[batch_size*batch_idx:min(batch_size*(batch_idx+1), masks.shape[0])]
        masks_i = torch.tensor(masks_i).bool()

        X_imp_i[repeat(masks_i, 'b d -> b k d', k=X_imp_i.shape[1])] = float('nan')

        X_test_true_nan_i = X_test_true_nan[batch_size*batch_idx:min(batch_size*(batch_idx+1), X_imp.shape[0])]

        cov_i = []
        for quant_level in quant_levels:

            ci = torch.nanquantile(X_imp_i, q=quant_level, dim=1)

            coverage = (X_test_true_nan_i <= ci)
            coverage = coverage.float()
            coverage[masks_i] = float('nan')

            cov_i.append(coverage)
        cov_i = torch.stack(cov_i, dim=1)

        all_coverage.append(cov_i)
    all_coverage = torch.concat(all_coverage, dim=0)

    ece = torch.mean(torch.abs(torch.nanmean(all_coverage, dim=0) - quant_levels[:, None]), dim=0)

    return ece

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

    # Load imputations
    versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.experiment_path.split('version_')[0])))
    if len(versions) > 1:
        print('More than one version is available:', versions, '. Stopping.')
        sys.exit()
    version = versions[-1]
    experiment_path = hparams.experiment_path.format(version)

    experiment_dir = experiment_dir.format(0)
    print('Original experiment path:', experiment_path)
    print('Results experiment path:', experiment_dir)
    # print(construct_experiment_dir(hparams))

    data = load_last_nstep_imputations_from_experiment(experiment_path,
                                                       last_n_steps=-1,
                                                       load_every_nth_step=1,
                                                       filename_prefix='imputations_')

    data['imputations'] = data['imputations'][:, :, -hparams.last_n_imputations::hparams.every_nth_step]
    X_imp = rearrange(data['imputations'], 'b k t ... -> b (t k) ...')
    X_test_true_dataset = datamodule.test_data[:][0]
    X_test_true = data['true_X'].squeeze(1)
    masks = data['masks'].squeeze(1)
    data_idx = data['data_idx']

    assert np.allclose(X_test_true, X_test_true_dataset),\
        'The true test data is not the same as the true test data in the imputation experiment.'
    # true_targets = datamodule.test_data[:][-2]
    # true_targets=torch.tensor(true_targets).float()

    if hparams.use_marginal_vae_samples_instead_of_imputations:
        X_imp = vae_marginal_samples_imputation(X_test_true_dataset, masks, num_imps=X_imp.shape[1], vae_model=model, batch_size=hparams.batch_size)

    # Print duration
    start = datetime.datetime.fromtimestamp(time.time())

    # ci, imp_coverage = compute_centered_ci_cr(X_imp, X_test_true, masks, hparams.cr_ci_level, batch_size=hparams.batch_size)

    # ece = compute_expected_calibration_err(X_imp, X_test_true, masks, batch_size=hparams.batch_size)

    all_quantiles = compute_quantiles(X_imp, masks, quantile_levels=hparams.quantile_levels, batch_size=hparams.batch_size)

    # Save scores
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    np.savez(os.path.join(experiment_dir, f'uci_imp_ci.npz'),
            # ci=asnumpy(ci),
            # imputation_coverage=asnumpy(imp_coverage),
            # ece=asnumpy(ece)

            quantiles_levels=hparams.quantile_levels,
            all_quantiles=asnumpy(all_quantiles),
            data_idx=asnumpy(data_idx)
            )

    if hparams.process_irwg_finalresample:
        data = load_last_nstep_imputations_from_experiment(experiment_path,
                                                       last_n_steps=-1,
                                                       load_every_nth_step=1,
                                                       filename_prefix='irwg_imputations_after_final_resampling_')

        data['imputations'] = data['imputations'][:, :, -hparams.last_n_imputations::hparams.every_nth_step]
        X_imp = rearrange(data['imputations'], 'b k t ... -> b (t k) ...')
        X_test_true_dataset = datamodule.test_data[:][0]
        X_test_true = data['true_X'].squeeze(1)
        masks = data['masks'].squeeze(1)

        assert np.allclose(X_test_true, X_test_true_dataset),\
            'The true test data is not the same as the true test data in the imputation experiment.'

        # ci, imp_coverage = compute_centered_ci_cr(X_imp, X_test_true, masks, hparams.cr_ci_level, batch_size=hparams.batch_size)

        # ece = compute_expected_calibration_err(X_imp, X_test_true, masks, batch_size=hparams.batch_size)

        all_quantiles = compute_quantiles(X_imp, masks, quantile_levels=hparams.quantile_levels, batch_size=hparams.batch_size)

        # Save scores
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        np.savez(os.path.join(experiment_dir, f'uci_imp_ci_irwg_resampled.npz'),
                # ci=asnumpy(ci),
                # imputation_coverage=asnumpy(imp_coverage),
                # ece=asnumpy(ece)

                quantiles_levels=hparams.quantile_levels,
                all_quantiles=asnumpy(all_quantiles),
                data_idx=asnumpy(data_idx)
                )



    end = datetime.datetime.fromtimestamp(time.time())
    total_time = relativedelta.relativedelta(end, start)
    print('Finished in', "%d hours, %d minutes and %d seconds" % (total_time.hours, total_time.minutes, total_time.seconds))

if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
