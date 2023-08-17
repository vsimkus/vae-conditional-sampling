import os
import sys
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from einops import rearrange
from torch.utils.data import DataLoader

from irwg.data.real_and_imputed_data import RealAndImputedContrastiveData


def load_last_nstep_imputations_from_experiment(path, last_n_steps, load_every_nth_step):
    imp_files = [f for f in os.listdir(path) if 'imputations_' in f and f.endswith('.npz')]
    batch_idxs = [int(f.split('imputations_')[1].split('.npz')[0]) for f in imp_files]
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

class RealVsImputedDataModule(pl.LightningDataModule):
    def __init__(self, imputed_data_experiment_path,
                 last_n_iterations: int = -1,
                 load_every_nth_step: int = 1,
                 add_train_to_original: bool = True):
        """
        Args:
            imputed_data_experiment_path: path to the experiment folder containing the imputed data
            last_n_iterations: number of imputed data iterations to use. -1 means all
            load_every_nth_step: load every nth step of the imputed data
            add_train_to_original: add the original training data to the original test data
        """
        super().__init__()
        self.save_hyperparameters()

    def set_orig_datamodule(self, orig_datamodule):
        self.orig_datamodule = orig_datamodule

    @property
    def num_workers(self):
        return self.orig_datamodule.hparams.num_workers

    @property
    def batch_size(self):
        return self.orig_datamodule.hparams.batch_size

    def setup(self, stage: Optional[str] = None):
        # rng = torch.Generator()
        # rng = rng.manual_seed(self.hparams.setup_seed)

        if stage == 'fit':
            versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(self.hparams.imputed_data_experiment_path.split('version_')[0])))
            if len(versions) > 1:
                print('More than one version is available:', versions, '. Stopping.')
                sys.exit()
            version = versions[-1]
            path = self.hparams.imputed_data_experiment_path.format(version)
            imputed_data = load_last_nstep_imputations_from_experiment(path,
                                                                       last_n_steps=self.hparams.last_n_iterations,
                                                                       load_every_nth_step=self.hparams.load_every_nth_step)
            self.train_data = RealAndImputedContrastiveData(self.orig_datamodule, imputed_data, add_train_to_original=True)
        elif stage == 'test':
            raise NotImplementedError()

    def train_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.num_workers is None else self.num_workers
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=num_workers,
                          shuffle=True,
        )

    def test_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.num_workers is None else self.num_workers
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          num_workers=num_workers,
                          shuffle=False,
        )
