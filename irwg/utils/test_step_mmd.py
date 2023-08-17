from typing import List, Tuple, Union
import math

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tqdm import tqdm

from irwg.sampling.imputation_metrics import mmd_linear_time_metric, mmd_metric
from irwg.utils.test_step_base import TestBase


class TestMMD(TestBase):
    """
    Computes reference MMD for specified sample sizes for a trained model

    Args:
        num_samples_y:          The number of samples from the model for Y
        num_samples_x:          The number of samples from the model for X
        repeat:                 Number of times to repeat for computing uncertainty
        fixed_y_samples:        Only sample Y once
        mmd_estimator_kernel:   Name of the kernel
        mmd_linear_time:        Use linear-time MMD estimator
    """
    def __init__(self,
                 num_samples_y: int,
                 num_samples_x: int,
                 repeat: int,
                 *,
                 fixed_y_samples: bool = False,
                 mmd_estimator_kernel: str = 'exp_avg_hamming',
                 mmd_linear_time: bool = False,
                 sampling_batchsize: int = -1
                ):
        super().__init__()
        self.save_hyperparameters()

    def set_model(self, model: pl.LightningModule):
        self.model = model

    def on_test_epoch_start(self):
        if self.hparams.fixed_y_samples:
            with torch.inference_mode():
                self.Y = self.sample_model(num_samples=self.hparams.num_samples_y)

    def sample_model(self, num_samples):
        if self.hparams.sampling_batchsize <= 0:
            return self.model.sample(num_samples=num_samples)
        else:
            X = []
            for t in range(math.ceil(num_samples / self.hparams.sampling_batchsize)):
                n = min(self.hparams.sampling_batchsize, num_samples - len(X)*self.hparams.sampling_batchsize)
                X.append(self.model.sample(num_samples=n).cpu())
            return torch.concat(X)

    def test_step(self,
                  batch: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
                  batch_idx: int) -> STEP_OUTPUT:
        # NOTE: The data is ignored
        # X, M = batch[:2]

        assert batch_idx == 0,\
            'Not supported for more than 1 batches!'

        with torch.inference_mode():
            mmds = []
            for t in tqdm(range(self.hparams.repeat), desc='MMD Repeats'):
                X = self.sample_model(num_samples=self.hparams.num_samples_x)
                if self.hparams.fixed_y_samples:
                    Y = self.Y
                else:
                    Y = self.sample_model(num_samples=self.hparams.num_samples_y)

                if self.hparams.mmd_linear_time:
                    mmd = mmd_linear_time_metric(X_imp=X,
                                                 X_ref=Y,
                                                 kernel=self.hparams.mmd_estimator_kernel)
                else:
                    mmd = mmd_metric(X_imp=X,
                                    X_ref=Y,
                                    kernel=self.hparams.mmd_estimator_kernel)

                mmds.append(mmd.cpu().numpy())

            avg_mmd = np.mean(mmds)
            mmd_std = np.std(mmds, ddof=1)

            self.logger.experiment.add_scalar('mmd_estimate/test', avg_mmd, 0)
            self.logger.experiment.add_scalar('mmd_estimate_std/test', mmd_std, 0)
