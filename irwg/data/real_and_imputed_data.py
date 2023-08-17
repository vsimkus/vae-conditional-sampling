
import os
import os.path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import scipy.stats
import torch
import torch.utils.data as data

from einops import repeat, rearrange


class RealAndImputedContrastiveData(data.Dataset):

    def __init__(self, original_datamodule, imputed_data, add_train_to_original=True):
        """
        """
        super().__init__()
        imputations = imputed_data['imputations']
        imputations = rearrange(imputations, 'b k t d -> (b k t) d')
        X_true = imputed_data['true_X'].squeeze(1)

        original_datamodule.setup('test')
        original_data = original_datamodule.test_data[:][0]
        assert np.allclose(X_true, original_data)

        if add_train_to_original:
            original_datamodule.setup('fit')
            original_data = np.concatenate([original_datamodule.train_data[:][0], original_data], axis=0)

        num_imputed = len(imputations)
        num_original = len(original_data)
        repeat_times = math.ceil(num_imputed / num_original)
        original_data = repeat(original_data, 'b d -> (k b) d', k=repeat_times)[: num_imputed]

        assert len(original_data) == len(imputations)

        self.targets = np.concatenate([np.ones(len(original_data)), np.zeros(len(imputations))], axis=0)
        self.data = np.concatenate([original_data, imputations], axis=0)


    def __getitem__(self, index):
        """
        Args:
            index: index of sample
        Returns:
            datapoint: dataset sample
            target: sample label (1 for original dataset, 1 for imputed dataset)
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]
