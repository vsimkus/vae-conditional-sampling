
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class QuadrantMissingnessDataset(Dataset):
    """
    Generates missing data in the given dataset.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.

    Args:
        dataset:        Fully-observed PyTorch dataset
        target_idx:     If the dataset returns tuples, then this should be the index
                        of the target data in the tuple for which the missing mask is added.
        total_miss:     Total fraction of values to be made missing
        # dims_missing_idx:   Which dimensions should have missingness.
        img_dims:       Used to calculate quadrants of the image.
        rng:            Torch random number generator.
    """
    def __init__(self,
                 dataset: Dataset,
                 target_idx: int = 0,
                 total_miss: float = 0.00,
                #  dims_missing_idx: List[int] = [-1,],
                 img_dims: List[int] = None,
                 rng: torch.Generator = None):
        super().__init__()

        assert total_miss in (0., 0.25, 0.5, 0.75),\
            'Missingness fraction can only be 0., 0.25, 0.5, or 0.75!'

        self.dataset = dataset
        self.target_idx = target_idx
        self.total_miss = total_miss
        # self.dims_missing_idx = dims_missing_idx
        self.rng = rng
        self.img_shape = img_dims

        data = self._get_target_data()
        self.miss_mask = self.generate_mis_mask(data)

    def _get_target_data(self):
        data = self.dataset[:]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx]

        # if isinstance(data, np.ndarray):
        #     data = torch.tensor(data)

        return data

    def generate_mis_mask(self, data):
        miss_mask = torch.ones(data.shape[0], *tuple(self.img_shape), dtype=torch.bool)

        if self.total_miss == 0.:
            return miss_mask.reshape(miss_mask.shape[0], -1).numpy()
        else:
            miss_quadrants = torch.multinomial(torch.ones((data.shape[0]), 4), num_samples=int(self.total_miss*4), replacement=False, generator=self.rng)

            # top-left
            miss_mask[(miss_quadrants == 0).sum(-1).bool(), :self.img_shape[0]//2, :self.img_shape[1]//2] = 0
            # top-right
            miss_mask[(miss_quadrants == 1).sum(-1).bool(), :self.img_shape[0]//2, self.img_shape[1]//2:] = 0
            # bot-left
            miss_mask[(miss_quadrants == 2).sum(-1).bool(), self.img_shape[1]//2:, :self.img_shape[1]//2] = 0
            # bot-right
            miss_mask[(miss_quadrants == 3).sum(-1).bool(), self.img_shape[1]//2:, self.img_shape[1]//2:] = 0

            return miss_mask.reshape(miss_mask.shape[0], -1).numpy()

    def __getitem__(self, idx):
        data = self.dataset[idx]
        miss_mask = self.miss_mask[idx]
        if isinstance(data, tuple):
            # Insert missingness mask after the target_idx tensor to which it corresponds
            data = (data[:self.target_idx+1]
                    + (miss_mask,)
                    + data[self.target_idx+1:])
        else:
            data = (data, miss_mask)

        return data

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def __len__(self):
        return len(self.dataset)
