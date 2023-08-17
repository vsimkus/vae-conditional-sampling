from functools import reduce
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class UniformMissingnessDataset(Dataset):
    """
    Generates missing data in the given dataset.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.

    Args:
        dataset:            Fully-observed PyTorch dataset
        target_idx:         If the dataset returns tuples, then this should be the index
                            of the target data in the tuple for which the missing mask is added.
        total_miss:         Total fraction of values to be made missing
        dims_missing_idx:   Which dimensions should have missingness.
        rng:                Torch random number generator.
    """
    def __init__(self,
                 dataset: Dataset,
                 target_idx: int = 0,
                 total_miss: float = 0.00,
                 dims_missing_idx: List[int] = [-1,],
                 rng: torch.Generator = None):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.total_miss = total_miss
        self.dims_missing_idx = dims_missing_idx
        self.rng = rng

        data = self._get_target_data()
        self.miss_mask = self.generate_uniform_mask(data)

    def _get_target_data(self):
        data = self.dataset[:]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx]

        # if isinstance(data, np.ndarray):
        #     data = torch.tensor(data)

        return data

    def generate_uniform_mask(self, data):
        # Works for PyTorch and Numpy
        shape = np.array(data.shape)[[0,] + self.dims_missing_idx]
        total_values = reduce(lambda x, y: x*y, shape, 1)

        # Generate appropriate number of missing values
        miss_mask = torch.ones(total_values, dtype=torch.bool)
        miss_mask[:int(self.total_miss*total_values)] = 0

        # Randomise mask
        rand_idx = torch.randperm(total_values, generator=self.rng)
        miss_mask = miss_mask[rand_idx]

        return miss_mask.numpy().reshape(shape)

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
