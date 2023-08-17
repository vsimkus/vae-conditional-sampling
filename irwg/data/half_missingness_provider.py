from functools import reduce
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class HalfMissingnessDataset(Dataset):
    """
    Generates missing data in the given dataset.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.

    Args:
        dataset:        Fully-observed PyTorch dataset
        top_miss:       Whether top half should be missing (True), or bottom half (False).
        target_idx:     If the dataset returns tuples, then this should be the index
                        of the target data in the tuple for which the missing mask is added.
        total_miss:     Total fraction of values to be made missing
        dims_missing_idx:   Which dimensions should have missingness.

        rng:            Torch random number generator.
    """
    def __init__(self,
                 dataset: Dataset,
                 target_idx: int = 0,
                 top_miss: bool = True,
                 total_miss: float = 0.00,
                 dims_missing_idx: List[int] = [-1,],
                 rng: torch.Generator = None):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.total_miss = total_miss
        self.dims_missing_idx = dims_missing_idx
        self.top_miss = top_miss
        self.rng = rng

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
        shape = np.array(data.shape)[[0,] + self.dims_missing_idx]
        num_dimensions = reduce(lambda x, y: x*y, np.array(data.shape)[self.dims_missing_idx], 1)

        miss_mask = torch.ones(*(data.shape[0], num_dimensions), dtype=torch.bool)
        if self.top_miss:
            miss_mask[:, :int(num_dimensions*self.total_miss)] = 0
        else:
            miss_mask[:, -int(num_dimensions*self.total_miss):] = 0

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
