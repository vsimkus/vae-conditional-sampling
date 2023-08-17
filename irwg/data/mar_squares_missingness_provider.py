from functools import reduce
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from irwg.utils.beta_binomial import BetaBinomial

class MARSquaresMissingness(Dataset):
    """
    Generates missing data in the given dataset.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.

    Args:
        dataset:        Fully-observed PyTorch dataset
        target_idx:     If the dataset returns tuples, then this should be the index
                        of the target data in the tuple for which the missing mask is added.
        dims_missing_idx:   Which dimensions should have missingness.
        img_dims:       Used to calculate quadrants of the image.
        rng:            Torch random number generator.
    """
    def __init__(self,
                 dataset: Dataset,
                 target_idx: int = 0,
                #  total_miss: float = 0.00,
                 dims_missing_idx: List[int] = [-1,],
                 img_dims: List[int] = None,
                 rng: torch.Generator = None,
                 block_types=1
                 ):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.img_dims = img_dims
        self.dims_missing_idx = dims_missing_idx
        self.rng = rng
        self.block_types = block_types

        self.miss_mask = self.generate_mis_mask()

    def generate_mis_mask(self):
        # (size_blocks, num_blocks)
        if self.block_types == 1:
            block_size_num_blocks = torch.tensor([[5, 10], [5, 12], [7, 5], [7, 6], [9, 3], [9, 4], [11, 2], [11, 3], [13, 1], [15, 1]])
        elif self.block_types == 2:
            block_size_num_blocks = torch.tensor([[10, 10], [10, 12], [14, 5], [14, 6], [18, 3], [18, 4], [22, 2], [22, 3], [26, 1], [30, 1]])

        # Sample locations from a Beta distribution
        location_distr = BetaBinomial(torch.tensor([3., 3.]), torch.tensor([3., 3.]), total_count=torch.tensor(self.img_dims)-1,)

        patterns = torch.zeros(len(self.dataset), torch.prod(torch.tensor(self.img_dims)), dtype=torch.bool)
        for i in range(len(self.dataset)):
            idx = torch.randint(len(block_size_num_blocks), size=(1,), generator=self.rng)[0]
            size_blocks, num_blocks = block_size_num_blocks[idx]

            locs = location_distr.sample((num_blocks,), generator=self.rng).long()

            mask = torch.ones(self.img_dims, dtype=torch.bool)
            for j in range(num_blocks):
                x, y = locs[j]
                x_min = max(x - torch.div(size_blocks, 2, rounding_mode='floor'), 0)
                x_max = min(x + torch.div(size_blocks, 2, rounding_mode='floor'), self.img_dims[0])
                y_min = max(y - torch.div(size_blocks, 2, rounding_mode='floor'), 0)
                y_max = min(y + torch.div(size_blocks, 2, rounding_mode='floor'), self.img_dims[0])
                mask[x_min:x_max, y_min:y_max] = 0.0

            patterns[i] = mask.flatten()
        return patterns


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


