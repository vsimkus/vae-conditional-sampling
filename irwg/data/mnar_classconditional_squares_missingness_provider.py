from functools import reduce
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from irwg.models.img_block_classcond_mnar_missingness_model import MNARBlockImgClassConditionalMissingnessModel


class MNARClassConditionalSquaresMissingness(Dataset):
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
                 max_concentration: float = 60,
                 num_patterns = None,
                 num_classes = None,
                 rng: torch.Generator = None):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.img_dims = img_dims
        self.dims_missing_idx = dims_missing_idx
        self.rng = rng

        self.miss_model = MNARBlockImgClassConditionalMissingnessModel.initialise(
                                num_classes=num_classes,
                                num_patterns=num_patterns,
                                img_shape=self.img_dims,
                                max_concentration=max_concentration,
                                rng=self.rng)

        # data = self._get_target_data()
        self.miss_mask = self.generate_mis_mask()

    def _get_target_data(self):
        data = self.dataset[:]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx]

        # if isinstance(data, np.ndarray):
        #     data = torch.tensor(data)

        return data

    def _get_class(self):
        data = self.dataset[:]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx+1]

        # if isinstance(data, np.ndarray):
        #     data = torch.tensor(data)

        return data

    def generate_mis_mask(self):
        clazz = self._get_class()
        return self.miss_model.sample_mask(clazz, self.rng)

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


