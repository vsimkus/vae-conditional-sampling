from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from irwg.models.logistic_missingness_model import MNARLogisticModel, MNARSelfLogisticModel, MARLogisticModel


class LogisticMissingnessDataset(Dataset):
    """
    Generates missing data in the given dataset using a Logistic model.
    The missingness is denoted by adding additional tensor M,
    whose size is the same as X.

    Args:
        dataset:            Fully-observed PyTorch dataset
        target_idx:         If the dataset returns tuples, then this should be the index
                            of the target data in the tuple for which the missing mask is added.
        total_miss:         Total fraction of values to be made missing
        frac_input_vars: Fraction of variables that should be used as inputs to the logistic model.
        exclude_inputs:     Should the inputs to the logistic regression be incomplete too
        dims_missing_idx:   Which dimensions should have missingness.
        miss_type
        rng:                Torch random number generator.
    """
    def __init__(self,
                 dataset: Dataset,
                 target_idx: int = 0,
                 total_miss: float = 0.00,
                 dims_missing_idx: List[int] = [-1,],
                 frac_input_vars: float = 0.3,
                 exclude_inputs: bool = True,
                 miss_type: str = 'MNAR',
                 rng: torch.Generator = None):
        super().__init__()
        self.dataset = dataset
        self.target_idx = target_idx
        self.total_miss = total_miss
        self.dims_missing_idx = dims_missing_idx
        self.rng = rng
        self.miss_type = miss_type

        self.frac_input_vars = frac_input_vars
        self.exclude_inputs = exclude_inputs

        data = self._get_target_data()
        self.create_missingness_model(data)
        self.miss_mask = self.generate_mask(data)

    def _get_target_data(self):
        data = self.dataset[:]
        if isinstance(data, tuple):
            # Get the data for which the missing masks are generated
            data = data[self.target_idx]

        # if isinstance(data, np.ndarray):
        #     data = torch.tensor(data)

        return data

    def create_missingness_model(self, data):
        if self.miss_type == 'MNAR':
            self.miss_model = MNARLogisticModel.initialise_from_data(X=data,
                                                                    total_miss=self.total_miss,
                                                                    frac_input_vars=self.frac_input_vars,
                                                                    exclude_inputs=self.exclude_inputs,
                                                                    rng=self.rng)
        elif self.miss_type == 'MNAR_self':
            self.miss_model = MNARSelfLogisticModel.initialise_from_data(X=data,
                                                                    total_miss=self.total_miss,
                                                                    rng=self.rng)
        elif self.miss_type == 'MAR':
            self.miss_model = MARLogisticModel.initialise_from_data(X=data,
                                                                    total_miss=self.total_miss,
                                                                    frac_input_vars=self.frac_input_vars,
                                                                    rng=self.rng)

    def generate_mask(self, data):
        return self.miss_model.sample_mask(data, self.rng)

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
