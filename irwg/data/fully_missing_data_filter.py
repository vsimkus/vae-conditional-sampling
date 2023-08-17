import numpy as np
import torch
from torch.utils.data import Dataset


class FullyMissingDataFilter(Dataset):
    """
    Removes samples that are fully missing.
    """
    def __init__(self, dataset: Dataset, miss_mask_idx: int = 1):
        self.dataset = dataset

        M = self.dataset[:][miss_mask_idx]
        if isinstance(M, np.ndarray):
            self.not_fully_missing_idx = np.argwhere(np.sum(M, axis=1) != 0).squeeze()
        elif isinstance(M, torch.Tensor):
            self.not_fully_missing_idx = torch.nonzero(torch.sum(M, dim=1) != 0, as_tuple=False).squeeze()
        else:
            raise TypeError('Invalid missingness mask type!')

    def __getitem__(self, idx):
        dataset_idx = self.not_fully_missing_idx[idx]
        return self.dataset[dataset_idx]

    def __setitem__(self, key, value):
        dataset_idx = self.not_fully_missing_idx[key]
        self.dataset[dataset_idx] = value

    def __len__(self):
        return len(self.not_fully_missing_idx)
