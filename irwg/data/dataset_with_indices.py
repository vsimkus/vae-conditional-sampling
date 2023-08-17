import numbers

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetWithIndices(Dataset):
    """
    Adds data index at the end of the returned values of any dataset
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.index_idx = 1
        if isinstance(self.dataset[0], tuple):
            self.index_idx = len(self.dataset[0])

    def __getitem__(self, idx):
        data = self.dataset[idx]
        idx = self._item_to_index(idx)

        if isinstance(data, tuple):
            return data + (idx,)
        else:
            return (data, idx)

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def _item_to_index(self, item):
        if isinstance(item, (numbers.Integral, np.ndarray, torch.Tensor)):
            return item
        if isinstance(item, slice):
            return np.arange(item.start or 0, item.stop or len(self), item.step or 1, dtype=np.int64)
        else:
            raise TypeError('{cls} indices must be integers, numpy arrays, tensors, or slices, not {idx}.'.format(
                cls=type(self.dataset).__name__,
                idx=type(item).__name__,
            ))

    def __len__(self):
        return len(self.dataset)
