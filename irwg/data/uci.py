
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data

from irwg.data.fetch_uci import dataset_loader as uci_dataset_loader


def load_and_split_dataset(root, dataset):
    data, targets = uci_dataset_loader(root, dataset)

    # Whiten data
    data = (data - data.mean(axis=0))/ data.std(axis=0, ddof=1)
    data = data.astype(np.float32)

    targets = targets.astype(np.float32)

    seed = 20221015
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data), axis=0)
    data = data[perm]
    targets = targets[perm]

    # Generate samples
    data_train = data[:int(len(data)*0.85)]
    data_val = data[int(len(data)*0.85):int(len(data)*0.9)]
    data_test = data[int(len(data)*0.9):]

    targets_train = targets[:int(len(data)*0.85)]
    targets_val = targets[int(len(data)*0.85):int(len(data)*0.9)]
    targets_test = targets[int(len(data)*0.9):]

    return data_train, data_val, data_test, targets_train, targets_val, targets_test


class UCIDataset(data.Dataset):
    """
    A dataset wrapper
    """

    def __init__(self, root: str,
                 dataset: str,
                 split: str = 'train',
                 *,
                 return_targets: bool = False,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            dataset:    dataset name
            split:      data split, e.g. train, val, test
            rng:        random number generator
        """
        super().__init__()

        root = os.path.join(os.path.expanduser(root), 'uci')

        # Load UCI dataset
        train, val, test, targets_train, targets_val, targets_test = load_and_split_dataset(root, dataset)
        if split == 'train':
            self.data = train
        elif split == 'val':
            self.data = val
        elif split == 'test':
            self.data = test

        self.return_targets = return_targets
        if return_targets:
            if split == 'train':
                self.targets = targets_train
            elif split == 'val':
                self.targets = targets_val
            elif split == 'test':
                self.targets = targets_test

    def __getitem__(self, index):
        """
        Args:
            index: index of sample
        Returns:
            image: dataset sample
        """
        if self.return_targets:
            return self.data[index], self.targets[index]
        return self.data[index]

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self):
        return self.data.shape[0]

