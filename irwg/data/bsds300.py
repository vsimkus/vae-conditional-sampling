import os.path

import h5py
import numpy as np
import torch
import torch.utils.data as data


def load_bsds300(root):
    f = h5py.File(root + 'BSDS300/BSDS300.hdf5', 'r')

    data_train = f['train']
    data_val = f['validation']
    data_test = f['test']

    return data_train, data_val, data_test

def save_splits(root):
    train, val, test = load_bsds300(root)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(root, 'BSDS300', '{}.npz'.format(name))
        np.savez_compressed(file, data)

class BSDS300(data.Dataset):
    def __init__(self, root, split='train',
                 train_noise: float = None,
                 rng: torch.Generator = None):
        path = os.path.join(root, 'BSDS300', '{}.npz'.format(split))
        self.data = np.load(path)['arr_0'].astype(np.float32)
        self.n, self.dim = self.data.shape

        self.data_min = np.min(self.data, axis=0)
        self.data_max = np.max(self.data, axis=0)

        self.index = np.arange(0, len(self.data), dtype=np.int64)

        self.add_gaussian_noise = split == 'train' and train_noise is not None
        self.train_noise = train_noise

        self.rng = rng

    def __getitem__(self, index):
        """
        Args:
            index: index of sample
        Returns:
            image: dataset sample
        """
        x = self.data[index]
        if self.add_gaussian_noise:
            x = x + torch.normal(0, self.train_noise, size=x.shape, generator=self.rng).numpy()
        return x

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self):
        return self.data.shape[0]


