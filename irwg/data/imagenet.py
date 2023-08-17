import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

data_filenames = {
    "imagenet32": {
        'orig_train': "imagenet32_train_data.npz",
        'orig_test': 'imagenet32_val_data.npz',
        'train': "imagenet32_train_data_train.npz",
        'val': "imagenet32_train_data_val.npz",
        'test': "imagenet32_val_data.npz",
    },
    "imagenet64": {
        'orig_train': "imagenet64_train_data.npz",
        'orig_test': 'imagenet64_val_data.npz',
        'train': "imagenet64_train_data_train.npz",
        'val': "imagenet64_train_data_val.npz",
        'test': "imagenet64_val_data.npz",
    }
}

def save_splits(data_root='./data', dataset='imagenet64'):
    filename = data_filenames[dataset]['orig_train']
    # Split the dataset according to the split in VDVAE paper

    trX = np.load(os.path.join(data_root, 'cifar10', filename), mmap_mode='r')['data']
    rng = np.random.default_rng(42)
    tr_va_split_indices = rng.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]

    # Save split data
    np.savez(os.path.join(data_root, 'cifar10', data_filenames[dataset]['train']), data=train)
    np.savez(os.path.join(data_root, 'cifar10', data_filenames[dataset]['val']), data=valid)


class ImageNet(data.Dataset):
    """
    A dataset wrapper for ImageNet
    """

    def __init__(self, root: str, dataset='imagenet64',
                 split: str = 'train',
                #  transform: Optional[Callable] = None,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            split:      data split, e.g. train, val, test
            # transforms: torchvision transforms to apply to the data
            rng:        random number generator used for noise
        """
        super().__init__()
        self.dataset = dataset

        filepath = os.path.join(root, 'imagenet', data_filenames[dataset][split])

        self.data = np.load(filepath)['data']

        # self.transform = transform

        # Preprocess in advance, so that we can modify the data after
        self.preprocess(rng=rng)

    def preprocess(self, rng):
        # Follow preprocessing from VDVAE paper
        # if self.dataset == 'imagenet32':
        #     shift = -116.2373
        #     scale = 1. / 69.37404
        #     untranspose = False
        # elif self.dataset == 'imagenet64':
        #     shift = -115.92961967
        #     scale = 1. / 69.37404
        #     untranspose = False
        # else:
        #     raise ValueError()

        # if untranspose:
        #     self.data = self.data.permute(0, 2, 3, 1)

        # self.data = self.data.astype(np.float32)
        # self.data += shift
        # self.data *= scale

        if self.dataset == 'imagenet32':
            img_size = 32
        elif self.dataset == 'imagenet64':
            img_size = 64
        img_size2 = img_size * img_size

        self.data = np.dstack((self.data[:, :img_size2], self.data[:, img_size2:2*img_size2], self.data[:, 2*img_size2:]))
        self.data = self.data.reshape((self.data.shape[0], img_size, img_size, 3))#.transpose(0, 3, 1, 2)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            img: image where target is index of the target class.
        """
        img = self.data[index]

        return img

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)
