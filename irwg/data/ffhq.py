import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data

data_filenames = {
    "ffhq256": {
        'orig': "ffhq-256.npy",
        'train': "ffhq-256-train.npy",
        'val': "ffhq-256-val.npy",
        'test': "ffhq-256-val.npy",
    }
}

def save_splits(data_root='./data', dataset='ffhq256'):
    filename = data_filenames[dataset]['orig']
    # Split the dataset according to the split in VDVAE paper

    trX = np.load(os.path.join(data_root, 'ffhq', filename), mmap_mode='r')
    rng = np.random.default_rng(5)
    tr_va_split_indices = rng.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # They did not significantly tune hyperparameters on ffhq-256/1024, and so simply evaluate on the test set

    # Save split data
    np.save(os.path.join(data_root, 'ffhq', data_filenames[dataset]['train']), train)
    np.save(os.path.join(data_root, 'ffhq', data_filenames[dataset]['val']), valid)


class FFHQ(data.Dataset):
    """
    A dataset wrapper for FFHQ
    """

    def __init__(self, root: str, dataset='ffhq256',
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

        filepath = os.path.join(root, 'ffhq', data_filenames[dataset][split])
        self.data = np.load(filepath)

        # self.transform = transform

        # Preprocess in advance, so that we can modify the data after
        self.preprocess(rng=rng)

    def preprocess(self, rng):
        # Follow preprocessing from VDVAE paper
        if self.dataset == 'ffhq256':
            untranspose = False

            # shift = -112.8666757481
            # scale = 1. / 69.84780273
        elif self.dataset == 'ffhq1024':
            untranspose = True

            # shift = -0.4387
            # scale = 1.0 / 0.2743
        else:
            raise ValueError()

        if untranspose:
            self.data = self.data.permute(0, 2, 3, 1)

        # self.data = self.data.astype(np.float32)
        # self.data += shift
        # self.data *= scale

    # def preprocess_func(self, x):
    #     'takes in a data example and returns the preprocessed input'
    #     'as well as the input processed for the loss'

    #     shift_loss = -127.5
    #     scale_loss = 1. / 127.5
    #     if self.dataset == 'ffhq256':
    #         shift = -112.8666757481
    #         scale = 1. / 69.84780273
    #         do_low_bit = True
    #         untranspose = False
    #     elif self.dataset == 'ffhq1024':
    #         shift = -0.4387
    #         scale = 1.0 / 0.2743
    #         shift_loss = -0.5
    #         scale_loss = 2.0
    #         do_low_bit = False
    #         untranspose = True
    #     else:
    #         raise ValueError('unknown dataset: ', self.dataset)

    #     inp = x[0].cuda(non_blocking=True).float()
    #     out = inp.clone()
    #     inp.add_(shift).mul_(scale)
    #     if do_low_bit:
    #         # 5 bits of precision
    #         out.mul_(1. / 8.).floor_().mul_(8.)
    #     out.add_(shift_loss).mul_(scale_loss)
    #     return inp, out

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
