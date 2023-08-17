import os
import pickle
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data

def flatten(outer):
    return [el for inner in outer for el in inner]

def load_cifar10(data_root, split):
              #one_hot=True):
    if split in ['train', 'val']:
        tr_data = [unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'data_batch_%d' % i)) for i in range(1, 6)]
        trX = np.vstack(data['data'] for data in tr_data)
        trY = np.asarray(flatten([data['labels'] for data in tr_data]))

        trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        trX, vaX, trY, vaY = train_test_split(trX, trY, test_size=5000, random_state=11172018)

        if split == 'train':
            return trX#, trY
        elif split == 'val':
            return vaX#, vaY
    elif split == 'test':
        te_data = unpickle_cifar10(os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
        teX = np.asarray(te_data['data'])
        # teY = np.asarray(te_data['labels'])
        teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        return teX#, teY

    # if one_hot:
    #     trY = np.eye(10, dtype=np.float32)[trY]
    #     vaY = np.eye(10, dtype=np.float32)[vaY]
    #     teY = np.eye(10, dtype=np.float32)[teY]
    # else:
    #     trY = np.reshape(trY, [-1, 1])
    #     vaY = np.reshape(vaY, [-1, 1])
    #     teY = np.reshape(teY, [-1, 1])
    # return (trX, trY), (vaX, vaY), (teX, teY)

class CIFAR10(data.Dataset):
    """
    A dataset wrapper for CIFAR10
    """

    def __init__(self, root: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            split:      data split, e.g. train, val, test
            transforms: torchvision transforms to apply to the data
            rng:        random number generator used for noise
        """
        super().__init__()

        root = os.path.join(root, 'cifar10')
        self.data = load_cifar10(root, split=split)

        self.transform = transform

        # Preprocess in advance, so that we can modify the data after
        # self.preprocess(rng=rng)

    # def preprocess(self, rng):
    #     # Follow preprocessing from VDVAE paper
    #     shift = -120.63838
    #     scale = 1. / 64.16736
    #     untranspose = False

    #     if untranspose:
    #         self.data = self.data.permute(0, 2, 3, 1)

    #     # self.data = self.data.astype(np.float32)
    #     # self.data += shift
    #     # self.data *= scale

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            img: image where target is index of the target class.
        """
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)
