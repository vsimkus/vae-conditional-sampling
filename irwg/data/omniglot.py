import os.path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image


def save_splits(seed=None):
    background_dataset = torchvision.datasets.Omniglot('./data', background=True, download=True)
    evaluation_dataset = torchvision.datasets.Omniglot('./data', background=False, download=True)

    # Load both datasets and transform to array
    background_data = [(torchvision.transforms.ToTensor()(background_dataset[i][0]).numpy(), background_dataset[i][1])
                       for i in range(len(background_dataset))]
    background_data, background_targets = map(list, zip(*background_data))
    background_data = np.squeeze(np.array(background_data), axis=1)
    background_targets = np.array(background_targets)

    evaluation_data = [(torchvision.transforms.ToTensor()(evaluation_dataset[i][0]).numpy(), evaluation_dataset[i][1])
                       for i in range(len(evaluation_dataset))]
    evaluation_data, evaluation_targets = map(list, zip(*evaluation_data))
    evaluation_data = np.squeeze(np.array(evaluation_data), axis=1)
    evaluation_targets = np.array(evaluation_targets)+np.max(background_targets)+1

    # Stack background and evaluation
    data = np.concatenate([background_data, evaluation_data], axis=0)
    targets = np.concatenate([background_targets, evaluation_targets], axis=0)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(data))
    idx = rng.permutation(idx)

    N_train = int(0.844 * len(data))
    N_val = int(0.006 * len(data))
    train_idx = idx[:N_train]
    val_idx = idx[N_train:N_train+N_val]
    test_idx = idx[N_train+N_val:]

    data_train, targets_train = data[train_idx], targets[train_idx]
    data_val, targets_val = data[val_idx], targets[val_idx]
    data_test, targets_test = data[test_idx], targets[test_idx]

    train = {
        'data': data_train,
        'targets': targets_train
    }

    val = {
        'data': data_val,
        'targets': targets_val
    }

    test = {
        'data': data_test,
        'targets': targets_test
    }

    # Save split data
    np.savez_compressed(os.path.join('./data', 'omniglot', 'train.npz'), **train)
    np.savez_compressed(os.path.join('./data', 'omniglot', 'val.npz'), **val)
    np.savez_compressed(os.path.join('./data', 'omniglot', 'test.npz'), **test)


class Omniglot(data.Dataset):
    """
    A dataset wrapper for Omniglot
    """

    def __init__(self, root: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 binarise_fixed: bool = False,
                 binarise_stochastic: bool = False,
                 rng: torch.Generator = None):
        """
        Args:
            root:                root directory that contains all data
            split:               data split, e.g. train, val, test
            transforms:          torchvision transforms to apply to the data
            binarise_fixed:      binarise the dataset once before running
            binarise_stochastic: binarise each batch differently
            rng:                 random number generator used for noise
        """
        super().__init__()

        filename = os.path.join(root, 'omniglot', f'{split}.npz')
        data = np.load(filename, allow_pickle=True)
        self.data = data['data']
        self.targets = data['targets']

        self.transform = transform
        self.binarise_fixed = binarise_fixed
        self.binarise_stochastic = binarise_stochastic

        assert not (self.binarise_fixed and self.binarise_stochastic),\
            'You can only use fixed binarisation or stochastic not both.'

        # Preprocess in advance, so that we can modify the data after
        self.preprocess(rng=rng)

    def preprocess(self, *, rng: torch.Generator):
        # Transform one sample to find out the shape of the transformed image
        temp = self.data[0]
        temp = Image.fromarray(temp, mode='F')
        if self.transform is not None:
            temp = self.transform(temp)
        temp = torchvision.transforms.ToTensor()(temp)
        temp = temp.flatten()

        # Transform all data
        original_data = self.data
        self.data = np.empty((len(original_data), *temp.shape), dtype=temp.numpy().dtype)
        for i, img in enumerate(original_data):
            img = Image.fromarray(img, mode='F')

            if self.transform is not None:
                img = self.transform(img)

            img = torchvision.transforms.ToTensor()(img).numpy()

            image_shape = img.shape[-2:]
            # We will use flatenned arrays
            img = img.flatten()

            self.data[i] = img

        self.image_shape = image_shape

        if self.binarise_fixed:
            self.data = (torch.rand(*self.data.shape, generator=rng).numpy() <= self.data).astype(self.data.dtype)
        elif self.binarise_stochastic:
            pass
        else:
            # Add noise and rescale to [0, 1]
            self.data = self.data*255
            self.data += torch.rand(*self.data.shape, generator=rng).numpy().astype(self.data.dtype)
            self.data /= 256

    @property
    def img_shape(self):
        return self.image_shape

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.binarise_stochastic:
            img = (np.random.rand(*img.shape) <= img).astype(self.data.dtype)

        return img, target

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    save_splits(seed=20220719)
