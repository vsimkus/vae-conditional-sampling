import numpy as np
import os
import pandas as pd

import torch
from collections import Counter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

"""
Adapted from https://github.com/bayesiains/nsf/
"""

def load_hepmass(root):
    def load_data(path):

        data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_train.csv'),
                                 index_col=False)
        data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_test.csv'),
                                index_col=False)

        return data_train, data_test

    def load_data_no_discrete(path):
        """Loads the positive class examples from the first 10% of the dataset."""
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data_ set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(path):

        data_train, data_test = load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_test

    def load_data_no_discrete_normalised_as_array(path):

        data_train, data_test = load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.values, data_test.values

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[:, np.array(
            [i for i in range(data_train.shape[1]) if i not in features_to_remove])]
        data_test = data_test[:, np.array(
            [i for i in range(data_test.shape[1]) if i not in features_to_remove])]

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    return load_data_no_discrete_normalised_as_array(
        path=os.path.join(root, 'hepmass')
    )


def save_splits(root):
    train, val, test = load_hepmass(root)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(root, 'hepmass', '{}.npz'.format(name))
        np.savez_compressed(file, data)


class UCI_HEPMASS(Dataset):
    def __init__(self, root, split='train',
                 train_noise: float = None,
                 rng: torch.Generator = None):
        path = os.path.join(root, 'hepmass', '{}.npz'.format(split))
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


def main():
    dataset = UCI_HEPMASS(root='./data', split='train')
    print(type(dataset.data))
    print("mean", dataset.data.mean(axis=0))
    print("median", np.median(dataset.data, axis=0))
    print(dataset.data.shape)
    print("min", dataset.data.min(axis=0))
    print("max", dataset.data.max(axis=0))
    print(np.where(dataset.data == dataset.data.max()))
    # plt.hist(dataset.data.reshape(-1), bins=250)
    # plt.show()

    # fig, axes = plt.subplots(3, 7, figsize=(10, 7), sharex=True, sharey=True)
    # axes = axes.reshape(-1)
    # for i, dimension in enumerate(dataset.data.T):
    #     axes[i].hist(dimension, bins=250)
    # fig.tight_layout()
    # plt.draw()

    corrmat = np.corrcoef(dataset.data, rowvar=False)
    abs_corrmat = np.abs(corrmat)
    triang_indices = np.triu_indices_from(corrmat, k=1)
    print('Median abs correlation: ', np.median(abs_corrmat[triang_indices]))
    print('Mean abs correlation: ', np.mean(abs_corrmat[triang_indices]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs_corrmat, interpolation='nearest')
    fig.colorbar(cax)
    plt.draw()

    import scipy.stats as ss

    def multivariate_kendalltau(X):
        kendal_taus = []
        kendal_taus_p = []
        for d1 in range(X.shape[-1]):
            for d2 in range(d1+1, X.shape[-1]):
                t = ss.kendalltau(X[:, d1], X[:, d2])
                kendal_taus.append(t[0])
                kendal_taus_p.append(t[1])
        idx = np.triu_indices(X.shape[-1], k=1)

        taus = np.zeros((X.shape[-1],)*2)
        taus[idx] = np.array(kendal_taus)

        p = np.zeros((X.shape[-1],)*2)
        p[idx] = np.array(kendal_taus_p)

        return taus, p

    taus, taus_p = multivariate_kendalltau(dataset.data)
    abs_taus = np.abs(taus)
    print('Median abs Kendal-tau: ', np.median(abs_taus[triang_indices]))
    print('Mean abs Kendal-tau: ', np.mean(abs_taus[triang_indices]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(abs_taus, interpolation='nearest')
    fig.colorbar(cax)
    plt.show()


if __name__ == '__main__':
    main()
