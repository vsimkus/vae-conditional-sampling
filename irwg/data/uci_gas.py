import numpy as np
import os
from numpy.lib.function_base import corrcoef
import pandas as pd

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


"""
Adapted from https://github.com/bayesiains/nsf/
"""


def load_gas(root):
    def load_data(file):
        data = pd.read_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.sum(axis=1)
        return B

    def load_data_and_clean(file):
        data = load_data(file)
        B = get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        data = (data - data.mean()) / data.std()

        return data.values

    def load_data_and_clean_and_split(file):
        data = load_data_and_clean(file)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    return load_data_and_clean_and_split(
        file=os.path.join(root, 'gas', 'ethylene_CO.pickle')
    )


def save_splits(root):
    train, val, test = load_gas(root)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(root, 'gas', '{}.npz'.format(name))
        np.savez_compressed(file, data)


class UCI_GAS(Dataset):
    def __init__(self, root, split='train',
                 train_noise: float = None,
                 rng: torch.Generator = None):
        path = os.path.join(root, 'gas', '{}.npz'.format(split))
        self.data = np.load(path)['arr_0'].astype(np.float32)
        self.n, self.dim = self.data.shape

        self.data_min = np.min(self.data, axis=0)
        self.data_max = np.max(self.data, axis=0)

        # self.index = np.arange(0, len(self.data), dtype=np.long)

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
    dataset = UCI_GAS(root='./data', split='train')
    print(type(dataset.data))
    print("mean", dataset.data.mean(axis=0))
    print("median", np.median(dataset.data, axis=0))
    print(dataset.data.shape)
    print("min", dataset.data.min(axis=0))
    print("max", dataset.data.max(axis=0))
    print(np.where(dataset.data == dataset.data.max()))
    # fig, axs = plt.subplots(3, 3, figsize=(10, 7), sharex=True, sharey=True)
    # axs = axs.reshape(-1)
    # for i, dimension in enumerate(dataset.data.T):
    #     axs[i].hist(dimension, bins=100)
    # fig.tight_layout()
    # plt.show()

    # import seaborn as sns
    # import pandas as pd

    # sns.pairplot(pd.DataFrame(dataset.data),
    #              plot_kws={'s': 6},
    #              diag_kws={'bins': 25})
    # plt.tight_layout()
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
