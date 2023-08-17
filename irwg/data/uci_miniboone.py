import numpy as np
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

"""
Adapted from https://github.com/bayesiains/nsf/
"""

def load_miniboone(root):
    def load_data(path):
        # NOTE: To remember how the pre-processing was done.
        # data_ = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
        # print data_.head()
        # data_ = data_.as_matrix()
        # # Remove some random outliers
        # indices = (data_[:, 0] < -100)
        # data_ = data_[~indices]
        #
        # i = 0
        # # Remove any features that have too many re-occuring real values.
        # features_to_remove = []
        # for feature in data_.T:
        #     c = Counter(feature)
        #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
        #     if max_count > 5:
        #         features_to_remove.append(i)
        #     i += 1
        # data_ = data_[:, np.array([i for i in range(data_.shape[1]) if i not in features_to_remove])]
        # np.save("~/data_/miniboone/data_.npy", data_)

        data = np.load(path)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(path):
        data_train, data_validate, data_test = load_data(path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    return load_data_normalised(
        path=os.path.join(root, 'miniboone', 'data.npy')
    )


def save_splits(root):
    train, val, test = load_miniboone(root)
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join(root, 'miniboone', '{}.npz'.format(name))
        np.savez_compressed(file, data)


class UCI_MINIBOONE(Dataset):
    def __init__(self, root, split='train',
                 train_noise: float = None,
                 rng: torch.Generator = None):
        path = os.path.join(root, 'miniboone', '{}.npz'.format(split))
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
    dataset = UCI_MINIBOONE(root='./data', split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    # plt.hist(dataset.data.reshape(-1), bins=250)
    # plt.show()

    # fig, axes = plt.subplots(5, 9, figsize=(10, 7), sharex=True, sharey=True)
    # axes = axes.reshape(-1)
    # for i, dimension in enumerate(dataset.data.T):
    #     axes[i].hist(dimension, bins=100)
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
