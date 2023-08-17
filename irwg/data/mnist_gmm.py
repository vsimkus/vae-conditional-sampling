import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.special as sspec
import torch
import torch.utils.data as data
import torchvision
from einops import rearrange

from irwg.data.mnist import MNIST

logit_margin = 0.15

def clip_unconstrained_samples_to_unconstraineddatarange(data):
    boundaries = np.array([0, 1])
    boundaries = preprocess(boundaries)
    return np.clip(data, boundaries[0], boundaries[1])

def preprocess(data):
    """
    Unconstrain data, such that Gaussian tails can correspond to valid values.
    """
    return sspec.logit(data*(1 - 2*logit_margin) + logit_margin)

def postprocess(data):
    """
    Transform data back to [0, 1] range.
    """
    # expit == sigmoid
    data = ((sspec.expit(data) - logit_margin) / (1. - 2*logit_margin))
    data = np.clip(data, 0., 1.)

    return data


def create_gmm_from_mnist():
    transform=torchvision.transforms.Resize((14,14),
                                            interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    dataset = MNIST(root='./data', split='train', transform=transform, binarise_fixed=False, binarise_stochastic=False, rng=None)
    data, labels = dataset[:]
    data = data.reshape(data.shape[0], -1)

    unconstrained_data = preprocess(data)

    comp_probs = np.ones(10, dtype=np.float32) / 10.
    means = np.empty((10, data.shape[-1]), dtype=np.float32)
    covs = np.empty((10, data.shape[-1], data.shape[-1]), dtype=np.float32)
    for i in range(10):
        idx = np.where(labels == i)
        data_i = unconstrained_data[idx]

        mean_i = np.mean(data_i, axis=0)
        cov_i = np.cov(data_i, rowvar=False)

        means[i] = mean_i
        covs[i] = cov_i

    return comp_probs, means, covs

def sample_mog(num_samples, comp_probs, means, *, covs=None, scale_trils=None):
    mix = torch.distributions.Categorical(probs=comp_probs)
    multi_norms = torch.distributions.MultivariateNormal(
        loc=means, covariance_matrix=covs, scale_tril=scale_trils)
    comp = torch.distributions.Independent(multi_norms, 0)
    mog = torch.distributions.MixtureSameFamily(mix, comp)

    return mog.sample(sample_shape=(num_samples,))

def sample_mog_and_get_component_idx(num_samples, comp_probs, means, *, covs=None, scale_trils=None):
    mix = torch.distributions.Categorical(probs=comp_probs)
    # multi_norms = torch.distributions.MultivariateNormal(
    #     loc=means, covariance_matrix=covs, scale_tril=scale_trils)
    # comp = torch.distributions.Independent(multi_norms, 0)
    # mog = torch.distributions.MixtureSameFamily(mix, comp)

    idx = mix.sample(sample_shape=(num_samples,))

    assert not (covs is not None and scale_trils is not None)
    if covs is not None:
        multi_norms = torch.distributions.MultivariateNormal(
            loc=means[idx], covariance_matrix=covs[idx])
    elif scale_trils is not None:
        multi_norms = torch.distributions.MultivariateNormal(
            loc=means[idx], covariance_matrix=covs[idx])

    return multi_norms.sample(), idx

def create_and_save_mnist_gmm_dataset(num_samples, root_dir='./data', filename='mnist_gmm_data.mat'):
    # Generate a Mixture-of-Gaussians distribution
    comp_probs, means, covs = create_gmm_from_mnist()
    comp_probs = torch.tensor(comp_probs)
    means = torch.tensor(means)
    covs = torch.tensor(covs)

    # Generate samples
    data_train, comp_train = sample_mog_and_get_component_idx(int(num_samples*0.9), comp_probs, means, covs=covs)
    data_train = data_train.float()
    comp_train = comp_train.flatten()
    data_val, comp_val = sample_mog_and_get_component_idx(int(num_samples*0.1), comp_probs, means, covs=covs)
    data_val = data_val.float()
    comp_val = comp_val.flatten()
    data_test, comp_test = sample_mog_and_get_component_idx(num_samples, comp_probs, means, covs=covs)
    data_test = data_test.float()
    comp_test = comp_test.flatten()

    data = {
        "train": data_train.numpy(),
        "val": data_val.numpy(),
        "test": data_test.numpy(),
        "train_comp": comp_train.numpy(),
        "val_comp": comp_val.numpy(),
        "test_comp": comp_test.numpy(),
        "comp_probs": comp_probs.numpy(),
        "means": means.numpy(),
        "covs": covs.numpy()
    }
    filename = os.path.join(root_dir, 'mnist_gmm', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    return data



class MNIST_GMM(data.Dataset):
    """
    A dataset wrapper
    """

    def __init__(self, root: str, filename='mnist_gmm_data',
                 split: str = 'train',
                 return_targets: bool = False,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            filename:   filename of the data file
            split:      data split, e.g. train, val, test
            return_targets: whether to return targets that are the component indices
            rng:        random number generator
        """
        super().__init__()
        self.return_targets = return_targets

        root = os.path.expanduser(root)
        filename = os.path.join(root, 'mnist_gmm', f'{filename}.mat')

        # Load Toy dataset
        self.data_file = sio.loadmat(filename)
        self.data = self.data_file[split]
        if self.return_targets:
            self.targets = self.data_file[f'{split}_comp'].flatten()

        self.data_min = np.min(self.data, axis=0)
        self.data_max = np.max(self.data, axis=0)

    def preprocess(self, data):
        return preprocess(data)

    def postprocess(self, data):
        return postprocess(data)

    def clip_unconstrained_samples_to_unconstraineddatarange(self, data):
        return clip_unconstrained_samples_to_unconstraineddatarange(data)

    def get_num_components(self):
        return self.data_file['comp_probs'].shape[-1]

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

class MNIST_GMM_customtest(MNIST_GMM):
    def __init__(self, *args, version=0, **kwargs):
        super().__init__(*args, **kwargs)

        if version == 0:
            self.data_idxs = [32, 36, 45, 119, 136, 176, 176, 191, 225, 229]
            self.masks = torch.tensor([
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False,  True,  True, False, False, False, False, False, False,False, False, False, False, False, False,  True,  True, False,False, False, False, False, False, False, False, False, False,False, False,  True,  True, False, False, False, False, False,False, False, False, False, False, False, False,  True,  True,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False,  True, False, False, False, False, False, False, False, False,  True, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False,  True, False, False, False, False, False, False, False, False, False,  True, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            ])
        elif version == 1:
            self.data_idxs = [36, 36, 36, 36, 36,
                              45, 45, 45, 45, 45,
                              136, 136, 136, 136, 136,
                              191, 191, 191, 191, 191,]
            self.masks = torch.tensor([
                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,False, False, False,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True, False, False, False,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True, False, False, False,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False],

                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,False, False, False,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True, False, False, False,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True, False, False, False,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False],

                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,False, False, False,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True, False, False, False,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True, False, False, False,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False],

                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False],
                [ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,False, False, False,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True, False, False, False,  True, True,  True,  True,  True,  True,  True,  True,  True,  True, True, False, False, False,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True, False, False, False,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True, False, False,False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True, False, False, False,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True, False,False, False,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False, False],
            ])
        else:
            raise ValueError(f'Version {version} not implemented.')

        self.data = self.data[self.data_idxs]
        self.data[~self.masks] = float('nan')
        self.data = torch.tensor(self.data)
        if self.return_targets:
            self.targets = self.targets[self.data_idxs]

if __name__ == '__main__':
    filename='mnist_gmm_data'
    # create_and_save_mnist_gmm_dataset(20000, root_dir='./data', filename=f'{filename}.mat')

    dataset = MNIST_GMM(root='./data', filename=filename, split='train')
    X = dataset[:]
    print(X.shape)

    X = dataset.postprocess(X)

    idx = np.random.randint(X.shape[0], size=49)
    X_i = X[idx]

    X_i = rearrange(X_i, '(n1 n2) (h w) -> (n1 h) (n2 w)', n1=7, n2=7, h=14, w=14)

    fig, axes = plt.subplots(1, 1, figsize=(12, 12), sharey=True)
    axes.imshow(X_i, cmap='gray')

    plt.show()

    # for i, cov in enumerate(dataset.data_file['covs']):
    #     cov = torch.tensor(cov)
    #     # cov = torch.diag(torch.diagonal(cov))
    #     cond = torch.linalg.cond(cov, p=2)
    #     print(f'Conditioning number of {i}-th covariance:', cond)
