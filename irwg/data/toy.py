
import os
import os.path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
import scipy.stats
import torch
import torch.utils.data as data

# from irwg.utils.incomplete_data_multivariate_normal import IncompleteDataMultivariateNormal


def generate_mog_model(num_components: int, dims: int, standardise: bool = True, convert_model_to_float: bool = True):
    """
    Generates a random Mixture of Gaussians model

    Args:
        num_components:   Number of components in the mixture
        dims:             Dimensionality of the data
        standardise:      If True standardise the marginal distribution of the mixture to unit variance
    Returns:

    """
    # Generate component mixture
    comp_probs_truth = scipy.stats.dirichlet.rvs(
        np.ones(num_components), size=1)[0]
    # comp_logits_truth = np.log(comp_probs_truth)

    # Generate Gaussians
    covs_truth = scipy.stats.invwishart.rvs(df=dims, scale=np.eye(dims), size=num_components)
    means_truth = np.random.randn(num_components, dims)*3

    if standardise:
        # Compute the mean of the MoG
        mean = (means_truth*comp_probs_truth[:, None]).sum(0)
        # Compute the covariance of the MoG
        dif = means_truth - mean
        cov = covs_truth + dif[..., None] @ dif[:, None, :]
        cov *= comp_probs_truth[:, None, None]
        cov = cov.sum(0)

        # Get standard deviation of marginals
        std = np.diagonal(cov)**(0.5)

        # Standardise the covariance matrices
        L = (np.linalg.cholesky(covs_truth)/std[None, :, None])
        covs_truth = L @ L.transpose((0, 2, 1))

        # Standardise the means
        means_truth = (means_truth-mean)/std

    comp_probs_truth = torch.tensor(comp_probs_truth)
    means_truth = torch.tensor(means_truth)
    covs_truth = torch.tensor(covs_truth)
    if convert_model_to_float:
        comp_probs_truth = comp_probs_truth.float()
        means_truth = means_truth.float()
        covs_truth = covs_truth.float()#+torch.eye(covs_truth.shape[-1], covs_truth.shape[-1])*1e-6
        # L_truth = torch.linalg.cholesky(covs_truth).float()+torch.eye(covs_truth.shape[-1], covs_truth.shape[-1])*1e-6
        # covs_truth = L_truth @ L_truth.transpose(-2, -1)

    return comp_probs_truth, means_truth, covs_truth


def create_grid2dmog_model(standardise: bool = True, convert_model_to_float: bool = True, version=1):
    if version == 1:
        sigma_1 = 2.
        gauss1 = torch.distributions.MultivariateNormal(torch.tensor([0., -15.]), torch.tensor([[sigma_1**2, 0.], [0., sigma_1**2]]))
        sigma_2 = 1.75
        gauss2 = torch.distributions.MultivariateNormal(torch.tensor([7.5, -7.5]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss3 = torch.distributions.MultivariateNormal(torch.tensor([0., -7.5]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss4 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, -7.5]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        sigma_3 = 1.5
        gauss5 = torch.distributions.MultivariateNormal(torch.tensor([-15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss6 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss7 = torch.distributions.MultivariateNormal(torch.tensor([0., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss8 = torch.distributions.MultivariateNormal(torch.tensor([7.5, 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss9 = torch.distributions.MultivariateNormal(torch.tensor([15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        sigma_4 = 1.25
        gauss10 = torch.distributions.MultivariateNormal(torch.tensor([3.75, 7.5]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss11 = torch.distributions.MultivariateNormal(torch.tensor([0., 7.5]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss12 = torch.distributions.MultivariateNormal(torch.tensor([-3.75, 7.5]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        sigma_5 = 1.
        gauss13 = torch.distributions.MultivariateNormal(torch.tensor([0., 15]), torch.tensor([[sigma_5**2, 0.], [0., sigma_5**2]]))

        covs_truth = torch.stack([gauss1.covariance_matrix,
                                gauss2.covariance_matrix,
                                gauss3.covariance_matrix,
                                gauss4.covariance_matrix,
                                gauss5.covariance_matrix,
                                gauss6.covariance_matrix,
                                gauss7.covariance_matrix,
                                gauss8.covariance_matrix,
                                gauss9.covariance_matrix,
                                gauss10.covariance_matrix,
                                gauss11.covariance_matrix,
                                gauss12.covariance_matrix,
                                gauss13.covariance_matrix])
        means_truth = torch.stack([gauss1.mean,
                                gauss2.mean,
                                gauss3.mean,
                                gauss4.mean,
                                gauss5.mean,
                                gauss6.mean,
                                gauss7.mean,
                                gauss8.mean,
                                gauss9.mean,
                                gauss10.mean,
                                gauss11.mean,
                                gauss12.mean,
                                gauss13.mean,])
    elif version == 2:
        sigma_1 = 2.
        gauss9 = torch.distributions.MultivariateNormal(torch.tensor([0., -15.]), torch.tensor([[sigma_1**2, 0.], [0., sigma_1**2]]))
        sigma_2 = 1.75
        gauss7 = torch.distributions.MultivariateNormal(torch.tensor([7.5, -7.5]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss5 = torch.distributions.MultivariateNormal(torch.tensor([0., -7.5]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss12 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, -7.5]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        sigma_3 = 1.5
        gauss3 = torch.distributions.MultivariateNormal(torch.tensor([-15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss10 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss1 = torch.distributions.MultivariateNormal(torch.tensor([0., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss13 = torch.distributions.MultivariateNormal(torch.tensor([7.5, 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss2 = torch.distributions.MultivariateNormal(torch.tensor([15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        sigma_4 = 1.25
        gauss4 = torch.distributions.MultivariateNormal(torch.tensor([3.75, 7.5]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss6 = torch.distributions.MultivariateNormal(torch.tensor([0., 7.5]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss11 = torch.distributions.MultivariateNormal(torch.tensor([-3.75, 7.5]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        sigma_5 = 1.
        gauss8 = torch.distributions.MultivariateNormal(torch.tensor([0., 15]), torch.tensor([[sigma_5**2, 0.], [0., sigma_5**2]]))

        covs_truth = torch.stack([gauss1.covariance_matrix,
                                gauss2.covariance_matrix,
                                gauss3.covariance_matrix,
                                gauss4.covariance_matrix,
                                gauss5.covariance_matrix,
                                gauss6.covariance_matrix,
                                gauss7.covariance_matrix,
                                gauss8.covariance_matrix,
                                gauss9.covariance_matrix,
                                gauss10.covariance_matrix,
                                gauss11.covariance_matrix,
                                gauss12.covariance_matrix,
                                gauss13.covariance_matrix])
        means_truth = torch.stack([gauss1.mean,
                                gauss2.mean,
                                gauss3.mean,
                                gauss4.mean,
                                gauss5.mean,
                                gauss6.mean,
                                gauss7.mean,
                                gauss8.mean,
                                gauss9.mean,
                                gauss10.mean,
                                gauss11.mean,
                                gauss12.mean,
                                gauss13.mean,])
    elif version == 3:
        sigma_1 = 2.
        gauss9 = torch.distributions.MultivariateNormal(torch.tensor([0., -30.]), torch.tensor([[sigma_1**2, 0.], [0., sigma_1**2]]))
        sigma_2 = 1.75
        gauss7 = torch.distributions.MultivariateNormal(torch.tensor([15, -15]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss5 = torch.distributions.MultivariateNormal(torch.tensor([0., -15]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss12 = torch.distributions.MultivariateNormal(torch.tensor([-15, -15]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        sigma_3 = 1.5
        gauss3 = torch.distributions.MultivariateNormal(torch.tensor([-30., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss10 = torch.distributions.MultivariateNormal(torch.tensor([-15, 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss1 = torch.distributions.MultivariateNormal(torch.tensor([0., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss13 = torch.distributions.MultivariateNormal(torch.tensor([15, 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss2 = torch.distributions.MultivariateNormal(torch.tensor([30., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        sigma_4 = 1.25
        gauss4 = torch.distributions.MultivariateNormal(torch.tensor([7.5, 15]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss6 = torch.distributions.MultivariateNormal(torch.tensor([0., 15]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss11 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, 15]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        sigma_5 = 1.
        gauss8 = torch.distributions.MultivariateNormal(torch.tensor([0., 30]), torch.tensor([[sigma_5**2, 0.], [0., sigma_5**2]]))

        covs_truth = torch.stack([gauss1.covariance_matrix,
                                gauss2.covariance_matrix,
                                gauss3.covariance_matrix,
                                gauss4.covariance_matrix,
                                gauss5.covariance_matrix,
                                gauss6.covariance_matrix,
                                gauss7.covariance_matrix,
                                gauss8.covariance_matrix,
                                gauss9.covariance_matrix,
                                gauss10.covariance_matrix,
                                gauss11.covariance_matrix,
                                gauss12.covariance_matrix,
                                gauss13.covariance_matrix])
        means_truth = torch.stack([gauss1.mean,
                                gauss2.mean,
                                gauss3.mean,
                                gauss4.mean,
                                gauss5.mean,
                                gauss6.mean,
                                gauss7.mean,
                                gauss8.mean,
                                gauss9.mean,
                                gauss10.mean,
                                gauss11.mean,
                                gauss12.mean,
                                gauss13.mean,])
    elif version == 4:
        sigma_1 = 2.
        gauss1 = torch.distributions.MultivariateNormal(torch.tensor([0., -30.]), torch.tensor([[sigma_1**2, 0.], [0., sigma_1**2]]))
        sigma_2 = 1.75
        gauss2 = torch.distributions.MultivariateNormal(torch.tensor([15., -15.]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss3 = torch.distributions.MultivariateNormal(torch.tensor([0., -15.]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss4 = torch.distributions.MultivariateNormal(torch.tensor([-15., -15.]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        sigma_3 = 1.5
        gauss5 = torch.distributions.MultivariateNormal(torch.tensor([-30., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss6 = torch.distributions.MultivariateNormal(torch.tensor([-15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss7 = torch.distributions.MultivariateNormal(torch.tensor([0., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss8 = torch.distributions.MultivariateNormal(torch.tensor([15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss9 = torch.distributions.MultivariateNormal(torch.tensor([30., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        sigma_4 = 1.25
        gauss10 = torch.distributions.MultivariateNormal(torch.tensor([7.5, 15.]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss11 = torch.distributions.MultivariateNormal(torch.tensor([0., 15.]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss12 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, 15.]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        sigma_5 = 1.
        gauss13 = torch.distributions.MultivariateNormal(torch.tensor([0., 30.]), torch.tensor([[sigma_5**2, 0.], [0., sigma_5**2]]))

        covs_truth = torch.stack([gauss1.covariance_matrix,
                                gauss2.covariance_matrix,
                                gauss3.covariance_matrix,
                                gauss4.covariance_matrix,
                                gauss5.covariance_matrix,
                                gauss6.covariance_matrix,
                                gauss7.covariance_matrix,
                                gauss8.covariance_matrix,
                                gauss9.covariance_matrix,
                                gauss10.covariance_matrix,
                                gauss11.covariance_matrix,
                                gauss12.covariance_matrix,
                                gauss13.covariance_matrix])
        means_truth = torch.stack([gauss1.mean,
                                gauss2.mean,
                                gauss3.mean,
                                gauss4.mean,
                                gauss5.mean,
                                gauss6.mean,
                                gauss7.mean,
                                gauss8.mean,
                                gauss9.mean,
                                gauss10.mean,
                                gauss11.mean,
                                gauss12.mean,
                                gauss13.mean,])
    elif version == 5:
        sigma_1 = 2.
        gauss1 = torch.distributions.MultivariateNormal(torch.tensor([-30., -30.]), torch.tensor([[sigma_1**2, 0.], [0., sigma_1**2]]))
        gauss2 = torch.distributions.MultivariateNormal(torch.tensor([30., -30.]), torch.tensor([[sigma_1**2, 0.], [0., sigma_1**2]]))
        sigma_2 = 1.75
        gauss3 = torch.distributions.MultivariateNormal(torch.tensor([22.5, -15.]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        gauss4 = torch.distributions.MultivariateNormal(torch.tensor([-22.5, -15.]), torch.tensor([[sigma_2**2, 0.], [0., sigma_2**2]]))
        sigma_3 = 1.5
        gauss5 = torch.distributions.MultivariateNormal(torch.tensor([15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        gauss6 = torch.distributions.MultivariateNormal(torch.tensor([-15., 0.]), torch.tensor([[sigma_3**2, 0.], [0., sigma_3**2]]))
        sigma_4 = 1.25
        gauss7 = torch.distributions.MultivariateNormal(torch.tensor([-7.5, 15.]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        gauss8 = torch.distributions.MultivariateNormal(torch.tensor([7.5, 15.]), torch.tensor([[sigma_4**2, 0.], [0., sigma_4**2]]))
        sigma_5 = 1.
        gauss9 = torch.distributions.MultivariateNormal(torch.tensor([0., 30.]), torch.tensor([[sigma_5**2, 0.], [0., sigma_5**2]]))

        covs_truth = torch.stack([gauss1.covariance_matrix,
                                gauss2.covariance_matrix,
                                gauss3.covariance_matrix,
                                gauss4.covariance_matrix,
                                gauss5.covariance_matrix,
                                gauss6.covariance_matrix,
                                gauss7.covariance_matrix,
                                gauss8.covariance_matrix,
                                gauss9.covariance_matrix])
        means_truth = torch.stack([gauss1.mean,
                                gauss2.mean,
                                gauss3.mean,
                                gauss4.mean,
                                gauss5.mean,
                                gauss6.mean,
                                gauss7.mean,
                                gauss8.mean,
                                gauss9.mean])

    else:
        raise ValueError()

    comp_probs_truth = torch.ones(covs_truth.shape[0])/covs_truth.shape[0]

    if standardise:
        # Compute the mean of the MoG
        mean = (means_truth*comp_probs_truth[:, None]).sum(0)
        # Compute the covariance of the MoG
        dif = means_truth - mean
        cov = covs_truth + dif[..., None] @ dif[:, None, :]
        cov *= comp_probs_truth[:, None, None]
        cov = cov.sum(0)

        # Get standard deviation of marginals
        std = np.diagonal(cov)**(0.5)

        # Standardise the covariance matrices
        L = (np.linalg.cholesky(covs_truth)/std[None, :, None])
        covs_truth = L @ L.transpose((0, 2, 1))

        # Standardise the means
        means_truth = (means_truth-mean)/std

    comp_probs_truth = torch.tensor(comp_probs_truth)
    means_truth = torch.tensor(means_truth)
    covs_truth = torch.tensor(covs_truth)
    if convert_model_to_float:
        comp_probs_truth = comp_probs_truth.float()
        means_truth = means_truth.float()
        covs_truth = covs_truth.float()

    return comp_probs_truth, means_truth, covs_truth

def create_and_save_grid2dmog_dataset(num_samples, root_dir='./data', filename='data_grid2dmog.mat', convert_model_to_float=True, version=1):
    # Generate a Mixture-of-Gaussians distribution
    comp_probs, means, covs = create_grid2dmog_model(standardise=True, convert_model_to_float=convert_model_to_float, version=version)

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
    filename = os.path.join(root_dir, 'toy', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    return data

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

# def mog_log_prob_miss(X, M, *, comp_probs, means, covs=None, scale_trils=None):
#     X = X.unsqueeze(-2)
#     M = M.unsqueeze(-2)

#     mix = torch.distributions.Categorical(probs=comp_probs)
#     incomp_multi_norms = IncompleteDataMultivariateNormal(means, covariance_matrix=covs, scale_tril=scale_trils)

#     log_prob_x = incomp_multi_norms.log_prob_mis(X, M)
#     log_mix_prob = torch.log_softmax(mix.logits, dim=-1)
#     return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)


def create_and_save_dataset(num_samples, dims, num_components, root_dir='./data', filename='data_mog.mat', convert_model_to_float=True):
    # Generate a Mixture-of-Gaussians distribution
    comp_probs, means, covs = generate_mog_model(num_components, dims, standardise=True, convert_model_to_float=convert_model_to_float)

    # Generate samples
    data_train = sample_mog(int(num_samples*0.9), comp_probs, means, covs=covs).float()
    data_val = sample_mog(int(num_samples*0.1), comp_probs, means, covs=covs).float()
    data_test = sample_mog(num_samples, comp_probs, means, covs=covs).float()

    data = {
        "train": data_train.numpy(),
        "val": data_val.numpy(),
        "test": data_test.numpy(),
        "comp_probs": comp_probs.numpy(),
        "means": means.numpy(),
        "covs": covs.numpy()
    }
    filename = os.path.join(root_dir, 'toy', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    return data

def create_and_save_dataset_with_batched_sampling(num_samples, dims, num_components, root_dir='./data', filename='data_mog.mat',
                                                  convert_model_to_float=True, sampling_batch=100, filename_no_params=None):
    # Generate a Mixture-of-Gaussians distribution
    comp_probs, means, covs = generate_mog_model(num_components, dims, standardise=True, convert_model_to_float=convert_model_to_float)

    # Generate samples
    data_train = []
    num_train_samples = int(num_samples*0.9)
    for i in range(math.ceil(num_train_samples/sampling_batch)):
        batch_samples = min(sampling_batch, num_train_samples-sampling_batch*i)
        data_train.append(sample_mog(batch_samples, comp_probs, means, covs=covs).float())
    data_train = torch.concat(data_train)

    data_val = []
    num_val_samples = int(num_samples*0.1)
    for i in range(math.ceil(num_val_samples/sampling_batch)):
        batch_samples = min(sampling_batch, num_val_samples-sampling_batch*i)
        data_val.append(sample_mog(batch_samples, comp_probs, means, covs=covs).float())
    data_val = torch.concat(data_val)

    data_test = []
    num_test_samples = num_samples
    for i in range(math.ceil(num_test_samples/sampling_batch)):
        batch_samples = min(sampling_batch, num_test_samples-sampling_batch*i)
        data_test.append(sample_mog(batch_samples, comp_probs, means, covs=covs).float())
    data_test = torch.concat(data_test)

    data = {
        "train": data_train.numpy(),
        "val": data_val.numpy(),
        "test": data_test.numpy(),
        "comp_probs": comp_probs.numpy(),
        "means": means.numpy(),
        "covs": covs.numpy()
    }
    filename = os.path.join(root_dir, 'toy', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    if filename_no_params is not None:
        filename_no_params = os.path.join(root_dir, 'toy', filename_no_params)

        data_no_params = {
            "train": data_train.numpy(),
            "val": data_val.numpy(),
            "test": data_test.numpy(),
        }
        sio.savemat(file_name=filename_no_params, mdict=data_no_params)

    return data

def create_and_save_larger_dataset(num_samples_train, original_filename, root_dir='./data', filename='data_mog_large.mat'):
    # Generate a Mixture-of-Gaussians distribution
    data_file = sio.loadmat(os.path.join(root_dir, 'toy', original_filename))
    comp_probs = torch.tensor(data_file['comp_probs'])#.flatten()
    means = torch.tensor(data_file['means'])
    covs = torch.tensor(data_file['covs'])

    # Generate samples
    data_train = sample_mog(int(num_samples_train), comp_probs, means, covs=covs).float()

    data = {
        "train": data_train.numpy(),
        "val": data_file['val'],
        "test": data_file['test'],
        "comp_probs": comp_probs.numpy(),
        "means": means.numpy(),
        "covs": covs.numpy()
    }
    filename = os.path.join(root_dir, 'toy', filename)

    # Save and return samples
    sio.savemat(file_name=filename, mdict=data)

    return data

def plot_data_pairwise(data, alpha=0.7):
    dims = data.shape[-1]
    min_v, max_v = data.min(), data.max()

    fig = plt.figure(figsize=(9,9))
    grid_spec = mpl.gridspec.GridSpec(ncols=dims-1, nrows=dims-1, figure=fig)
    for i in range(0, dims):
        for j in range(i+1, dims):
            # Plot pairs of dimensions
            ax = fig.add_subplot(grid_spec[i, j-1])

            ax.scatter(data[:, j], data[:, i], alpha=alpha)

            # Set common limits
            ax.set_ylim(min_v, max_v)
            ax.set_xlim(min_v, max_v)

            if i == 0:
                ax.set_title(f'dim={i}', fontsize=12)
            if j == 1:
                ax.set_ylabel(f'dim={j}', fontsize=12)

        # Create dummy axes to add labels on the left-hand side
        if 0 < i < (dims-1):
            ax = fig.add_subplot(dims-1, dims-1, i*(dims-1)+1)
            ax.set_ylabel(f'dim={i}', fontsize=12)
            # Hide the dummy axes
            ax.xaxis.set_visible(False)
            plt.setp(ax.spines.values(), visible=False)
            ax.tick_params(left=False, labelleft=False)
            ax.patch.set_visible(False)

    grid_spec.tight_layout(fig)

    return fig


class ToyDataset(data.Dataset):
    """
    A dataset wrapper
    """

    def __init__(self, root: str, filename='data_mog2',
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
        filename = os.path.join(root, 'toy', f'{filename}.mat')

        # Load Toy dataset
        self.data_file = sio.loadmat(filename)
        self.data = self.data_file[split]
        if self.return_targets:
            self.targets = self.data_file[f'{split}_comp'].flatten()

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

class ToyCustomTestDataset(data.Dataset):
    """
    A dataset wrapper
    """

    def __init__(self, root: str,
                 split: str = 'test',
                 dataset_idx: int = 0,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            split:      data split, e.g. train, val, test
            rng:        random number generator
        """
        super().__init__()

        na = float('nan')
        if dataset_idx == 0:
            self.data = torch.tensor([[na,  -2. ],
                                      [na,  -1. ],
                                      [na,  -0.5],
                                      [na,   0. ],
                                      [na,   0.5],
                                      [na,   1. ],
                                      [na,   2. ],
                                      [-2.,  na ],
                                      [-1.,  na ],
                                      [-0.5, na ],
                                      [ 0.,  na ],
                                      [ 0.5, na ],
                                      [ 1.,  na ],
                                      [ 2.,  na ],
                                      ], dtype=torch.float32)
        else:
            raise ValueError()

    def __getitem__(self, index):
        """
        Args:
            index: index of sample
        Returns:
            image: dataset sample
        """
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    filename='data_mog_10d'
    dataset = ToyDataset(root='./data', filename=filename, split='train')
    X = dataset[:]

    print(X.shape)

    fig = plot_data_pairwise(X)
    # plt.show()

    plt.savefig(f'./data/toy/{filename}_pairwise_plots.pdf')

    # create_and_save_dataset_with_batched_sampling(num_samples=5000, dims=500, num_components=100, root_dir='./data',
    #                                               filename='data_mog_500d.mat', convert_model_to_float=False,
    #                                               sampling_batch=100, filename_no_params='data_mog_500d_no_params.mat')

    # create_and_save_larger_dataset(num_samples_train=100000, original_filename='data_mog_10d.mat',
    #                                root_dir='./data', filename='data_mog_10d_large.mat')

    # create_and_save_dataset(num_samples=10000, dims=10, num_components=20,
    #                         root_dir='./data', filename='data_mog_10d.mat',
    #                         convert_model_to_float=False)


    # filename='data_grid2dmog'
    # create_and_save_grid2dmog_dataset(num_samples=30000, filename=f'{filename}.mat', version=1)
    # dataset = ToyDataset(root='./data', filename=filename, split='train')
    # X = dataset[:]

    # print(X.shape)

    # fig = plot_data_pairwise(X, alpha=0.2)
    # plt.show()

    # filename='data_grid2dmog2'
    # create_and_save_grid2dmog_dataset(num_samples=30000, filename=f'{filename}.mat', version=2)
    # dataset = ToyDataset(root='./data', filename=filename, split='train')
    # X = dataset[:]

    # print(X.shape)

    # fig = plot_data_pairwise(X, alpha=0.2)
    # plt.show()

    # filename='data_grid2dmog3'
    # create_and_save_grid2dmog_dataset(num_samples=30000, filename=f'{filename}.mat', version=3)
    # dataset = ToyDataset(root='./data', filename=filename, split='train')
    # X = dataset[:]

    # print(X.shape)

    # fig = plot_data_pairwise(X, alpha=0.2)
    # plt.show()

    # filename='data_grid2dmog4'
    # create_and_save_grid2dmog_dataset(num_samples=30000, filename=f'{filename}.mat', version=4)
    # dataset = ToyDataset(root='./data', filename=filename, split='train')
    # X = dataset[:]

    # print(X.shape)

    # fig = plot_data_pairwise(X, alpha=0.2)
    # plt.show()

    # filename='data_grid2dmog5'
    # create_and_save_grid2dmog_dataset(num_samples=30000, filename=f'{filename}.mat', version=5)
    # dataset = ToyDataset(root='./data', filename=filename, split='train')
    # X = dataset[:]

    # print(X.shape)

    # fig = plot_data_pairwise(X, alpha=0.2)
    # plt.show()
