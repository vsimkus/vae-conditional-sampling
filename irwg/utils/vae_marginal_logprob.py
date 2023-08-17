import numpy as np
import torch
from einops import rearrange, reduce

from irwg.models.neural_nets import FullyConnectedNetwork, ResidualFCNetwork

def create_2D_grid(min_v, max_v, bins=100):
    """Creates the grid for integrating in 2D"""
    x0, x1 = np.mgrid[min_v:max_v:(max_v-min_v)/bins,
                    min_v:max_v:(max_v-min_v)/bins]
    pos = np.empty(x0.shape + (2,))
    pos[:, :, 0] = x0
    pos[:, :, 1] = x1
    return x0, x1, torch.tensor(pos).float()


def create_Z(dim, device=torch.device('cpu'), *, min, max, bins=1000):
    """Creates the "grid" for integrated in 1D or 2D"""
    steps = bins
    if dim == 1:
        Z = rearrange(torch.linspace(min, max, steps=steps), 'z -> z 1')
        Z = Z.to(device)
        # "Volume" of each integrand
        dz = (max-min)/steps
        grid = None
    elif dim == 2:
        z0, z1, Z = create_2D_grid(min, max, bins=steps)
        Z = rearrange(Z, 'z1 z2 d -> (z1 z2) d')
        Z = Z.to(device)
        # "Volume" of each integrand
        dz = ((max-min)/steps)**2
        grid = (z0, z1)
    else:
        raise NotImplementedError('Function not implemented for dimensionality greater than 2.')
    return Z, dz, grid

# def logTrapezoid2DExp(log_X, dx):
#     """Performs log-trapezoid-exp in a numerically safe way for 2D grids"""
#     # See e.g. https://math.stackexchange.com/questions/2891298/derivation-of-2d-trapezoid-rule
#     # Apply the 1D trapz two times.
#     steps = 500
#     log_X = rearrange(log_X, '(z1 z2) d -> z1 z2 d', z1=steps, z2=steps)
#     log_X[0, ...] -= torch.log(torch.tensor(2))
#     log_X[-1, ...] -= torch.log(torch.tensor(2))
#     log_X = torch.logsumexp(log_X, dim=0) - torch.log(torch.tensor(dx))
#     log_X[0, ...] -= torch.log(torch.tensor(2))
#     log_X[-1, ...] -= torch.log(torch.tensor(2))
#     log_X = torch.logsumexp(log_X, dim=0) - torch.log(torch.tensor(dx))
#     return log_X

def get_marginal_logprob(self, batch, *, latent_min, latent_max, latent_bins, compute_complete=False, marginal_eval_batchsize=-1):
    """
    Estimates marginal log-probability using the Riemann sum.
    """
    X, M = batch[:2]

    latent_dim = None
    if isinstance(self.generator_network, FullyConnectedNetwork):
        latent_dim = self.generator_network.layer_dims[0]
    elif isinstance(self.generator_network, ResidualFCNetwork):
        latent_dim = self.generator_network.input_dim

    assert latent_dim <= 2,\
        'Cannot numerically integrate for dims > 2!'

    # Create the grid
    Z, dz, _ = create_Z(latent_dim, device=X.device, min=latent_min, max=latent_max, bins=latent_bins)

    # Compute prior logprob
    prior_dist = self.get_prior()
    prior_logprob = prior_dist.log_prob(Z)
    prior_logprob = reduce(prior_logprob, 'z d -> z 1', 'sum')

    # Compute the parameters of the generator
    generator_params = self.generator_network(Z)
    generator_params = rearrange(generator_params, 'z pd -> z 1 pd')

    # Compute the conditional log-likelihood of each data-point
    generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)

    if marginal_eval_batchsize == -1:
        # Eval all in one go
        comp_cond_logprob = generator_distr.log_prob(X)
        cond_logprob = comp_cond_logprob*M
        cond_logprob = reduce(cond_logprob, 'z b d -> z b', 'sum')

        # Compute the marginal log_probability
        marginal_logprob = torch.logsumexp(prior_logprob + cond_logprob + torch.log(torch.tensor(dz)), dim=0)
        # marginal_logprob = logTrapezoid2DExp(prior_logprob + cond_logprob, dx=dz**0.5)

        if compute_complete:
            # Compute marginal log_probability on complete data too
            comp_cond_logprob = reduce(comp_cond_logprob, 'z b d -> z b', 'sum')
            complete_marginal_logprob = torch.logsumexp(prior_logprob + comp_cond_logprob + torch.log(torch.tensor(dz)), dim=0)
            return marginal_logprob, complete_marginal_logprob
        else:
            return marginal_logprob
    else:
        # Eval in batches
        marginal_logprob = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
        if compute_complete:
            complete_marginal_logprob = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)

        indices = rearrange(torch.arange(X.shape[0] - X.shape[0] % marginal_eval_batchsize), '(b k) -> b k', k=marginal_eval_batchsize)
        for i in range(len(indices)):
            idx = indices[i]
            Xi, Mi = X[idx], M[idx]

            comp_cond_logprob_i = generator_distr.log_prob(Xi)
            cond_logprob_i = comp_cond_logprob_i*Mi
            cond_logprob_i = reduce(cond_logprob_i, 'z b d -> z b', 'sum')

            # Compute the marginal log_probability
            marginal_logprob[idx] = torch.logsumexp(prior_logprob + cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

            if compute_complete:
                # Compute marginal log_probability on complete data too
                comp_cond_logprob_i = reduce(comp_cond_logprob_i, 'z b d -> z b', 'sum')
                complete_marginal_logprob[idx] = torch.logsumexp(prior_logprob + comp_cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

        # Eval for the rest of datapoints
        if X.shape[0] % marginal_eval_batchsize != 0:
            idx = torch.arange(X.shape[0] - X.shape[0] % marginal_eval_batchsize, X.shape[0])
            Xi, Mi = X[idx], M[idx]

            comp_cond_logprob_i = generator_distr.log_prob(Xi)
            cond_logprob_i = comp_cond_logprob_i*Mi
            cond_logprob_i = reduce(cond_logprob_i, 'z b d -> z b', 'sum')

            # Compute the marginal log_probability
            marginal_logprob[idx] = torch.logsumexp(prior_logprob + cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

            if compute_complete:
                # Compute marginal log_probability on complete data too
                comp_cond_logprob_i = reduce(comp_cond_logprob_i, 'z b d -> z b', 'sum')
                complete_marginal_logprob[idx] = torch.logsumexp(prior_logprob + comp_cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

        if compute_complete:
            return marginal_logprob, complete_marginal_logprob
        else:
            return marginal_logprob

def get_latent_post_mean(self, batch, *, latent_min, latent_max, latent_bins, compute_complete=False, marginal_eval_batchsize=-1):
    X, M = batch[:2]

    latent_dim = None
    if isinstance(self.generator_network, FullyConnectedNetwork):
        latent_dim = self.generator_network.layer_dims[0]
    elif isinstance(self.generator_network, ResidualFCNetwork):
        latent_dim = self.generator_network.input_dim

    assert latent_dim <= 2,\
        'Cannot numerically integrate for dims > 2!'

    # Create the grid
    Z, dz, _ = create_Z(latent_dim, device=X.device, min=latent_min, max=latent_max, bins=latent_bins)

    # Compute prior logprob
    prior_dist = self.get_prior()
    prior_logprob = prior_dist.log_prob(Z)
    prior_logprob = reduce(prior_logprob, 'z d -> z 1', 'sum')

    # Compute the parameters of the generator
    generator_params = self.generator_network(Z)
    generator_params = rearrange(generator_params, 'z pd -> z 1 pd')

    # Compute the conditional log-likelihood of each data-point
    generator_distr = self.create_distribution(generator_params, self.hparams.generator_distribution)

    if marginal_eval_batchsize == -1:
        # Eval all in one go
        comp_cond_logprob = generator_distr.log_prob(X)
        cond_logprob = comp_cond_logprob*M
        cond_logprob = reduce(cond_logprob, 'z b d -> z b', 'sum')

        # Compute the marginal log_probability
        marginal_logprob = torch.logsumexp(prior_logprob + cond_logprob + torch.log(torch.tensor(dz)), dim=0)
        # marginal_logprob = logTrapezoid2DExp(prior_logprob + cond_logprob, dx=dz**0.5)

        cond_logprob = prior_logprob + cond_logprob - marginal_logprob.unsqueeze(0)
        cond_mean = torch.sum(torch.exp(cond_logprob).unsqueeze(-1) * Z.unsqueeze(1) * torch.tensor(dz), dim=0)

        if compute_complete:
            # Compute marginal log_probability on complete data too
            comp_cond_logprob = reduce(comp_cond_logprob, 'z b d -> z b', 'sum')
            complete_marginal_logprob = torch.logsumexp(prior_logprob + comp_cond_logprob + torch.log(torch.tensor(dz)), dim=0)

            complete_cond_logprob = prior_logprob + comp_cond_logprob - complete_marginal_logprob.unsqueeze(0)
            complete_cond_mean = torch.sum(torch.exp(complete_cond_logprob).unsqueeze(-1) * Z.unsqueeze(1) * torch.tensor(dz), dim=0)

            return cond_mean, cond_logprob, Z, complete_cond_mean, complete_cond_logprob
        else:
            return cond_mean, cond_logprob, Z
    else:
        # Eval in batches
        # marginal_logprob = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
        cond_mean = torch.empty(X.shape[0], latent_dim, dtype=X.dtype, device=X.device)
        cond_logprob = torch.empty(Z.shape[0], X.shape[0], latent_dim, dtype=X.dtype, device=X.device)
        if compute_complete:
            # complete_marginal_logprob = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
            complete_cond_mean = torch.empty(X.shape[0], latent_dim, dtype=X.dtype, device=X.device)
            complete_cond_logprob = torch.empty(Z.shape[0], X.shape[0], latent_dim, dtype=X.dtype, device=X.device)

        indices = rearrange(torch.arange(X.shape[0] - X.shape[0] % marginal_eval_batchsize), '(b k) -> b k', k=marginal_eval_batchsize)
        for i in range(len(indices)):
            idx = indices[i]
            Xi, Mi = X[idx], M[idx]

            comp_cond_logprob_i = generator_distr.log_prob(Xi)
            cond_logprob_i = comp_cond_logprob_i*Mi
            cond_logprob_i = reduce(cond_logprob_i, 'z b d -> z b', 'sum')

            # Compute the marginal log_probability
            marginal_logprob = torch.logsumexp(prior_logprob + cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

            cond_logprob_i = prior_logprob + cond_logprob_i - marginal_logprob.unsqueeze(0)
            cond_logprob[:, idx] = cond_logprob_i
            cond_mean[idx] = torch.sum(torch.exp(cond_logprob_i).unsqueeze(-1) * Z.unsqueeze(1) * torch.tensor(dz), dim=0)

            if compute_complete:
                # Compute marginal log_probability on complete data too
                comp_cond_logprob_i = reduce(comp_cond_logprob_i, 'z b d -> z b', 'sum')
                complete_marginal_logprob = torch.logsumexp(prior_logprob + comp_cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

                complete_cond_logprob_i = prior_logprob + comp_cond_logprob_i - complete_marginal_logprob.unsqueeze(0)
                complete_cond_logprob_i[:, idx] = complete_cond_logprob_i
                complete_cond_mean[idx] = torch.sum(torch.exp(complete_cond_logprob_i).unsqueeze(-1) * Z.unsqueeze(1) * torch.tensor(dz), dim=0)

        # Eval for the rest of datapoints
        if X.shape[0] % marginal_eval_batchsize != 0:
            idx = torch.arange(X.shape[0] - X.shape[0] % marginal_eval_batchsize, X.shape[0])
            Xi, Mi = X[idx], M[idx]

            comp_cond_logprob_i = generator_distr.log_prob(Xi)
            cond_logprob_i = comp_cond_logprob_i*Mi
            cond_logprob_i = reduce(cond_logprob_i, 'z b d -> z b', 'sum')

            # Compute the marginal log_probability
            marginal_logprob = torch.logsumexp(prior_logprob + cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

            cond_logprob_i = prior_logprob + cond_logprob_i - marginal_logprob.unsqueeze(0)
            cond_logprob[:, idx] = cond_logprob_i
            cond_mean[idx] = torch.sum(torch.exp(cond_logprob_i).unsqueeze(-1) * Z.unsqueeze(1) * torch.tensor(dz), dim=0)

            if compute_complete:
                # Compute marginal log_probability on complete data too
                comp_cond_logprob_i = reduce(comp_cond_logprob_i, 'z b d -> z b', 'sum')
                complete_marginal_logprob = torch.logsumexp(prior_logprob + comp_cond_logprob_i + torch.log(torch.tensor(dz)), dim=0)

                complete_cond_logprob_i = prior_logprob + comp_cond_logprob_i - complete_marginal_logprob.unsqueeze(0)
                complete_cond_logprob_i[:, idx] = complete_cond_logprob_i
                complete_cond_mean[idx] = torch.sum(torch.exp(complete_cond_logprob_i).unsqueeze(-1) * Z.unsqueeze(1) * torch.tensor(dz), dim=0)

        if compute_complete:
            return cond_mean, cond_logprob, Z, complete_cond_mean, complete_cond_logprob
        else:
            return cond_mean, cond_logprob, Z
