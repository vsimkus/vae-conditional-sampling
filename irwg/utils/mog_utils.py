import numpy as np
import torch
from einops import rearrange, repeat, asnumpy
from sklearn.covariance import OAS as OAS_Covariance
from sklearn.covariance import GraphicalLasso as GraphicalLassoCovariance
from sklearn.covariance import LedoitWolf as LedoitWolfCovariance
from tqdm import tqdm
import math


def compute_kldivs_between_categoricals(log_p, log_q):
    diff = log_p - log_q

    p = torch.exp(log_p)
    p = np.round(p, decimals=8) # To avoid some numerical instabilities
    kl_fow = torch.sum(p * diff, dim=-1)

    q = torch.exp(log_q)
    q = np.round(q, decimals=8) # To avoid some numerical instabilities
    kl_rev = torch.sum(q * -diff, dim=-1)

    # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
    log_pm = torch.logsumexp(rearrange([log_p, log_q], 'p ... -> p ...'), dim=0) - np.log(2)

    jsd = 0.5*(torch.sum(p * (log_p - log_pm), dim=-1) + torch.sum(q * (log_q - log_pm), dim=-1))

    return kl_fow, kl_rev, jsd

def compute_joint_p_c_given_xo_xm(X, comp_log_probs, means, covs):
    """
    Computes p(c | xo, xm)
    pi_c_|xo,xm = \frac{pi_c N(xo, xm | mu_c, sigma_c)}{\sum_k pi_k N(xo, xm | mu_k, sigma_k)}
    """
    X = X.unsqueeze(1)

    multi_norms = torch.distributions.MultivariateNormal(
        loc=means, covariance_matrix=covs)

    log_prob_x_given_c = multi_norms.log_prob(X)
    log_prob_xc = log_prob_x_given_c + comp_log_probs

    comp_log_probs_given_x = log_prob_xc - torch.logsumexp(log_prob_xc, dim=-1, keepdim=True)

    return comp_log_probs_given_x

def compute_conditional_mog_parameters(X, M, comp_log_probs, means, covs):
    """
    Computes the parameters of the conditional mixture of Gaussians
    p(x) = \sum_c pi_c N(x | mu_c, sigma_c)
    p(xo | xm)  = \frac{\sum_c pi_c N(xo, xm | mu_c, sigma_c)}{\sum_k pi_k N(xo | k)}
                = \sum_c ( \frac{pi_c N(xo | mu_c, sigma_c)}{\sum_k pi_k N(xo | mu_k, sigma_k)} ) N(xm | mu_c, sigma_c) )
                = \sum_c pi_c_|xo N(xm | mu_c, sigma_c)
    where pi_c_|xo = \frac{pi_c N(xo | mu_c, sigma_c)}{\sum_k pi_k N(xo | mu_k, sigma_k)}
    """
    X = X.unsqueeze(1)
    M = M.unsqueeze(1)
    M_not = ~M

    covs = covs.unsqueeze(0)
    means = means.unsqueeze(0)

    sigma_mm = covs * M_not.unsqueeze(-1) * M_not.unsqueeze(-2)
    sigma_mo = covs * M_not.unsqueeze(-1) * M.unsqueeze(-2)
    sigma_oo = covs * M.unsqueeze(-1) * M.unsqueeze(-2)

    M_eye = torch.eye(X.shape[-1]).unsqueeze(0).unsqueeze(0) * M_not.unsqueeze(-1)
    sigma_oo_with_ones_in_missing_diag_entries = sigma_oo + M_eye

    # sigma_oo_inv = torch.inverse(sigma_oo_with_ones_in_missing_diag_entries) - M_eye
    # sigma_mo_oo_inv = sigma_mo @ sigma_oo_inv

    xo_minus_mean = (X - means)*M
    sigma_oo_inv_mult_xo_minus_mean = torch.linalg.solve(sigma_oo_with_ones_in_missing_diag_entries, xo_minus_mean)


    # Compute N(xm | xo; k) for each k and each xo

    # means_m_given_o = means*M_not + (sigma_mo_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze(-1)
    means_m_given_o = means*M_not + (sigma_mo @ sigma_oo_inv_mult_xo_minus_mean.unsqueeze(-1)).squeeze(-1)
    # covs_m_given_o = sigma_mm - sigma_mo_oo_inv @ sigma_mo.transpose(-1, -2)
    covs_m_given_o = sigma_mm - sigma_mo @ torch.linalg.solve(sigma_oo_with_ones_in_missing_diag_entries, sigma_mo.transpose(-1, -2))

    # Compute the component coefficients pi_k given obs

    D = M.sum(-1)
    # cond_log_probs_oo = 0.5*(-D * torch.log(torch.tensor(2*np.pi))
    #             -torch.logdet(sigma_oo_with_ones_in_missing_diag_entries)
    #             -(xo_minus_mean.unsqueeze(-2) @ sigma_oo_inv @ xo_minus_mean.unsqueeze(-1)).squeeze()
    # )
    cond_log_probs_oo = 0.5*(-D * torch.log(torch.tensor(2*np.pi))
                -torch.logdet(sigma_oo_with_ones_in_missing_diag_entries)
                -(xo_minus_mean.unsqueeze(-2) @ sigma_oo_inv_mult_xo_minus_mean.unsqueeze(-1)).squeeze()
    )

    joint_log_probs_oo = comp_log_probs + cond_log_probs_oo
    comp_log_probs_given_o = joint_log_probs_oo - torch.logsumexp(joint_log_probs_oo, dim=-1, keepdim=True)

    return comp_log_probs_given_o, means_m_given_o, covs_m_given_o

def compute_conditional_mog_parameters_nonbatched(x, m, comp_log_probs, means, covs):
    """
    Computes the parameters of the conditional mixture of Gaussians
    p(x) = \sum_c pi_c N(x | mu_c, sigma_c)
    p(xo | xm)  = \frac{\sum_c pi_c N(xo, xm | mu_c, sigma_c)}{\sum_k pi_k N(xo | k)}
                = \sum_c ( \frac{pi_c N(xo | mu_c, sigma_c)}{\sum_k pi_k N(xo | mu_k, sigma_k)} ) N(xm | mu_c, sigma_c) )
                = \sum_c pi_c_|xo N(xm | mu_c, sigma_c)
    where pi_c_|xo = \frac{pi_c N(xo | mu_c, sigma_c)}{\sum_k pi_k N(xo | mu_k, sigma_k)}

    Use it to verify the above batched version.
    """
    comp_log_probs = comp_log_probs.squeeze(0)

    m = m.bool()
    xo = x[m]
    means_xo = means[:, m]
    covs_xo = covs[:, m, :][:, :, m]

    log_probs_xo = []
    for i in range(len(comp_log_probs)):
        mean_i = means_xo[i]
        cov_i = covs_xo[i]

        norm = torch.distributions.MultivariateNormal(mean_i, cov_i)
        log_prob_xo = norm.log_prob(xo)
        log_probs_xo.append(log_prob_xo)
    log_probs_xo = torch.stack(log_probs_xo)

    assert len(comp_log_probs) == len(log_probs_xo)

    log_pi_c_given_xo = comp_log_probs + log_probs_xo
    log_pi_c_given_xo = log_pi_c_given_xo - torch.logsumexp(log_pi_c_given_xo, dim=0, keepdim=True)

    total_prob = torch.exp(log_pi_c_given_xo).sum()
    assert np.allclose(total_prob, 1)

    m_not = ~m
    means_xm = means[:, m_not]
    covs_xm = covs[:, m_not, :][:, :, m_not]
    covs_c = covs[:, m_not, :][:, :, m]

    means_xm_given_xo = []
    covs_xm_given_xo = []
    for i in range(len(comp_log_probs)):
        mean_i_xm = means_xm[i]
        cov_i_xm = covs_xm[i]
        cov_i_xo = covs_xo[i]
        mean_i_xo = means_xo[i]
        cov_i_c = covs_c[i]

        mean_i_xm_given_xo = mean_i_xm + (cov_i_c @ torch.linalg.solve(cov_i_xo, (xo - mean_i_xo).unsqueeze(-1))).squeeze(-1)
        means_xm_given_xo.append(mean_i_xm_given_xo)

        cov_i_xm_given_xo = cov_i_xm - (cov_i_c @ torch.linalg.solve(cov_i_xo, cov_i_c.transpose(-1, -2)))
        covs_xm_given_xo.append(cov_i_xm_given_xo)
    means_xm_given_xo = torch.stack(means_xm_given_xo)
    covs_xm_given_xo = torch.stack(covs_xm_given_xo)

    means_new = torch.zeros_like(means)
    means_new[:, m_not] = means_xm_given_xo
    covs_new = torch.zeros_like(covs)
    for j, i in enumerate(torch.where(m_not)[0]):
        temp = torch.zeros_like(covs_new[:, i, :])
        temp[:, m_not] = covs_xm_given_xo[:, j]
        covs_new[:, i, :] = temp

    return log_pi_c_given_xo[None, :], means_new, covs_new


def batched_update_mogs_parameters(X, M, comp_log_probs, means, covs, use_solver=False):
    """
    Fits multiple sparse MoGs using an iteration of EM
    """
    # Compute log p(x, z) for all z
    joint_log_probs = batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs, means, covs, use_solver=use_solver)

    cond_log_probs = joint_log_probs - torch.logsumexp(joint_log_probs, dim=-1, keepdim=True)

    log_responsibilities = torch.logsumexp(cond_log_probs, dim=1)

    # membership = cond_probs / responsibilities
    log_membership = cond_log_probs - log_responsibilities.unsqueeze(1)
    membership = torch.exp(log_membership)

    # Compute updates (using log-domain for numerical stability)

    # component log-probs
    comp_log_probs_up = torch.logsumexp(cond_log_probs, dim=1) - torch.log(torch.tensor(cond_log_probs.shape[1]))

    # means
    means_up = (X.unsqueeze(-2) * membership.unsqueeze(-1)).sum(1)

    # covariances
    dif = (X.unsqueeze(-2) - means_up.unsqueeze(1))
    covs_up = (((dif.unsqueeze(-1) @ dif.unsqueeze(-2)))*membership.unsqueeze(-1).unsqueeze(-1)).sum(1)
    # Ensure positive definiteness by adding a small value to the diagonal
    covs_up += torch.eye(covs.shape[-1]).unsqueeze(0)*1e-6#1e-4#*1e-5

    return comp_log_probs_up, means_up, covs_up

def update_mog_parameters_nonbatched(X, M, comp_log_probs, means, covs):
    # Based on Murphy's book Chapter 8.7.3.2
    comp_log_probs = comp_log_probs.squeeze(0)
    means = means.squeeze(0)
    covs = covs.squeeze(0)

    all_joint_log_probs = []
    for i in range(X.shape[0]):
        x = X[i]
        m = M
        joint_log_probs = compute_joint_log_probs_sparse_mogs_nonbatched(x, m, comp_log_probs, means, covs)
        all_joint_log_probs.append(joint_log_probs)
    all_joint_log_probs = torch.stack(all_joint_log_probs)

    # all_joint_log_probs2 = compute_joint_log_probs_sparse_mogs_batched_for_single_miss_pattern(X, M, comp_log_probs, means, covs)
    # assert torch.allclose(all_joint_log_probs2, all_joint_log_probs)

    # Responsibility
    all_joint_log_probs = all_joint_log_probs - torch.logsumexp(all_joint_log_probs, dim=-1, keepdim=True)
    # Membership
    log_membership = all_joint_log_probs - torch.logsumexp(all_joint_log_probs, dim=0, keepdim=True)

    # Mean update
    # means_up = (torch.exp(all_joint_log_probs).unsqueeze(-1)*X.unsqueeze(-2)).sum(0) / torch.exp(all_joint_log_probs).sum(0).unsqueeze(-1)
    means_up = (torch.exp(log_membership).unsqueeze(-1)*X.unsqueeze(-2)).sum(0)

    # Component log-prob update
    # comp_log_probs_up = torch.log(torch.exp(all_joint_log_probs).mean(0))
    comp_log_probs_up = torch.logsumexp(all_joint_log_probs, dim=0) - torch.log(torch.tensor(all_joint_log_probs.shape[0]))

    # Covariance update
    diff = (X.unsqueeze(-2) - means_up.unsqueeze(0)).unsqueeze(-1)
    # covs_up = (torch.exp(all_joint_log_probs).unsqueeze(-1).unsqueeze(-1)*(diff @ diff.transpose(-1, -2))).sum(0)
    # covs_up = covs_up / torch.exp(all_joint_log_probs).sum(0).unsqueeze(-1).unsqueeze(-1)
    covs_up = (torch.exp(log_membership).unsqueeze(-1).unsqueeze(-1)*(diff @ diff.transpose(-1, -2))).sum(0)
    covs_up += torch.eye(covs.shape[-1]).unsqueeze(0)*1e-6#1e-4#*1e-5

    return comp_log_probs_up.unsqueeze(0), means_up.unsqueeze(0), covs_up.unsqueeze(0)

def update_mog_parameters_batched_per_datapoint(X, M, comp_log_probs, means, covs):
    # Based on Murphy's book Chapter 8.7.3.2
    comp_log_probs = comp_log_probs.squeeze(0)
    means = means.squeeze(0)
    covs = covs.squeeze(0)

    all_joint_log_probs = compute_joint_log_probs_sparse_mogs_batched_for_single_miss_pattern(X, M, comp_log_probs, means, covs)

    # Responsibility
    all_joint_log_probs = all_joint_log_probs - torch.logsumexp(all_joint_log_probs, dim=-1, keepdim=True)
    # Membership
    log_membership = all_joint_log_probs - torch.logsumexp(all_joint_log_probs, dim=0, keepdim=True)

    # Mean update
    # means_up = (torch.exp(all_joint_log_probs).unsqueeze(-1)*X.unsqueeze(-2)).sum(0) / torch.exp(all_joint_log_probs).sum(0).unsqueeze(-1)
    means_up = (torch.exp(log_membership).unsqueeze(-1)*X.unsqueeze(-2)).sum(0)

    # Component log-prob update
    # comp_log_probs_up = torch.log(torch.exp(all_joint_log_probs).mean(0))
    comp_log_probs_up = torch.logsumexp(all_joint_log_probs, dim=0) - torch.log(torch.tensor(all_joint_log_probs.shape[0]))

    # Covariance update
    diff = (X.unsqueeze(-2) - means_up.unsqueeze(0)).unsqueeze(-1)
    # covs_up = (torch.exp(all_joint_log_probs).unsqueeze(-1).unsqueeze(-1)*(diff @ diff.transpose(-1, -2))).sum(0)
    # covs_up = covs_up / torch.exp(all_joint_log_probs).sum(0).unsqueeze(-1).unsqueeze(-1)
    covs_up = (torch.exp(log_membership).unsqueeze(-1).unsqueeze(-1)*(diff @ diff.transpose(-1, -2))).sum(0)
    covs_up += torch.eye(covs.shape[-1]).unsqueeze(0)*1e-6#1e-4#*1e-5

    return comp_log_probs_up.unsqueeze(0), means_up.unsqueeze(0), covs_up.unsqueeze(0)

def batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs, means, covs, use_solver=False):
    """
    Computes the log probability of the data xo given the mogs parameters
    p(xm, c) = pi_c p(xm|c)
    """
    M = M.unsqueeze(1)
    M_not = ~M

    sigma_mm = covs * M_not.unsqueeze(-1) * M_not.unsqueeze(-2)

    M_not_eye = torch.eye(X.shape[-1]).unsqueeze(0).unsqueeze(0) * M.unsqueeze(-1)
    sigma_mm_with_ones_in_missing_diag_entries = sigma_mm + M_not_eye
    if not use_solver:
        sigma_mm_inv = torch.inverse(sigma_mm_with_ones_in_missing_diag_entries)# - M_not_eye
        # NOTE: Cholesky inverse should be more stable for SPD matrices than the standard inverse (above)
        # sigma_mm_inv = torch.cholesky_inverse(torch.linalg.cholesky(sigma_mm_with_ones_in_missing_diag_entries))
    # NOTE: standard pytorch cholesky requires SPD matrices (above), instead we use our own implementation that accepts semi-definite matrices too
    # sigma_mm_inv = torch.cholesky_inverse(semi_definite_symmetric_cholesky(sigma_mm_with_ones_in_missing_diag_entries))
    # L = semi_definite_symmetric_cholesky(sigma_mm_with_ones_in_missing_diag_entries)
    # L_inv = torch.inverse(L)
    # sigma_mm_inv = L_inv @ L_inv.transpose(-1, -2)

    if not use_solver:
        sigma_mm_inv = sigma_mm_inv.unsqueeze(1)
    sigma_mm_with_ones_in_missing_diag_entries = sigma_mm_with_ones_in_missing_diag_entries.unsqueeze(1)

    xm_minus_mean = ((X.unsqueeze(-2) - means.unsqueeze(1))*M_not.unsqueeze(1))

    # NOTE: torch.linalg.solve should be numerically more stable _and_ faster,
    # *BUT* it is slow in this case because it broadcasts the sigma to the large dimensions of xm
    # Hence, it is faster to invert the matrix first (as above)
    if use_solver:
        sigma_mm_inv_mult_xm_minus_mean = torch.linalg.solve(sigma_mm_with_ones_in_missing_diag_entries, xm_minus_mean.unsqueeze(-1))

    D = M_not.sum(-1)
    if not use_solver:
        cond_log_probs_mm = 0.5*(-D.unsqueeze(1) * torch.log(torch.tensor(2*np.pi))
                                 -torch.logdet(sigma_mm_with_ones_in_missing_diag_entries)
                                 -(xm_minus_mean.unsqueeze(-2) @ (sigma_mm_inv @ xm_minus_mean.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        )
    else:
        cond_log_probs_mm = 0.5*(-D.unsqueeze(1) * torch.log(torch.tensor(2*np.pi))
                                -torch.logdet(sigma_mm_with_ones_in_missing_diag_entries)
                                -(xm_minus_mean.unsqueeze(-2) @ sigma_mm_inv_mult_xm_minus_mean).squeeze(-1)
        )

    joint_log_probs = comp_log_probs.unsqueeze(1) + cond_log_probs_mm

    return joint_log_probs

def compute_joint_log_probs_sparse_mogs_nonbatched(x, m, comp_log_probs, means, covs):
    comp_log_probs = comp_log_probs.squeeze(0)

    m = m.bool()
    m_not = ~m
    xm = x[m_not]
    means = means[:, m_not]
    covs = covs[:, m_not, :][:, :, m_not]

    log_probs = []
    for i in range(len(comp_log_probs)):
        norm = torch.distributions.MultivariateNormal(means[i], covs[i])
        log_probs.append(norm.log_prob(xm))
    log_probs = torch.stack(log_probs)

    log_probs = log_probs + comp_log_probs

    return log_probs

def compute_joint_log_probs_sparse_mogs_batched_for_single_miss_pattern(x, m, comp_log_probs, means, covs, *, validate_args=False):
    comp_log_probs = comp_log_probs.squeeze(0)

    m = m.bool()
    m_not = ~m
    xm = x[:, m_not]
    means = means[:, m_not]
    covs = covs[:, m_not, :][:, :, m_not]

    # NOTE: in pytorch Cholesky decomp is only implemented for positive definite matrices
    L, info = torch.linalg.cholesky_ex(covs)
    errors = info > 0
    if errors.sum() > 0:
        print('WARNING: Adding small value to diagonal to avoid Cholesky decomposition errors')
        L[errors] += torch.eye(covs.shape[-1])*1e-1
    # Instead we can implement Cholesky decomposition for positive semi-definite matrices using eigen decomposition
    # L = semi_definite_symmetric_cholesky(covs)

    norm = torch.distributions.MultivariateNormal(means, scale_tril=L, validate_args=validate_args)
    log_probs = norm.log_prob(xm.unsqueeze(1))

    log_probs = log_probs + comp_log_probs.unsqueeze(0)

    return log_probs

def compute_joint_log_probs_sparse_mogs_diag_batched_for_single_miss_pattern(x, m, comp_log_probs, means, vars, *, validate_args=False):
    comp_log_probs = comp_log_probs.squeeze(0)

    m = m.bool()
    m_not = ~m
    xm = x[:, m_not]
    means = means[:, m_not]
    vars = vars[:, m_not]

    stds = vars**0.5

    norm = torch.distributions.Normal(means, scale=stds, validate_args=validate_args)
    log_probs = norm.log_prob(xm.unsqueeze(1))
    log_probs = log_probs.sum(dim=-1)

    log_probs = log_probs + comp_log_probs.unsqueeze(0)

    return log_probs

def batched_fit_mogs_sparse(X, M, num_components, num_iterations, *, use_solver=False,
                            dont_reset_on_failure=False,
                            init_params_to_true_joint=False,
                            init_params_to_true_cond=False,
                            true_comp_log_probs=None,
                            true_means=None,
                            true_covs=None,
                            true_cond_comp_log_probs=None,
                            true_cond_means=None,
                            true_cond_covs=None,
                            ):
    assert not (init_params_to_true_joint and init_params_to_true_cond)
    dim = X.shape[-1]
    stdv = 1.0 / M.sum(-1)
    stdv[stdv == float('inf')] = 1.

    # Initialise params
    if init_params_to_true_joint:
        print('Initialising MoGs to true joint values')
        means = repeat(true_means, 'c d -> b c d', b=X.shape[0])
        covs = repeat(true_covs, 'c d1 d2 -> b c d1 d2', b=X.shape[0])
        comp_log_probs = torch.zeros(X.shape[0], num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)
    elif init_params_to_true_cond:
        print('Initialising MoGs to true conditional params')
        means = true_cond_means.clone()
        covs = true_cond_covs.clone()
        comp_log_probs = torch.zeros(X.shape[0], num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)
    else:
        means = torch.rand(X.shape[0], num_components, dim)*6 - 3
        # covs = torch.rand(X.shape[0], num_components, dim, dim)*stdv.reshape(-1, 1, 1, 1) - stdv.reshape(-1, 1, 1, 1)
        # covs = covs.transpose(-2, -1) @ covs
        covs = torch.randn(X.shape[0], num_components, dim, dim)*stdv.reshape(-1, 1, 1, 1)
        covs = covs @ covs.transpose(-2, -1)
        comp_log_probs = torch.zeros(X.shape[0], num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)

    for t in tqdm(range(num_iterations), desc='Fitting MoGs'):
        comp_log_probs, means, covs = batched_update_mogs_parameters(X, M, comp_log_probs, means, covs, use_solver=use_solver)

        if not dont_reset_on_failure:
            # Check for failures
            # NOTE: here I'm using magnitude of mean values as an indication of divergence, I do not indend to have such large values in the ground truth.
            idx = torch.where(~(comp_log_probs.sum(-1)).isfinite() | ~(means.sum(-1).sum(-1)).isfinite() | ~(covs.sum(-1).sum(-1).sum(-1)).isfinite()
                            | ((means > 30) | (means < -30)).any(-1).any(-1) | ((covs > 25) | (covs < -25)).any(-1).any(-1).any(-1))
            if len(idx[0]) > 0:
                print(f'Resetting some MoGs at timestep {t}:', idx[0])

                if init_params_to_true_joint:
                    n_reset = idx[0].shape[0]
                    means[idx] = repeat(true_means, 'c d -> b c d', b=n_reset)
                    covs[idx] = repeat(true_covs, 'c d1 d2 -> b c d1 d2', b=n_reset)
                    k = torch.zeros(idx[0].shape[0], num_components)
                    k -= torch.logsumexp(k, dim=-1, keepdim=True)
                    comp_log_probs[idx] = k
                elif init_params_to_true_cond:
                    means[idx] = true_cond_means.clone()[idx]
                    covs[idx] = true_cond_covs.clone()[idx]
                    k = torch.zeros(idx[0].shape[0], num_components)
                    k -= torch.logsumexp(k, dim=-1, keepdim=True)
                    comp_log_probs[idx] = k
                else:
                    means[idx] = torch.rand(idx[0].shape[0], num_components, dim)*6 - 3
                    c = torch.rand(idx[0].shape[0], num_components, dim, dim)*stdv[idx].reshape(-1, 1, 1, 1) - stdv[idx].reshape(-1, 1, 1, 1)
                    c = c.transpose(-2, -1) @ c
                    covs[idx] = c
                    k = torch.zeros(idx[0].shape[0], num_components)
                    k -= torch.logsumexp(k, dim=-1, keepdim=True)
                    comp_log_probs[idx] = k

    return comp_log_probs, means, covs

def batched_per_datapoint_fit_mogs_sparse(X, M, num_components, num_iterations, *,
                                          dont_reset_on_failure=False,
                                          init_params_to_true_joint=False,
                                          init_params_to_true_cond=False,
                                          true_comp_log_probs=None,
                                          true_means=None,
                                          true_covs=None,
                                          true_cond_comp_log_probs=None,
                                          true_cond_means=None,
                                          true_cond_covs=None,
                                          ):
    assert not (init_params_to_true_joint and init_params_to_true_cond)
    dim = X.shape[-1]
    stdv = 1.0 / M.sum(-1)
    stdv[stdv == float('inf')] = 1.

    # Initialise params
    if init_params_to_true_joint:
        print('Initialising MoGs to true joint params')
        means = repeat(true_means, 'c d -> b c d', b=X.shape[0])
        covs = repeat(true_covs, 'c d1 d2 -> b c d1 d2', b=X.shape[0])
        comp_log_probs = torch.zeros(X.shape[0], num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)
    elif init_params_to_true_cond:
        print('Initialising MoGs to true conditional params')
        means = true_cond_means.clone()
        covs = true_cond_covs.clone()
        comp_log_probs = torch.zeros(X.shape[0], num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)
    else:
        means = torch.rand(X.shape[0], num_components, dim)*6 - 3
        # covs = torch.rand(X.shape[0], num_components, dim, dim)*stdv.reshape(-1, 1, 1, 1) - stdv.reshape(-1, 1, 1, 1)
        # covs = covs.transpose(-2, -1) @ covs
        covs = torch.randn(X.shape[0], num_components, dim, dim)*stdv.reshape(-1, 1, 1, 1)
        covs = covs @ covs.transpose(-2, -1)
        comp_log_probs = torch.zeros(X.shape[0], num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)

    for t in tqdm(range(num_iterations), desc='Fitting MoGs'):
        for i in range(X.shape[0]):
            X_i = X[i]
            M_i = M[i]
            comp_log_probs_i = comp_log_probs[i]
            means_i = means[i]
            covs_i = covs[i]

            comp_log_probs_i, means_i, covs_i = update_mog_parameters_batched_per_datapoint(X_i, M_i, comp_log_probs_i, means_i, covs_i)

            comp_log_probs[i] = comp_log_probs_i
            means[i] = means_i
            covs_i = covs_i

        if not dont_reset_on_failure:
            # Check for failures
            # NOTE: here I'm using magnitude of mean values as an indication of divergence, I do not indend to have such large values in the ground truth.
            idx = torch.where(~(comp_log_probs.sum(-1)).isfinite() | ~(means.sum(-1).sum(-1)).isfinite() | ~(covs.sum(-1).sum(-1).sum(-1)).isfinite()
                            | ((means > 30) | (means < -30)).any(-1).any(-1) | ((covs > 25) | (covs < -25)).any(-1).any(-1).any(-1))
            if len(idx[0]) > 0:
                print(f'Resetting some MoGs at timestep {t}:', idx[0])

                if init_params_to_true_joint:
                    n_reset = idx[0].shape[0]
                    means[idx] = repeat(true_means, 'c d -> b c d', b=n_reset)
                    covs[idx] = repeat(true_covs, 'c d1 d2 -> b c d1 d2', b=n_reset)
                    k = torch.zeros(idx[0].shape[0], num_components)
                    k -= torch.logsumexp(k, dim=-1, keepdim=True)
                    comp_log_probs[idx] = k
                elif init_params_to_true_cond:
                    means[idx] = true_cond_means.clone()[idx]
                    covs[idx] = true_cond_covs.clone()[idx]
                    k = torch.zeros(idx[0].shape[0], num_components)
                    k -= torch.logsumexp(k, dim=-1, keepdim=True)
                    comp_log_probs[idx] = k
                else:
                    means[idx] = torch.rand(idx[0].shape[0], num_components, dim)*6 - 3
                    c = torch.rand(idx[0].shape[0], num_components, dim, dim)*stdv[idx].reshape(-1, 1, 1, 1) - stdv[idx].reshape(-1, 1, 1, 1)
                    c = c.transpose(-2, -1) @ c
                    covs[idx] = c
                    k = torch.zeros(idx[0].shape[0], num_components)
                    k -= torch.logsumexp(k, dim=-1, keepdim=True)
                    comp_log_probs[idx] = k

    return comp_log_probs, means, covs


def sample_sparse_mog(num_samples, M, comp_log_probs, cond_means, cond_covs, *, sampling_batch_size=None):
    M_eye = torch.eye(cond_covs.shape[-1]).unsqueeze(0).unsqueeze(0) * (M).unsqueeze(1).unsqueeze(-1)
    cond_covs = cond_covs + M_eye
    # NOTE: in pytorch Cholesky decomp is only implemented for positive definite matrices
    L, info = torch.linalg.cholesky_ex(cond_covs)
    # errors = info > 0
    # if errors.sum() > 0:
    #     breakpoint()
    # Instead we can implement Cholesky decomposition for positive semi-definite matrices using eigen decomposition
    # L = semi_definite_symmetric_cholesky(cond_covs)

    # Sample component index
    comp_probs = torch.exp(comp_log_probs)
    comp_distr = torch.distributions.Categorical(probs=comp_probs, validate_args=True)
    component_idx = comp_distr.sample(sample_shape=(num_samples,))
    component_idx = component_idx.T

    if sampling_batch_size is None:
        # Select covariance (L)
        L_ = L[torch.arange(M.shape[0])[:, None], component_idx]

        # Select mean
        means_ = cond_means[torch.arange(M.shape[0])[:, None], component_idx]

        distr = torch.distributions.MultivariateNormal(loc=means_, scale_tril=L_,
                                                    validate_args=False) # Turn off validation, since it does not like negative diagonals
        X = distr.sample()*(~M).unsqueeze(1)
    else:
        X = []
        for b in range(math.ceil(num_samples/sampling_batch_size)):
            component_idx_b = component_idx[:, b*sampling_batch_size:min((b+1)*sampling_batch_size, num_samples)]

            # Select covariance (L)
            L_b = L[torch.arange(M.shape[0])[:, None], component_idx_b]

            # Select mean
            means_ = cond_means[torch.arange(M.shape[0])[:, None], component_idx_b]

            distr = torch.distributions.MultivariateNormal(loc=means_, scale_tril=L_b,
                                                        validate_args=False)

            X_b = distr.sample()*(~M).unsqueeze(1)
            X.append(X_b)
        X = torch.cat(X, dim=1)

    return X.float()

def sample_sparse_mog_diag(num_samples, M, comp_log_probs, cond_means, cond_vars, *, sampling_batch_size=None):
    # Sample component index
    comp_probs = torch.exp(comp_log_probs)
    comp_distr = torch.distributions.Categorical(probs=comp_probs, validate_args=True)
    component_idx = comp_distr.sample(sample_shape=(num_samples,))
    component_idx = component_idx.T

    cond_stds = cond_vars**0.5
    cond_stds = cond_stds*~M[:, None] + 1.*M[:, None] # Set observed dims to 1

    if sampling_batch_size is None:
        # Select covariance (L)
        std_ = cond_stds[torch.arange(M.shape[0])[:, None], component_idx]

        # Select mean
        means_ = cond_means[torch.arange(M.shape[0])[:, None], component_idx]

        distr = torch.distributions.Normal(loc=means_, scale=std_, validate_args=False)
        X = distr.sample()*(~M).unsqueeze(1)
    else:
        X = []
        for b in range(math.ceil(num_samples/sampling_batch_size)):
            component_idx_b = component_idx[:, b*sampling_batch_size:min((b+1)*sampling_batch_size, num_samples)]

            # Select covariance (L)
            std_b = cond_stds[torch.arange(M.shape[0])[:, None], component_idx_b]

            # Select mean
            means_b = cond_means[torch.arange(M.shape[0])[:, None], component_idx_b]

            distr = torch.distributions.Normal(loc=means_b, scale=std_b, validate_args=False)

            X_b = distr.sample()*(~M).unsqueeze(1)
            X.append(X_b)
        X = torch.cat(X, dim=1)

    return X.float()

def compute_kl_div_for_sparse_mogs(params1, params2, M, N=1000000, use_solver=False, return_jsd_midpoint=False):
    """Estimates the KL divergence between two mixtures of Gaussians"""
    comp_log_probs1 = params1['comp_log_probs']
    means1 = params1['means']
    covs1 = params1['covs']

    comp_log_probs2 = params2['comp_log_probs']
    means2 = params2['means']
    covs2 = params2['covs']

    if comp_log_probs2.dtype != comp_log_probs1.dtype:
        comp_log_probs2 = comp_log_probs2.to(comp_log_probs1.dtype)
        means2 = means2.to(means1.dtype)
        covs2 = covs2.to(covs1.dtype)

    X = sample_sparse_mog(num_samples=N, M=M, comp_log_probs=comp_log_probs1, cond_means=means1, cond_covs=covs1)
    log_p1 = batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs=comp_log_probs1, means=means1, covs=covs1, use_solver=use_solver)
    log_p1 = torch.logsumexp(log_p1, dim=-1)
    log_p2 = batched_compute_joint_log_probs_sparse_mogs(X, M, comp_log_probs=comp_log_probs2, means=means2, covs=covs2, use_solver=use_solver)
    log_p2 = torch.logsumexp(log_p2, dim=-1)

    kl = (log_p1 - log_p2).mean(dim=1)

    if not return_jsd_midpoint:
        return kl
    else:
        # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
        log_pm = torch.logsumexp(rearrange([log_p1, log_p2], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_p1.device, dtype=log_p1.dtype))
        jsd_term = (log_p1 - log_pm).mean(dim=1)

        return kl, jsd_term

def compute_kl_div_for_sparse_mogs_perdatapoint(params1, params2, M, N=1000000, return_jsd_midpoint=False, validate_args=True):
    """Estimates the KL divergence between two mixtures of Gaussians"""
    comp_log_probs1 = params1['comp_log_probs']
    means1 = params1['means']
    covs1 = params1['covs']

    comp_log_probs2 = params2['comp_log_probs']
    means2 = params2['means']
    covs2 = params2['covs']

    if comp_log_probs2.dtype != comp_log_probs1.dtype:
        comp_log_probs2 = comp_log_probs2.to(comp_log_probs1.dtype)
        means2 = means2.to(means1.dtype)
        covs2 = covs2.to(covs1.dtype)

    X = sample_sparse_mog(num_samples=N, M=M, comp_log_probs=comp_log_probs1, cond_means=means1, cond_covs=covs1)
    log_p1 = torch.empty(M.shape[0], N, device=X.device, dtype=X.dtype)
    log_p2 = torch.empty(M.shape[0], N, device=X.device, dtype=X.dtype)
    for i in range(M.shape[0]):
        X_i = X[i]
        M_i = M[i]
        comp_log_probs1_i = comp_log_probs1[i]
        means1_i = means1[i]
        covs1_i = covs1[i]
        comp_log_probs2_i = comp_log_probs2[i]
        means2_i = means2[i]
        covs2_i = covs2[i]

        log_p1_i = compute_joint_log_probs_sparse_mogs_batched_for_single_miss_pattern(
            X_i, M_i, comp_log_probs=comp_log_probs1_i, means=means1_i, covs=covs1_i, validate_args=validate_args)
        log_p1_i = torch.logsumexp(log_p1_i, dim=-1)
        log_p1[i] = log_p1_i
        log_p2_i = compute_joint_log_probs_sparse_mogs_batched_for_single_miss_pattern(
            X_i, M_i, comp_log_probs=comp_log_probs2_i, means=means2_i, covs=covs2_i, validate_args=validate_args)
        log_p2_i = torch.logsumexp(log_p2_i, dim=-1)
        log_p2[i] = log_p2_i

    kl = (log_p1 - log_p2).mean(dim=1)

    if not return_jsd_midpoint:
        return kl
    else:
        # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
        log_pm = torch.logsumexp(rearrange([log_p1, log_p2], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_p1.device, dtype=log_p1.dtype))
        jsd_term = (log_p1 - log_pm).mean(dim=1)

        return kl, jsd_term

def compute_kl_div_for_sparse_mogs_diag_perdatapoint(params1, params2, M, N=1000000, return_jsd_midpoint=False, validate_args=True):
    """Estimates the KL divergence between two mixtures of diagonal Gaussians"""
    comp_log_probs1 = params1['comp_log_probs']
    means1 = params1['means']
    vars1 = params1['vars']

    comp_log_probs2 = params2['comp_log_probs']
    means2 = params2['means']
    vars2 = params2['vars']

    if comp_log_probs2.dtype != comp_log_probs1.dtype:
        comp_log_probs2 = comp_log_probs2.to(comp_log_probs1.dtype)
        means2 = means2.to(means1.dtype)
        vars2 = vars2.to(vars1.dtype)

    X = sample_sparse_mog_diag(num_samples=N, M=M, comp_log_probs=comp_log_probs1, cond_means=means1, cond_vars=vars1)
    log_p1 = torch.empty(M.shape[0], N, device=X.device, dtype=X.dtype)
    log_p2 = torch.empty(M.shape[0], N, device=X.device, dtype=X.dtype)
    for i in range(M.shape[0]):
        X_i = X[i]
        M_i = M[i]
        comp_log_probs1_i = comp_log_probs1[i]
        means1_i = means1[i]
        vars1_i = vars1[i]
        comp_log_probs2_i = comp_log_probs2[i]
        means2_i = means2[i]
        vars2_i = vars2[i]

        log_p1_i = compute_joint_log_probs_sparse_mogs_diag_batched_for_single_miss_pattern(
            X_i, M_i, comp_log_probs=comp_log_probs1_i, means=means1_i, vars=vars1_i, validate_args=validate_args)
        log_p1_i = torch.logsumexp(log_p1_i, dim=-1)
        log_p1[i] = log_p1_i
        log_p2_i = compute_joint_log_probs_sparse_mogs_diag_batched_for_single_miss_pattern(
            X_i, M_i, comp_log_probs=comp_log_probs2_i, means=means2_i, vars=vars2_i, validate_args=validate_args)
        log_p2_i = torch.logsumexp(log_p2_i, dim=-1)
        log_p2[i] = log_p2_i

    kl = (log_p1 - log_p2).mean(dim=1)

    if not return_jsd_midpoint:
        return kl
    else:
        # log-avg-exp to compute log_pm, where pm is the midpoint distribution 0.5(p1 + p2)
        log_pm = torch.logsumexp(rearrange([log_p1, log_p2], 'p ... -> p ...'), dim=0) - torch.log(torch.tensor(2, device=log_p1.device, dtype=log_p1.dtype))
        jsd_term = (log_p1 - log_pm).mean(dim=1)

        return kl, jsd_term


def semi_definite_symmetric_cholesky(A):
    """
    Slightly slower than Cholesky, but allows semi-definite matrices

    See https://math.stackexchange.com/questions/45963/relation-between-cholesky-and-svd
    """
    eigenvals, eigenvecs = torch.linalg.eigh(A + torch.eye(A.shape[-1])*1e-6)

    B = eigenvecs.transpose(-1, -2) * eigenvals.unsqueeze(-1)**0.5
    B = B.to(A.dtype)

    _, R = torch.linalg.qr(B)

    L = R.transpose(-1, -2)

    return L

def compute_p_z_given_xo_from_imps(imps_b, params, *, K):
    # Get conditional log-probs from imputations
    imps_b = torch.tensor(rearrange(imps_b, 'b k d -> (b k) d')).to(torch.float32)
    log_prob_c_given_x_b = compute_joint_p_c_given_xo_xm(imps_b,
                                                         params['comp_logits'],
                                                         params['means'],
                                                         params['covs'])
    log_prob_c_given_x_b = rearrange(log_prob_c_given_x_b, '(b k) c -> b k c', k=K)
    # p(z | xo, xm)
    log_prob_c_given_ximps_b = log_prob_c_given_x_b

    # p(z | xo)
    mog_cond_log_probs_from_imps_b = torch.logsumexp(log_prob_c_given_ximps_b, dim=1) - np.log(K, dtype=np.float32)

    return {
        'log_prob_c_given_ximps_b': log_prob_c_given_ximps_b,
        'mog_cond_log_probs_from_imps_b': mog_cond_log_probs_from_imps_b,
    }

def supervised_mog_fit_using_groundtruth_z(imps_b, masks_b, true_params, *, K):
    # def is_psd(mat):
    #     return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>=0).all())
    def is_pd(mat):
        return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real>0).all())

    # Compute true conditional MoG parameters
    mog_true_cond_comp_log_probs_b, mog_true_cond_means_b, mog_true_cond_covs_b = compute_conditional_mog_parameters(
        torch.tensor(imps_b[:, 0, :]), torch.tensor(masks_b).squeeze(1),
        true_params['comp_logits'],
        true_params['means'],
        true_params['covs'])

    out = compute_p_z_given_xo_from_imps(imps_b, params=true_params, K=K)
    log_prob_c_given_ximps_b = out['log_prob_c_given_ximps_b']
    mog_cond_log_probs_from_imps_b = out['mog_cond_log_probs_from_imps_b']

    # NOTE: Added later
    # Fit the conditional Gaussians (without EM)
    imps_b = rearrange(imps_b, '(b k) d -> b k d', k=K)

    # Get weights for each imputation using the ground truth model
    # prob_c_given_ximps_b = torch.exp(log_prob_c_given_ximps_b)

    # Fit the conditional Gaussians using the weights
    cond_covs_from_imps_b = []
    cond_means_from_imps_b = []
    for i in range(imps_b.shape[0]):
        cond_covs_i = []
        cond_means_i = []
        mask_i = masks_b[i].squeeze(0)
        imps_b_i = imps_b[i, :, :][:, ~mask_i]
        D_i = imps_b_i.shape[-1]

        for c in range(log_prob_c_given_ximps_b.shape[-1]):
            # W = prob_c_given_ximps_b[i, :, c]
            # sum_W = W.sum()
            # W = W / sum_W
            log_W = log_prob_c_given_ximps_b[i, :, c]
            log_sum_W = torch.logsumexp(log_W, dim=0)
            sum_W = torch.exp(log_sum_W)
            log_W_norm = log_W - log_sum_W
            W = torch.exp(log_W_norm)
            print('sum_W', sum_W)
            if torch.any(torch.isclose(sum_W, torch.tensor(0., dtype=W.dtype))):
                # Just place a dummy diagonal Normal distribution if the weights are all zero
                print('All weights are zero')
                cov_i_c = torch.eye(imps_b_i.shape[-1])
                mean_i_c = torch.zeros((imps_b_i.shape[-1],))
            elif torch.any(sum_W <= D_i/3):
                print('Very low effective sample size')
                # If we have very few effective samples, we can't compute the covariance matrix
                # So we fit a diagonal Normal distribution instead
                def weighted_variance(X, W, sum_W):
                    # Compute the weighted mean
                    weighted_mean = torch.sum(X*W[:, None], dim=0)

                    # Compute the weighted variance
                    weighted_variance = torch.sum(((X - weighted_mean) ** 2)*W[:, None], dim=0) * sum_W / (sum_W-1)
                    return weighted_variance
                cov_i_c = torch.diag(weighted_variance(imps_b_i, W, sum_W)) + torch.eye(imps_b_i.shape[-1])*1e-6
                mean_i_c = torch.sum(imps_b_i*W[:, None], dim=0)
            else:
                print('Fitting normally')
                mean_i_c = torch.sum(imps_b_i*W[:, None], dim=0)
                # cov_i_c = torch.cov(imps_b_i.T, aweights=W) #+ torch.eye(imps_b_i.shape[-1])*1e-6
                diff = imps_b_i - mean_i_c
                W_sqrt = torch.sqrt(W)
                diff = diff * W_sqrt[:, None]
                cov_i_c = ((diff.T @ diff) * (sum_W / (sum_W - 1))) + torch.eye(diff.shape[-1])*1e-6
                # breakpoint()
                # cov_i_c = torch.cov(imps_b_i[W > 0.].T)
                # compute_gaussian_kl_div(asnumpy(mog_true_cond_means_b[i, c][~mask_i]), asnumpy(mog_true_cond_covs_b[i,c][~mask_i, :][:, ~mask_i]), asnumpy(mean_i_c), asnumpy(cov_i_c))

                # N_samples = 10000
                # idx = torch.multinomial(W, num_samples=N_samples, replacement=True)
                # imps_b_i_temp = imps_b_i[idx]

                # mean_i_c = torch.mean(imps_b_i_temp, dim=0)
                # # diff2 = imps_b_i_temp - mean_i_c
                # # cov_i_c = ((diff2.T @ diff2) / (N_samples - 1)) + torch.eye(mean_i_c.shape[-1])*1e-6
                # cov_i_c = torch.cov(imps_b_i_temp.T) + torch.eye(mean_i_c.shape[-1])*1e-6
                # breakpoint()

            if not is_pd(cov_i_c):
                # print('Not PSD')
                print('Not Positive Definite')
                # If the effective number of samples is too low, the covariance matrix may not be PSD
                # We can use one of the "robust" covariance estimation methods instead
                # https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/old_IDAPILecture16.pdf
                # https://scikit-learn.org/stable/modules/covariance.html
                N_samples = 1000
                idx = torch.multinomial(W, num_samples=N_samples, replacement=True)
                imps_b_i_temp = imps_b_i[idx]

                # glc = GraphicalLassoCovariance(max_iter=100, mode='lars')
                # glc.fit(asnumpy(imps_b_i_temp))

                # cov_i_c = torch.tensor(glc.covariance_.astype(np.float32))

                oas = OAS_Covariance()
                oas.fit(asnumpy(imps_b_i_temp))
                cov_i_c = torch.tensor(oas.covariance_.astype(np.float32)) + torch.eye(imps_b_i.shape[-1])*1e-6
                # lw = LedoitWolfCovariance()
                # lw.fit(asnumpy(imps_b_i_temp))
                # cov_i_c = torch.tensor(lw.covariance_.astype(np.float32))

                assert is_pd(cov_i_c)

                del imps_b_i_temp
                del idx
                # del glc
                del oas
                # del lw

            cond_covs_i.append(cov_i_c)
            cond_means_i.append(mean_i_c)
        cond_covs_i = torch.stack(cond_covs_i, dim=0)
        cond_means_i = torch.stack(cond_means_i, dim=0)

        # Place in the full-sized matrix
        cond_covs_i_temp = torch.zeros(cond_covs_i.shape[0], imps_b.shape[-1], imps_b.shape[-1])
        mask_i_mask_i = torch.tensor(~mask_i[None, :] & ~mask_i[:, None])
        cond_covs_i_temp[:, mask_i_mask_i] = cond_covs_i.reshape(cond_covs_i.shape[0], -1)
        cond_covs_i = cond_covs_i_temp

        cond_means_i_temp = torch.zeros(cond_covs_i.shape[0], imps_b.shape[-1])
        cond_means_i_temp[:, ~mask_i] = cond_means_i
        cond_means_i = cond_means_i_temp

        cond_covs_from_imps_b.append(cond_covs_i)
        cond_means_from_imps_b.append(cond_means_i)
    cond_covs_from_imps_b = torch.stack(cond_covs_from_imps_b, dim=0)
    cond_means_from_imps_b = torch.stack(cond_means_from_imps_b, dim=0)

    return {
        # p(z | xo)
        'mog_true_cond_comp_log_probs_b': mog_true_cond_comp_log_probs_b,
        # p(xm | z, xo)
        'mog_true_cond_means_b': mog_true_cond_means_b,
        'mog_true_cond_covs_b': mog_true_cond_covs_b,

        # qp(z | xo, xm), where xm are imputations
        'log_prob_c_given_ximps_b': log_prob_c_given_ximps_b,
        # qp(z | xo)
        'mog_cond_log_probs_from_imps_b': mog_cond_log_probs_from_imps_b,

        # q(xm | xo, z)
        'cond_means_from_imps_b': cond_means_from_imps_b,
        'cond_covs_from_imps_b': cond_covs_from_imps_b
    }

def supervised_mog_diag_fit_using_groundtruth_z(imps_b, masks_b, true_params, *, K, min_max_vals=(None, None), clamp_imps_effectivesamples=-1):
    # Compute true conditional MoG parameters
    mog_true_cond_comp_log_probs_b, mog_true_cond_means_b, mog_true_cond_covs_b = compute_conditional_mog_parameters(
        torch.tensor(imps_b[:, 0, :]), torch.tensor(masks_b).squeeze(1),
        true_params['comp_logits'],
        true_params['means'],
        true_params['covs'])

    # Get conditional log-probs from imputations
    imps_b = torch.tensor(rearrange(imps_b, 'b k d -> (b k) d')).to(torch.float32)
    log_prob_c_given_x_b = compute_joint_p_c_given_xo_xm(imps_b,
                                                         true_params['comp_logits'],
                                                         true_params['means'],
                                                         true_params['covs'])
    log_prob_c_given_x_b = rearrange(log_prob_c_given_x_b, '(b k) c -> b k c', k=K)
    # p(z | xo, xm)
    log_prob_c_given_ximps_b = log_prob_c_given_x_b

    # p(z | xo)
    mog_cond_log_probs_from_imps_b = torch.logsumexp(log_prob_c_given_ximps_b, dim=1) - np.log(K, dtype=np.float32)

    # Fit the conditional diagonal Gaussians (without EM)
    imps_b = rearrange(imps_b, '(b k) d -> b k d', k=K)

    # Get weights for each imputation using the ground truth model
    # prob_c_given_ximps_b = torch.exp(log_prob_c_given_ximps_b)

    # Fit the conditional Gaussians using the weights
    cond_vars_from_imps_b = []
    cond_means_from_imps_b = []
    for i in range(imps_b.shape[0]):
        cond_vars_i = []
        cond_means_i = []
        mask_i = masks_b[i].squeeze(0)
        imps_b_i = imps_b[i, :, :]

        for c in range(log_prob_c_given_ximps_b.shape[-1]):
            # W = prob_c_given_ximps_b[i, :, c]
            # sum_W = W.sum()
            # W = W / sum_W
            log_W = log_prob_c_given_ximps_b[i, :, c]
            log_sum_W = torch.logsumexp(log_W, dim=0)
            sum_W = torch.exp(log_sum_W)
            log_W_norm = log_W - log_sum_W
            W = torch.exp(log_W_norm)

            imps_b_i_c = imps_b_i
            # Clamping the imputations to data range if the effective sample size is very low
            if sum_W <= clamp_imps_effectivesamples:
                imps_b_i_c = torch.clamp(imps_b_i, min=min_max_vals[0], max=min_max_vals[1])

            def weighted_mean_and_variance(X, W, sum_W):
                X = X.to(torch.float64)
                # Compute the weighted mean
                weighted_mean = torch.sum(X*W[:, None], dim=0)

                # Compute the weighted variance
                weighted_variance = torch.sum(((X - weighted_mean) ** 2)*W[:, None], dim=0) * sum_W / (sum_W-1)

                weighted_mean = weighted_mean.to(torch.float32)
                weighted_variance = weighted_variance.to(torch.float32)
                return weighted_mean, weighted_variance
            # breakpoint()
            mean_i_c, var_i_c = weighted_mean_and_variance(imps_b_i_c, W, sum_W) #+ 1e-6
            # breakpoint()
            if (~torch.isfinite(var_i_c)).any():
                print('WARNING: Some variances are not finite, setting to 1e-6')
                var_i_c[~torch.isfinite(var_i_c)] = 1e-6
            # For numerical reasons
            var_i_c = torch.clamp(var_i_c, min=1e-6)

            # if i == 2 and c == 7:
            #     breakpoint()

            cond_vars_i.append(var_i_c)
            cond_means_i.append(mean_i_c)
        cond_vars_i = torch.stack(cond_vars_i, dim=0)
        cond_means_i = torch.stack(cond_means_i, dim=0)

        # Place set the observed dims to zero
        cond_vars_i[:, mask_i] = 0.
        cond_means_i[:, mask_i] = 0.

        cond_vars_from_imps_b.append(cond_vars_i)
        cond_means_from_imps_b.append(cond_means_i)
    cond_vars_from_imps_b = torch.stack(cond_vars_from_imps_b, dim=0)
    cond_means_from_imps_b = torch.stack(cond_means_from_imps_b, dim=0)

    mog_true_cond_vars_b = torch.diagonal(mog_true_cond_covs_b, dim1=-2, dim2=-1)
    return {
        # p(z | xo)
        'mog_true_cond_comp_log_probs_b': mog_true_cond_comp_log_probs_b,
        # p(xm | z, xo)
        'mog_true_cond_means_b': mog_true_cond_means_b,
        # 'mog_true_cond_covs_b': mog_true_cond_covs_b,
        'mog_true_cond_vars_b': mog_true_cond_vars_b,

        # qp(z | xo, xm), where xm are imputations
        'log_prob_c_given_ximps_b': log_prob_c_given_ximps_b,
        # qp(z | xo)
        'mog_cond_log_probs_from_imps_b': mog_cond_log_probs_from_imps_b,

        # q(xm | xo, z)
        'cond_means_from_imps_b': cond_means_from_imps_b,
        'cond_vars_from_imps_b': cond_vars_from_imps_b
    }

if __name__ == '__main__':
    from irwg.data.toy import ToyDataset
    filename='data_mog_10d'
    dataset = ToyDataset(root='./data', filename=filename, split='test')

    true_params = {
        'comp_log_probs': torch.tensor(np.log(dataset.data_file['comp_probs'])),
        'covs': torch.tensor(dataset.data_file['covs']),
        'means': torch.tensor(dataset.data_file['means'])
    }
    data = torch.tensor(dataset[:22])
    masks = torch.rand(data.shape) > 0.5
    # Make sure there is at least one complete and one missing observation in each sample
    while (masks.sum(dim=-1) == 0).any() or (masks.float().mean(dim=-1) == 1).any():
        masks = torch.rand(data.shape) > 0.5

    # Verify conditional mog parameters

    comp_log_probs_given_o, means_m_given_o, covs_m_given_o = compute_conditional_mog_parameters(data, masks, true_params['comp_log_probs'], true_params['means'], true_params['covs'])
    for i in range(len(data)):
        x = data[i][:]
        m = masks[i][:]

        log_pi_i, means_i, covs_i = compute_conditional_mog_parameters_nonbatched(x, m,
                                                                                 true_params['comp_log_probs'], true_params['means'], true_params['covs'])


        assert(np.allclose(log_pi_i, comp_log_probs_given_o[i]))
        assert(np.allclose(means_i, means_m_given_o[i]))
        assert(np.allclose(covs_i, covs_m_given_o[i]))

    # Verify joint log_probs

    joint_log_probs_batched = batched_compute_joint_log_probs_sparse_mogs(data.unsqueeze(1), masks, comp_log_probs_given_o, means_m_given_o, covs_m_given_o)
    joint_log_probs_batched = joint_log_probs_batched.squeeze(1)
    for i in range(len(data)):
        x = data[i][:]
        m = masks[i][:]

        joint_log_probs = compute_joint_log_probs_sparse_mogs_nonbatched(x, m,
                                                                   comp_log_probs_given_o[i], means_m_given_o[i], covs_m_given_o[i])
        assert(np.allclose(joint_log_probs, joint_log_probs_batched[i]))

    # Verify EM updates
    num_components = comp_log_probs_given_o.shape[-1]
    dim = data.shape[-1]

    # Using one pattern of missingness
    for m in masks:
        # m = masks[0]
        # Initialise params
        stdv = 1.0 / m.sum(-1)
        means = torch.rand(1, num_components, dim)*6 - 3
        # covs = torch.rand(X.shape[0], num_components, dim, dim)*stdv.reshape(-1, 1, 1, 1) - stdv.reshape(-1, 1, 1, 1)
        # covs = covs.transpose(-2, -1) @ covs
        covs = torch.randn(1, num_components, dim, dim)*stdv.reshape(-1, 1, 1, 1)
        covs = covs @ covs.transpose(-2, -1)
        comp_log_probs = torch.zeros(1, num_components)
        comp_log_probs -= torch.logsumexp(comp_log_probs, dim=-1, keepdim=True)

        # To avoid numerical discrepancies in comparison due to precision
        means = means.to(torch.double)
        covs = covs.to(torch.double)
        comp_log_probs = comp_log_probs.to(torch.double)

        num_em_iterations = 5
        data_m = data * ~m
        for iter in range(num_em_iterations):
            comp_log_probs_up, means_up, covs_up = batched_update_mogs_parameters(
                data_m.unsqueeze(0), m.unsqueeze(0), comp_log_probs, means, covs, use_solver=False)

            comp_log_probs_up_ref, means_up_ref, covs_up_ref = update_mog_parameters_nonbatched(data_m, m, comp_log_probs, means, covs)

            assert(np.allclose(comp_log_probs_up, comp_log_probs_up_ref, rtol=1e-5, atol=1e-5))
            assert(np.allclose(means_up, means_up_ref, rtol=1e-5, atol=1e-5))
            assert(np.allclose(covs_up, covs_up_ref, rtol=1e-5, atol=1e-5))

            # Finally, make sure that the observed dimensions are not relevant for the update
            comp_log_probs_up_fixing_observed, means_up_fixing_observed, covs_up_fixing_observed = batched_update_mogs_parameters(
                data_m.unsqueeze(0), m.unsqueeze(0), comp_log_probs, means*~m, covs*~m.unsqueeze(-2)*~m.unsqueeze(-1), use_solver=False)

            assert(np.allclose(comp_log_probs_up, comp_log_probs_up_fixing_observed, rtol=1e-5, atol=1e-5))
            assert(np.allclose(means_up, means_up_fixing_observed, rtol=1e-5, atol=1e-5))
            assert(np.allclose(covs_up, covs_up_fixing_observed, rtol=1e-5, atol=1e-5))

            covs = covs_up
            comp_log_probs = comp_log_probs_up
            means = means_up

        # Verify that the observed dimension covariances are zero
        convs_oo = covs[:, :, :, m][:, :, m]
        # Subtract the diagonal
        convs_oo = convs_oo - torch.eye(convs_oo.shape[-1])*convs_oo[0,0,0,0]
        assert(np.allclose(convs_oo, 0, atol=1e-5))

    # Verify log p(c | x) computation
    log_p_c_given_x = compute_joint_p_c_given_xo_xm(data, comp_log_probs=true_params['comp_log_probs'], means=true_params['means'], covs=true_params['covs'])
    log_p_c_given_x_from_sparse_impl, _, _ = compute_conditional_mog_parameters(data, torch.ones_like(masks), comp_log_probs=true_params['comp_log_probs'], means=true_params['means'], covs=true_params['covs'])

    assert np.allclose(log_p_c_given_x, log_p_c_given_x_from_sparse_impl)
