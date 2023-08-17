import math

import torch
import tqdm
from einops import reduce, rearrange, repeat
from irwg.models.vae import DISTRIBUTION, EPSILON
from irwg.utils.softplus_inverse import torch_softplus_inverse


from irwg.utils.weights import (
    compute_log_unnormalised_importance_weights,
    compute_mis_log_unnormalised_importance_weights)

def expand_M_dim(M, *, data_channel_dim=None):
    if data_channel_dim is not None:
        return M.unsqueeze(data_channel_dim)
    return M

def pseudo_gibbs_iteration(model, X, M, clip_values_min=None, clip_values_max=None, *, data_channel_dim=None):
    # Create latent distribution and sample
    var_latent_params = model.predict_var_latent_params(X, M)

    # Sample latent variables
    var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
    Z_imp = var_latent_distr.sample()

    # Create the distribution of the missingness model
    mis_params = model.generator_network(Z_imp)
    mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

    # Sample missing values
    X_m = mis_distr.sample()

    # Workaround for unstable Pseudo-Gibbs sampler
    if clip_values_min is not None:
        X_m = torch.max(X_m, clip_values_min)
    if clip_values_max is not None:
        X_m = torch.min(X_m, clip_values_max)
    # Another safety
    is_nan = ~(X_m.isfinite())
    X_m[is_nan] = X[is_nan]

    # Set imputed missing values
    M_expanded = expand_M_dim(M, data_channel_dim=data_channel_dim)
    return X*M_expanded + X_m*(~M_expanded), Z_imp

def pseudo_gibbs(model, X, M, num_iterations, metric = None, keep_last_n_iterations = 1, batchsize=-1, data_channel_dim=None):
    imp_seq = []
    stats = []
    for t in tqdm.tqdm(range(num_iterations)):
        if batchsize <= 0:
            X, Z_imp = pseudo_gibbs_iteration(model, X, M)
        else:
            X_stack = []
            for b in range(math.ceil(X.shape[0] // batchsize)):
                X_b = X[b*batchsize, min((b+1)*batchsize, X.shape[0])]
                M_b = M[b*batchsize, min((b+1)*batchsize, M.shape[0])]
                X_b, Z_imp_b = pseudo_gibbs_iteration(model, X_b, M_b, data_channel_dim=data_channel_dim)
                X_stack.append(X_b)
            X = torch.vstack(X_stack)

        if metric is not None:
            stats.append(metric(X))

        if t >= num_iterations-keep_last_n_iterations:
            imp_seq.append(X.cpu())
    return rearrange(imp_seq, 't ... -> t ...'), stats

def metropolis_within_gibbs(model, X, M, pseudo_warmup_iterations, num_iterations, metric = None, keep_last_n_iterations = 1, *, data_channel_dim=None):
    # M_not = ~M
    M_expanded = expand_M_dim(M, data_channel_dim=data_channel_dim)
    M_not_expanded = ~M_expanded

    imp_seq = []
    stats = []
    stats_acceptance = []
    for t in tqdm.tqdm(range(pseudo_warmup_iterations)):
        # Create latent distribution and sample
        var_latent_params = model.predict_var_latent_params(X, M)

        # Sample latent variables
        var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
        Z_new = var_latent_distr.sample()

        # Create the distribution of the missingness model
        mis_params = model.generator_network(Z_new)
        mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # Sample missing values
        X_m = mis_distr.sample()

        # Set imputed missing values
        # X_new =  X*M + X_m*M_not
        X_new = X*M_expanded + X_m*M_not_expanded
        X = X_new

        if metric is not None:
            stats.append(metric(X))

        stats_acceptance.append(1)
        if t >= pseudo_warmup_iterations+num_iterations-keep_last_n_iterations:
            imp_seq.append(X.cpu())

    # prior_distr = torch.distributions.Normal(0, 1)
    prior_distr = model.get_prior()

    Z_old = Z_new
    mis_params_old = mis_params
    Z_old_prior_logprob = reduce(prior_distr.log_prob(Z_old), '... d -> ...', 'sum')
    X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

    for t in tqdm.tqdm(range(pseudo_warmup_iterations, pseudo_warmup_iterations+num_iterations)):
        # Create latent distribution and sample
        var_latent_params = model.predict_var_latent_params(X, M)

        # Sample latent variables
        var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
        Z_new = var_latent_distr.sample()
        Z_new_var_logprob = reduce(var_latent_distr.log_prob(Z_new), '... d -> ...', 'sum')
        Z_old_var_logprob = reduce(var_latent_distr.log_prob(Z_old), '... d -> ...', 'sum')

        # Eval prior
        Z_new_prior_logprob = reduce(prior_distr.log_prob(Z_new), '... d -> ...', 'sum')

        # Create the distribution of the missingness model
        mis_params = model.generator_network(Z_new)
        mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        X_new_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

        log_accept = (X_new_logprob + Z_new_prior_logprob) - (X_old_logprob + Z_old_prior_logprob) + (Z_old_var_logprob - Z_new_var_logprob)

        # acceptance_prob = torch.exp(torch.clamp(log_accept, -25, 25))
        acceptance_prob = torch.exp(log_accept)

        acceptance_samples = torch.rand_like(acceptance_prob)
        accepted = acceptance_samples < acceptance_prob

        accepted_not = ~accepted
        Z_old = Z_old*accepted_not.unsqueeze(-1) + Z_new*accepted.unsqueeze(-1)
        Z_old_prior_logprob = Z_old_prior_logprob*accepted_not + Z_new_prior_logprob*accepted

        mis_params_old = mis_params_old*accepted_not.unsqueeze(-1) + mis_params*accepted.unsqueeze(-1)
        mis_distr = model.create_distribution(mis_params_old, model.hparams.generator_distribution)

        # Sample missing values
        X_m = mis_distr.sample()

        # Set imputed missing values
        # X_new =  X*M + X_m*M_not
        X_new = X*M_expanded + X_m*M_not_expanded
        X = X_new

        X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

        if metric is not None:
            stats.append(metric(X))

        stats_acceptance.append(accepted.float().mean())
        if t >= pseudo_warmup_iterations+num_iterations-keep_last_n_iterations:
            imp_seq.append(X.cpu())

    return rearrange(imp_seq, 't ... -> t ...'), stats, stats_acceptance

def mnar_metropolis_within_gibbs(model, X, M, pseudo_warmup_iterations, num_iterations, metric = None, keep_last_n_iterations = 1,
                                 miss_model: torch.nn.Module = None, *, data_channel_dim=None):
    # M_not = ~M
    M_expanded = expand_M_dim(M, data_channel_dim=data_channel_dim)
    M_not_expanded = ~M_expanded

    imp_seq = []
    stats = []
    stats_acceptance = []
    # NOTE: no correction during pseudo-Gibbs
    for t in tqdm.tqdm(range(pseudo_warmup_iterations)):
        # Create latent distribution and sample
        var_latent_params = model.predict_var_latent_params(X, M)

        # Sample latent variables
        var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
        Z_new = var_latent_distr.sample()

        # Create the distribution of the missingness model
        mis_params = model.generator_network(Z_new)
        mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # Sample missing values
        X_m = mis_distr.sample()

        # Set imputed missing values
        # X_new =  X*M + X_m*M_not
        X_new = X*M_expanded + X_m*M_not_expanded
        X = X_new

        if metric is not None:
            stats.append(metric(X))

        stats_acceptance.append(1)
        if t >= pseudo_warmup_iterations+num_iterations-keep_last_n_iterations:
            imp_seq.append(X.cpu())

    # prior_distr = torch.distributions.Normal(0, 1)
    prior_distr = model.get_prior()
    del mis_params, mis_distr, var_latent_distr, X_new, X_m, var_latent_params

    Z = Z_new
    # mis_params_old = mis_params
    # Z_old_prior_logprob = reduce(prior_distr.log_prob(Z_old), '... d -> ...', 'sum')
    # X_old_logprob = reduce(mis_distr.log_prob(X), '... d -> ...', 'sum')

    for t in tqdm.tqdm(range(pseudo_warmup_iterations, pseudo_warmup_iterations+num_iterations)):
        # Create latent distribution and sample
        var_latent_params = model.predict_var_latent_params(X, M)

        # Sample latent variables
        var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
        Z_new = var_latent_distr.sample()
        Z_new_var_logprob = reduce(var_latent_distr.log_prob(Z_new), '... d -> ...', 'sum')
        Z_old_var_logprob = reduce(var_latent_distr.log_prob(Z), '... d -> ...', 'sum')

        # Eval prior
        Z_new_prior_logprob = reduce(prior_distr.log_prob(Z_new), '... d -> ...', 'sum')
        Z_old_prior_logprob = reduce(prior_distr.log_prob(Z), '... d -> ...', 'sum')

        # Create the distribution of the missingness model
        mis_params_new = model.generator_network(Z_new)
        mis_params_old = model.generator_network(Z)
        mis_distr_new = model.create_distribution(mis_params_new, model.hparams.generator_distribution)
        mis_distr_old = model.create_distribution(mis_params_old, model.hparams.generator_distribution)

        X_new_logprob = reduce(mis_distr_new.log_prob(X), '... d -> ...', 'sum')
        X_old_logprob = reduce(mis_distr_old.log_prob(X), '... d -> ...', 'sum')

        log_accept = (X_new_logprob + Z_new_prior_logprob) - (X_old_logprob + Z_old_prior_logprob) + (Z_old_var_logprob - Z_new_var_logprob)

        acceptance_prob = torch.exp(log_accept)

        acceptance_samples = torch.rand_like(acceptance_prob)
        accepted_z = acceptance_samples < acceptance_prob

        accepted_z_not = ~accepted_z
        Z = Z*accepted_z_not.unsqueeze(-1) + Z_new*accepted_z.unsqueeze(-1)
        # Z_old_prior_logprob = Z_old_prior_logprob*accepted_not + Z_new_prior_logprob*accepted

        #
        # Sample XM with MH acceptance using the miss_model
        #

        mis_params = mis_params_old*accepted_z_not.unsqueeze(-1) + mis_params_new*accepted_z.unsqueeze(-1)
        mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

        # Sample missing values
        X_m = mis_distr.sample()

        # Set imputed missing values
        # X_new =  X*M + X_m*M_not
        X_new = X*M_expanded + X_m*M_not_expanded

        mis_log_prob_old = miss_model.log_prob(X, M)
        mis_log_prob_new = miss_model.log_prob(X_new, M)

        log_accept = mis_log_prob_new - mis_log_prob_old

        acceptance_prob = torch.exp(log_accept)

        acceptance_samples = torch.rand_like(acceptance_prob)
        accepted_x = acceptance_samples < acceptance_prob

        accepted_x_not = ~accepted_x
        X = X*accepted_x_not.unsqueeze(-1) + X_new*accepted_x.unsqueeze(-1)

        if metric is not None:
            stats.append(metric(X))

        stats_acceptance.append(0.5*(accepted_z.float().mean()+accepted_x.float().mean()))
        if t >= pseudo_warmup_iterations+num_iterations-keep_last_n_iterations:
            imp_seq.append(X.cpu())

    return rearrange(imp_seq, 't ... -> t ...'), stats, stats_acceptance


# IRWG

def gr_resampling(log_weights, generator_params, resampling_method='multinomial'):
    B, K = log_weights.shape[1], log_weights.shape[2]
    # Shape (i b k)
    # NOTE: here I treat i*k as the number of importance samples
    log_weights = rearrange(log_weights, 'i b k -> b (i k)')

    # Sample from the importance-weighted distribution
    if resampling_method == 'multinomial':
        importance_distr = torch.distributions.Categorical(logits=log_weights)
        idx = importance_distr.sample(sample_shape=(K,))
    elif resampling_method == 'systematic':
        idx = _systematic_sample(log_weights, num_dependent_samples=K)
        idx = rearrange(idx, 'b k -> k b')
    else:
        raise NotImplementedError()

    # Get generator params for the corresponding Z's
    mis_params = rearrange(generator_params, 'i b k ... -> b (i k) ...')
    mis_params = mis_params[torch.arange(B, device=generator_params.device),
                            idx,
                            ...]
    mis_params = rearrange(mis_params, 'k b ... -> b k ...')

    norm_log_weights = log_weights - log_weights.logsumexp(dim=-1, keepdim=True)
    return mis_params, norm_log_weights

def lr_resampling(log_weights, generator_params, resampling_method='multinomial'):
    B, K = log_weights.shape[1], log_weights.shape[2]
    # Shape (i b k)
    log_weights = rearrange(log_weights, 'i b k -> b k i')

    # Sample from the importance-weighted distribution
    if resampling_method == 'multinomial':
        importance_distr = torch.distributions.Categorical(logits=log_weights)
        idx = importance_distr.sample()
    elif resampling_method == 'systematic':
        idx = _systematic_sample(log_weights, num_dependent_samples=1)
        idx = rearrange(idx, 'b k 1 -> b k')
    else:
        raise NotImplementedError()

    # Get generator params for the corresponding Z's
    mis_params = generator_params[idx,
                                  torch.arange(B, device=generator_params.device)[:, None],
                                  torch.arange(K, device=generator_params.device),
                                  ...]

    norm_log_weights = log_weights - log_weights.logsumexp(dim=-1, keepdim=True)
    norm_log_weights = rearrange(norm_log_weights, 'b k i -> b (k i)')
    return mis_params, norm_log_weights

def glr_resampling(log_weights, generator_params, *, cluster_size, pad_to=None, resampling_method='multinomial'):
    """Global-local resampling"""
    I, B, K = log_weights.shape[0], log_weights.shape[1], log_weights.shape[2]

    if pad_to is not None:
        padding = torch.ones(I, B, pad_to-K, device=log_weights.device)*float('-inf')
        log_weights = torch.cat([log_weights, padding], dim=-1)
    K2 = log_weights.shape[2] // cluster_size

    # Shape (i b k)
    log_weights = rearrange(log_weights, 'i b (c k) -> b k (i c)', c=cluster_size)

    # Sample from the importance-weighted distribution
    if resampling_method == 'multinomial':
        importance_distr = torch.distributions.Categorical(logits=log_weights)
        idx = importance_distr.sample(sample_shape=(cluster_size,))
    elif resampling_method == 'systematic':
        idx = _systematic_sample(log_weights, num_dependent_samples=cluster_size)
        idx = rearrange(idx, 'b k c -> c b k')
    else:
        raise NotImplementedError()

    # Get generator params for the corresponding Z's
    if pad_to is not None:
        P = generator_params.shape[-1]
        padding = torch.ones(I, B, pad_to-K, P, device=generator_params.device)*float('-inf')
        generator_params = torch.cat([generator_params, padding], dim=-2)

    mis_params = rearrange(generator_params, 'i b (c k) p -> b k (i c) p', c=cluster_size)
    mis_params = mis_params[torch.arange(B, device=generator_params.device)[:, None],
                            torch.arange(K2, device=generator_params.device),
                            idx,
                            ...]
    mis_params = rearrange(mis_params, 'c b k d -> b (c k) d')

    norm_log_weights = log_weights - log_weights.logsumexp(dim=-1, keepdim=True)
    if pad_to is not None:
        norm_log_weights = rearrange(norm_log_weights, 'b k (i c) -> i b (c k)', i=I)
        norm_log_weights = norm_log_weights[:, :, :K]
        norm_log_weights = rearrange(norm_log_weights, 'i b k -> b (i k)')
    else:
        norm_log_weights = rearrange(norm_log_weights, 'b k (i c) -> b (k i c)', i=I)

    return mis_params, norm_log_weights

def compute_groupeddmis_log_unnormalised_importance_weights(X, M, Z_imp,
            model,
            miss_model,
            var_latent_params,
            prior_distr,
            generator_distr,
            *,
            cluster_size: int,
            var_comp_neg_idx:int=-2,
            pad_to: int = None,
        ):
    B, K, ZPD = var_latent_params.shape[0], var_latent_params.shape[1], var_latent_params.shape[2]
    K2 = K // cluster_size

    # var_latent_params # (b k d)
    if pad_to is not None:
        padding = torch.ones(B, pad_to-K, ZPD, device=X.device)*float('-inf')
        var_latent_params = torch.cat([var_latent_params, padding], dim=1)

        pad_mask = torch.cat([torch.zeros(K, device=X.device, dtype=torch.bool),
                                torch.ones(pad_to-K, device=X.device, dtype=torch.bool)])
        pad_mask = rearrange(pad_mask, '(c k) -> k c', c=cluster_size)

        I = Z_imp.shape[0]
        ZD = Z_imp.shape[-1]
        Z_imp = torch.cat([Z_imp, torch.zeros(I, B, pad_to-K, ZD, device=X.device)], dim=2)

    Z_imp = rearrange(Z_imp, 'i b (c k) d -> i b k c d', c=cluster_size)
    var_latent_params = rearrange(var_latent_params, 'b (c k) d -> b k c d', c=cluster_size)
    # Turn off args validation since the padded arguments are not valid
    var_latent_distr = model.create_distribution(var_latent_params,
                                                    model.hparams.var_latent_distribution,
                                                    validate_args=False)

    # NOTE: Adapted version of `multiple_importance_logprob()`
    # Compute the log-prob of samples under the mixture var distribution
    Z_aug = torch.swapaxes(torch.unsqueeze(Z_imp, 0), 0, var_comp_neg_idx)

    latent_logprob = var_latent_distr.log_prob(Z_aug)
    latent_logprob = reduce(latent_logprob, '... d -> ...', 'sum')
    if pad_to is None:
        latent_logprob = torch.logsumexp(latent_logprob, dim=var_comp_neg_idx+1, keepdim=True)
        latent_logprob -= torch.log(torch.tensor(Z_imp.shape[var_comp_neg_idx], device=Z_imp.device))
    else:
        latent_logprob[..., pad_mask] = float('-inf')
        latent_logprob = torch.logsumexp(latent_logprob, dim=var_comp_neg_idx+1, keepdim=True)
        num_components_per_group = (~pad_mask).sum(-1, keepdim=True)
        latent_logprob -= torch.log(num_components_per_group)

    # Undo the swap axes above
    latent_logprob = torch.swapaxes(latent_logprob, 0, var_comp_neg_idx+1).squeeze(0)
    # NOTE: End of adapted `multiple_importance_logprob()`

    Z_imp = rearrange(Z_imp, 'i b k c d -> i b (c k) d', c=cluster_size)
    latent_logprob = rearrange(latent_logprob, 'i b k c -> i b (c k)')
    if pad_to is not None:
        latent_logprob = latent_logprob[..., :K]
        Z_imp = Z_imp[:, :, :K, :]

    # Compute cross-entropy term of observed data
    generator_logprob = generator_distr.log_prob(X)*M
    generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

    # Compute prior latent probability
    prior_logprob = prior_distr.log_prob(Z_imp)
    prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

    # Compute the total IWELBO weights
    log_weights = generator_logprob + prior_logprob - latent_logprob

    # If Missingness model is given - compute probability of missingness pattern (for MNAR data)
    if miss_model is not None:
        X_m_for_miss_model = generator_distr.sample()
        X_for_miss_model = X*M + X_m_for_miss_model*~M
        log_prob_M = miss_model.log_prob(X_for_miss_model, M)

        log_weights += log_prob_M

    return log_weights

def ir_resampling(log_weights, generator_params, resampling_method='multinomial'):
    B, K = log_weights.shape[1], log_weights.shape[2]
    # Shape (i b k)
    log_weights = rearrange(log_weights, 'i b k -> b i k')
    log_weights = log_weights.reshape(B, K, -1)

    # Sample from the importance-weighted distribution
    if resampling_method == 'multinomial':
        importance_distr = torch.distributions.Categorical(logits=log_weights)
        idx = importance_distr.sample()
    elif resampling_method == 'systematic':
        idx = _systematic_sample(log_weights, num_dependent_samples=1)
        idx = rearrange(idx, 'b k2 1 -> b k2')
    else:
        raise NotImplementedError()

    # Get generator params for the corresponding Z's
    mis_params = select_gen_params_for_ir(generator_params, idx, B, K)

    norm_log_weights = log_weights.reshape(B, -1, K)
    norm_log_weights = rearrange(norm_log_weights, 'b i k -> b (i k)')

    return mis_params, norm_log_weights

def importance_resampling_gibbs_iteration(model, X, M, num_imp_samples, *, glr_cluster_size=None,
                                          weighting='dmis', resampling_method='multinomial', resampling_scheme='gr',
                                          num_prior_replenish_proposals=0,
                                        #   return_weights=False,
                                          weight_transform=None,
                                          use_historical_proposals=None,
                                          num_historical_proposals=None,
                                          history=None,
                                          miss_model: torch.nn.Module = None,
                                          accept_reject_xm: bool = False,
                                          data_channel_dim=None,
                                          var_proposal_temperature=None,
                                          var_proposal_anneal_type_to_prior=None,
                                          return_resampled_latents=False,
                                          ):
    K = X.shape[1]
    M_expanded = expand_M_dim(M, data_channel_dim=data_channel_dim)
    M_not_expanded = ~M_expanded

    pad_glr_to = None
    if use_historical_proposals is not None and use_historical_proposals:
        if resampling_scheme == 'glr':
            # Handle the case when using GLR and history together
            # to make sure that each cluster only gets historical samples from the same cluster
            # If we have K=5 chains, and 10 historical samples, and we want to split into 3 clusters (of size 5, hence glr_cluster_size=5)
            # We create an index as follows:
            # [0, 3, 0, 3, 0]
            # [1, 4, 1, 4, 1] -> [0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 2, 0, 1, 2]
            # [2, 2, 2, 2, 2]
            # We then get rid of the first K elements, which are the K chains that do not come from history
            # So the final chain_idx is:
            # [2, 0, 1, 2, 3, 4, 2, 0, 1, 2]

            assert ((K + num_historical_proposals) % glr_cluster_size == 0)
            num_glr_groups = (K + num_historical_proposals) // glr_cluster_size
            assert (num_glr_groups <= K)

            # Create an remainder index to finish the "block", in the above example it's just [2]
            num_rem = math.ceil(K / num_glr_groups)*num_glr_groups-K
            rem_idx = torch.arange(num_glr_groups-num_rem, num_glr_groups)

            # Create the basic block, in the above example it is [0, 1, 2, 3, 4, 2]
            base_idx = torch.concat([torch.arange(K), rem_idx], dim=0)

            # Repeat the basic block
            num_reps_of_base = (K + num_historical_proposals) // len(base_idx)
            chain_idx = torch.concat(num_reps_of_base*[base_idx], dim=0)

            if (K + num_historical_proposals) % len(base_idx) != 0:
                num_final = (K + num_historical_proposals) - len(chain_idx)
                chain_idx = torch.concat([chain_idx, base_idx[:num_final]], dim=0)

            # Remove the first K elements that do not come from history
            chain_idx = chain_idx[K:]

            hist_proposals = history.sample_history_with_chain_idx_for_each_sample(num_historical_proposals, chain_idx)
        else:
            hist_proposals = history.sample_history(num_historical_proposals)

        # H B D
        hist_proposals = rearrange(hist_proposals, 'h b ... -> b h ...').to(X.device)
        X = torch.concat([X, hist_proposals], dim=1)
    elif use_historical_proposals is not None and not use_historical_proposals:
        if resampling_scheme == 'glr' and num_historical_proposals is not None:
            pad_glr_to = K + num_historical_proposals

    # Create latent distribution and sample
    var_latent_params = model.predict_var_latent_params(X, M)

    if var_proposal_temperature is not None:
        var_latent_params = model.distribution_param_tempering(var_latent_params,
                                                               model.hparams.var_latent_distribution,
                                                               temperature=var_proposal_temperature)
    if var_proposal_anneal_type_to_prior is not None:
                var_latent_params = model.anneal_var_distribution_to_prior_based_on_missingness(
                    var_latent_params, M,
                    var_distribution=model.hparams.var_latent_distribution,
                    anneal_type=var_proposal_anneal_type_to_prior)

    if num_prior_replenish_proposals > 0:
        # NOTE: assuming standard prior, also only works with gaussian variational distributions
        if model.hparams.var_latent_distribution == DISTRIBUTION.normal:
            replenish_means = torch.zeros(X.shape[0], num_prior_replenish_proposals, var_latent_params.shape[-1]//2, device=X.device)
            replenish_scales = torch.ones(X.shape[0], num_prior_replenish_proposals, var_latent_params.shape[-1]//2, device=X.device)
            replenish_scales = torch_softplus_inverse(replenish_scales)

            replenish_params = rearrange([replenish_means, replenish_scales], 'params ... d -> ... (params d)')

        elif model.hparams.var_latent_distribution == DISTRIBUTION.normal_with_eps:
            replenish_means = torch.zeros(X.shape[0], num_prior_replenish_proposals, var_latent_params.shape[-1]//2, device=X.device)
            replenish_scales = torch.ones(X.shape[0], num_prior_replenish_proposals, var_latent_params.shape[-1]//2, device=X.device)
            replenish_scales = torch_softplus_inverse(replenish_scales - EPSILON)

            replenish_params = rearrange([replenish_means, replenish_scales], 'params ... d -> ... (params d)')
        elif model.hparams.var_latent_distribution == DISTRIBUTION.beta:
            # Set to uniform prior
            replenish_alpha = torch.ones(X.shape[0], num_prior_replenish_proposals, var_latent_params.shape[-1]//2, device=X.device)
            replenish_alpha = torch_softplus_inverse(replenish_alpha)
            replenish_beta = torch.ones(X.shape[0], num_prior_replenish_proposals, var_latent_params.shape[-1]//2, device=X.device)
            replenish_beta = torch_softplus_inverse(replenish_beta)

            replenish_params = rearrange([replenish_alpha, replenish_beta], 'params ... d -> ... (params d)')
        else:
            raise NotImplementedError()

        var_latent_params = torch.concat([var_latent_params, replenish_params], dim=1)

        # repeat X to match tensor sizes
        X_r = repeat(X[:, 0, ...], 'b ... -> b r ...', r=num_prior_replenish_proposals)
        X = torch.concat([X, X_r], dim=1)

    # Sample latent variables
    var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)
    Z_imp = var_latent_distr.sample(sample_shape=(num_imp_samples,))

    # Shape (i b k d)

    # Create the distribution of the missingness model
    generator_params = model.generator_network(Z_imp)
    generator_distr = model.create_distribution(generator_params, model.hparams.generator_distribution)

    # Create prior distribution
    # NOTE: Assume standard prior
    # prior_distr = torch.distributions.Normal(loc=0., scale=1.)
    prior_distr = model.get_prior()

    # if num_prior_replenish_proposals > 0:
    #     # Remove so that there is only 1 prior proposal in the mixture
    #     var_latent_params = var_latent_params[:, :-(num_prior_replenish_proposals-1)]

    # Compute (unnormalised)-log-importance-weights
    if weighting == 'dmis':
        log_weights = compute_mis_log_unnormalised_importance_weights(X, M, Z_imp,
                                                                      miss_model=miss_model,
                                                                      var_latent_distr=var_latent_distr,
                                                                      var_comp_neg_idx=-2,
                                                                      prior_distr=prior_distr,
                                                                      generator_distr=generator_distr)
    elif weighting == 'dmis_within_groups':
        assert glr_cluster_size is not None and glr_cluster_size > 1
        log_weights = compute_groupeddmis_log_unnormalised_importance_weights(X, M, Z_imp,
                           model=model,
                           miss_model=miss_model,
                           var_latent_params=var_latent_params,
                           prior_distr=prior_distr,
                           generator_distr=generator_distr,
                           var_comp_neg_idx=-2,
                           cluster_size=glr_cluster_size,
                           pad_to=pad_glr_to)
    elif weighting == 'smis':
        log_weights = compute_log_unnormalised_importance_weights(X, M, Z_imp,
                                                                  miss_model=miss_model,
                                                                  var_latent_distr=var_latent_distr,
                                                                  prior_distr=prior_distr,
                                                                  generator_distr=generator_distr)
    else:
        raise NotImplementedError()

    transformed_log_weights = log_weights
    if weight_transform is not None:
        transformed_log_weights = weight_transform(log_weights=log_weights)

    if resampling_scheme == 'gr':
        mis_params, norm_log_weights = gr_resampling(transformed_log_weights, generator_params, resampling_method=resampling_method)
        latents_resampled = None
        if return_resampled_latents:
            latents_resampled, _ = gr_resampling(transformed_log_weights, Z_imp, resampling_method=resampling_method)
    elif resampling_scheme == 'lr':
        mis_params, norm_log_weights = lr_resampling(transformed_log_weights, generator_params, resampling_method=resampling_method)
        latents_resampled = None
        if return_resampled_latents:
            latents_resampled, _ = lr_resampling(transformed_log_weights, Z_imp, resampling_method=resampling_method)
    elif resampling_scheme == 'glr':
        assert glr_cluster_size is not None and glr_cluster_size > 1
        mis_params, norm_log_weights = glr_resampling(transformed_log_weights, generator_params,
                                    cluster_size=glr_cluster_size,
                                    pad_to=pad_glr_to,
                                    resampling_method=resampling_method)
        latents_resampled = None
        if return_resampled_latents:
            latents_resampled, _ = glr_resampling(transformed_log_weights, Z_imp,
                                    cluster_size=glr_cluster_size,
                                    pad_to=pad_glr_to,
                                    resampling_method=resampling_method)
    elif resampling_scheme == 'ir':
        mis_params, norm_log_weights = ir_resampling(transformed_log_weights, generator_params, resampling_method=resampling_method)
        latents_resampled = None
        if return_resampled_latents:
            latents_resampled, _ = ir_resampling(transformed_log_weights, Z_imp, resampling_method=resampling_method)
    else:
        raise NotImplementedError()

    #
    if num_prior_replenish_proposals > 0 or (num_historical_proposals is not None and num_historical_proposals > 0):
        # assert mis_params.shape[1] == K+num_prior_replenish_proposals+num_historical_proposals
        mis_params = mis_params[:, :K, ...] # Ideally this should be random but with Multinomial sampler it won't matter
        X = X[:, :K, ...]

    mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

    # Sample missing values
    X_m = mis_distr.sample()

    acceptance_x = torch.tensor(1.)
    if accept_reject_xm:
        # Set imputed missing values
        X_new =  X*M_expanded + X_m*M_not_expanded

        mis_log_prob_old = miss_model.log_prob(X, M)
        mis_log_prob_new = miss_model.log_prob(X_new, M)

        log_accept = mis_log_prob_new - mis_log_prob_old

        acceptance_prob = torch.exp(log_accept)

        acceptance_samples = torch.rand_like(acceptance_prob)
        accepted_x = acceptance_samples < acceptance_prob

        accepted_x_not = ~accepted_x
        X = X*accepted_x_not.unsqueeze(-1) + X_new*accepted_x.unsqueeze(-1)

        acceptance_x = accepted_x.float().mean()

    # Compute average ESS
    log_weights_reshaped = rearrange(log_weights, 'i b k -> b (i k)')
    log_norm_weights = log_weights_reshaped - torch.logsumexp(log_weights_reshaped, dim=-1, keepdim=True)
    ess = torch.exp(-torch.logsumexp(2 * log_norm_weights, -1))
    # avg_ess = ess.mean(dim=0).cpu()

    transformedlog_weights_reshaped = rearrange(transformed_log_weights, 'i b k -> b (i k)')
    transformed_log_norm_weights = transformedlog_weights_reshaped - torch.logsumexp(transformedlog_weights_reshaped, dim=-1, keepdim=True)
    transfomer_ess = torch.exp(-torch.logsumexp(2 * transformed_log_norm_weights, -1))
    # transformed_avg_ess = transfomer_ess.mean(dim=0).cpu()

    # Compute average perplexity
    perplexity = torch.exp(-torch.sum(log_norm_weights*torch.exp(log_norm_weights), dim=-1))

    # Set imputed missing values
    # if return_weights:
    if return_resampled_latents:
        return X*M_expanded + X_m*M_not_expanded, ess, transfomer_ess, perplexity, log_weights_reshaped.shape[-1], norm_log_weights , Z_imp, log_weights, acceptance_x, latents_resampled
    else:
        return X*M_expanded + X_m*M_not_expanded, ess, transfomer_ess, perplexity, log_weights_reshaped.shape[-1], norm_log_weights , Z_imp, log_weights, acceptance_x
    # else:
    #     return X*M_expanded + X_m*M_not_expanded, ess, transfomer_ess, perplexity, log_weights_reshaped.shape[-1], norm_log_weights , acceptance_x

def select_gen_params_for_ir(generator_params, idx, B, K):
        mis_params = rearrange(generator_params, 'i b k p -> b i k p')
        mis_params = mis_params.reshape(B, K, -1, generator_params.shape[-1])
        mis_params = rearrange(mis_params, 'b k2 i2 p -> i2 b k2 p')
        mis_params = mis_params[idx,
                                torch.arange(B, device=generator_params.device)[:, None],
                                torch.arange(K, device=generator_params.device),
                                ...]
        return mis_params

def importance_resampling_gibbs(model, X, M, num_imp_samples, num_iterations,
                                glr_cluster_size=None,
                                mode='dmis_gr', resampling_mode='multinomial',
                                metric = None, keep_last_n_iterations = 1, batchsize=-1,
                                num_prior_replenish_proposals = 0,
                                data_channel_dim=None):
    if mode.endswith('gr'):
        resampling_scheme = 'gr'
    elif mode.endswith('lr'):
        resampling_scheme = 'lr'
    elif mode.endswith('ir'):
        resampling_scheme = 'ir'
    else:
        raise NotImplementedError()

    if mode.startswith('dmis'):
        weighting = 'dmis'
    elif mode.startswith('smis'):
        weighting = 'smis'
    else:
        raise NotImplementedError()

    imp_seq = []
    stats = []
    stats_ess = []
    for t in tqdm.tqdm(range(num_iterations)):
        if batchsize <= 0:
            X, ess, _ = importance_resampling_gibbs_iteration(model, X, M, num_imp_samples,
                                                               glr_cluster_size=glr_cluster_size,
                                                               weighting=weighting,
                                                               resampling_method=resampling_mode,
                                                               resampling_scheme=resampling_scheme,
                                                               num_prior_replenish_proposals=num_prior_replenish_proposals,
                                                               data_channel_dim=data_channel_dim)
            avg_ess = ess.mean(dim=0)
        else:
            X_stack = []
            ess = []
            for b in range(math.ceil(X.shape[0] // batchsize)):
                X_b = X[b*batchsize:min((b+1)*batchsize, X.shape[0])]
                M_b = M[b*batchsize:min((b+1)*batchsize, M.shape[0])]
                X_b, ess_b, _ = importance_resampling_gibbs_iteration(model, X_b.float(), M_b, num_imp_samples,
                                                                   glr_cluster_size=glr_cluster_size,
                                                                   weighting=weighting,
                                                                   resampling_method=resampling_mode,
                                                                   resampling_scheme=resampling_scheme,
                                                                   num_prior_replenish_proposals=num_prior_replenish_proposals,
                                                                   data_channel_dim=data_channel_dim)
                ess_b = ess_b.mean(dim=0)
                X_stack.append(X_b.bool())
                ess.append(ess_b*X_b.shape[0])
            X = torch.vstack(X_stack)
            avg_ess = torch.sum(torch.stack(ess))/X.shape[0]

        if metric is not None:
            stats.append(metric(X))

        stats_ess.append(avg_ess)
        if t >= num_iterations-keep_last_n_iterations:
            imp_seq.append(X.cpu())

    return rearrange(imp_seq, 't ... -> t ...'), stats, stats_ess


def _systematic_sample(log_weights, *, num_dependent_samples=None, sample_shape=torch.Size()):
    # Systematic sampling preserves diversity better than multinomial sampling via Categorical(probs).sample().
    if not isinstance(sample_shape, torch.Size):
        sample_shape = torch.Size(sample_shape)

    if num_dependent_samples is None:
        num_dependent_samples = log_weights.size(-1)

    # Normalise weights to probabilities
    log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
    probs = torch.exp(log_weights)
    # NOTE: Fix rare numerical error case when probabilities dont sum to one
    # eventually causing an out-of-bounds index
    probs /= probs.sum(dim=-1, keepdim=True)

    # Compute the output shape
    batch_shape, size = probs.shape[:-1], probs.size(-1)
    extended_shape = sample_shape + batch_shape + (size,)
    probs = probs.expand(extended_shape)

    # Sample index systematically
    U = torch.rand(extended_shape[:-1] + (1,), device=log_weights.device)
    n = probs.cumsum(-1).mul_(num_dependent_samples).add_(U)
    n = n.floor_().clamp_(min=0, max=num_dependent_samples).long()
    diff = probs.new_zeros(extended_shape[:-1] + (num_dependent_samples + 1,), device=log_weights.device)
    diff.scatter_add_(-1, n, torch.ones_like(probs, device=log_weights.device))
    index = diff[..., :-1].cumsum(-1).long()

    # assert index.max() < log_weights.shape[-1]

    return index

# def _residual_sample(log_weights, *, num_samples=None, sample_shape=torch.Size()) -> torch.Tensor:
#     if not isinstance(sample_shape, torch.Size):
#         sample_shape = torch.Size(sample_shape)

#     # Normalise weights to probabilities
#     log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
#     probs = torch.exp(log_weights)
#     # NOTE: Fix rare numerical error case when probabilities dont sum to one
#     # eventually causing an out-of-bounds index
#     probs /= probs.sum(dim=-1, keepdim=True)

#     if num_samples is None:
#         num_samples = log_weights.size(-1)

#     # Compute the output shape
#     batch_shape, size = probs.shape[:-1], probs.size(-1)
#     extended_shape = sample_shape + batch_shape + (size,)
#     probs = probs.expand(extended_shape)

#     # Compute main quantities for residual sampling
#     m_probs = num_samples * probs
#     int_part = torch.floor(m_probs).long()
#     sum_int_part = torch.sum(int_part, dim=-1)
#     residuals = m_probs - int_part
#     sum_residuals = num_samples - sum_int_part

#     breakpoint()

#     # importance_distr = torch.distributions.Categorical(probs=residuals/sum_residuals)
#     # idx = importance_distr.sample(sample_shape=(X.shape[1],))
#     # torch.multinomial(residuals/sum_residuals)

#     out = torch.ones_like(w, dtype=torch.long)

#     numelems = floored.sum(-1)
#     res /= numelems

#     intpart = floored.long()
#     ranged = torch.arange(w.shape[-1], dtype=intpart.dtype, device=w.device) * out

#     modded = ranged.repeat_interleave(intpart)
#     aslong = numelems.long()

#     out[:aslong] = modded

#     if numelems == w.shape[-1]:
#         return out

#     out[aslong:] = torch.multinomial(res, w.shape[-1] - aslong, replacement=True)

#     return out

# def residual(W, M):
#     """Residual resampling.
#     """
#     N = W.shape[0]
#     A = np.empty(M, dtype=np.int64)
#     MW = M * W
#     intpart = np.floor(MW).astype(np.int64)
#     sip = np.sum(intpart)
#     res = MW - intpart
#     sres = M - sip
#     A[:sip] = np.arange(N).repeat(intpart)
#     # each particle n is repeated intpart[n] times
#     if sres > 0:
#         A[sip:] = multinomial(res / sres, M=sres)
#     return A

def standard_importance_resampling(model, X, M, num_imp_samples, *, glr_cluster_size=None,
                                   weighting='dmis', resampling_method='multinomial', resampling_scheme='gr',
                                   return_weights=False,
                                   weight_transform=None,
                                   data_channel_dim=None):
    K = X.shape[1]

    # Create latent distribution and sample
    var_latent_params = model.predict_var_latent_params(X, M)

    # Sample latent variables
    var_latent_distr = model.create_distribution(var_latent_params, model.hparams.var_latent_distribution)

    # # Replace with just the prior
    # assert isinstance(var_latent_distr, (torch.distributions.Normal, torch.distributions.Uniform))
    # # var_latent_distr = torch.distributions.Normal(torch.zeros_like(var_latent_distr.mean), torch.ones_like(var_latent_distr.stddev))
    var_latent_distr_temp = model.get_prior()
    if isinstance(var_latent_distr_temp, torch.distributions.Normal):
        shape = var_latent_distr.batch_shape
        var_latent_distr = torch.distributions.Normal(torch.ones(shape, device=X.device, dtype=X.dtype)*var_latent_distr_temp.mean,
                                                      torch.ones(shape, device=X.device, dtype=X.dtype)*var_latent_distr_temp.scale)
    elif isinstance(var_latent_distr_temp, torch.distributions.Uniform):
        shape = var_latent_distr.batch_shape
        var_latent_distr = torch.distributions.Uniform(low=torch.ones(shape, device=X.device, dtype=X.dtype)*var_latent_distr_temp.low,
                                                       high=torch.ones(shape, device=X.device, dtype=X.dtype)*var_latent_distr_temp.high)
    else:
        raise NotImplementedError()

    Z_imp = var_latent_distr.sample(sample_shape=(num_imp_samples,))

    # Shape (i b k d)

    # Create the distribution of the missingness model
    generator_params = model.generator_network(Z_imp)
    generator_distr = model.create_distribution(generator_params, model.hparams.generator_distribution)

    # Create prior distribution
    # NOTE: Assume standard prior
    # prior_distr = torch.distributions.Normal(loc=0., scale=1.)
    prior_distr = model.get_prior()

    # Compute (unnormalised)-log-importance-weights
    if weighting == 'dmis':
        log_weights = compute_mis_log_unnormalised_importance_weights(X, M, Z_imp,
                                                                    var_latent_distr=var_latent_distr,
                                                                    var_comp_neg_idx=-2,
                                                                    prior_distr=prior_distr,
                                                                    generator_distr=generator_distr)
    elif weighting == 'smis':
        log_weights = compute_log_unnormalised_importance_weights(X, M, Z_imp,
                                                                          var_latent_distr,
                                                                          prior_distr,
                                                                          generator_distr)
    else:
        raise NotImplementedError()

    transformed_log_weights = log_weights
    if weight_transform is not None:
        transformed_log_weights = weight_transform(log_weights=log_weights)

    if resampling_scheme == 'gr':
        mis_params, norm_log_weights = gr_resampling(transformed_log_weights, generator_params, resampling_method=resampling_method)
    elif resampling_scheme == 'lr':
        mis_params, norm_log_weights = lr_resampling(transformed_log_weights, generator_params, resampling_method=resampling_method)
    elif resampling_scheme == 'glr':
        assert glr_cluster_size is not None and glr_cluster_size > 1
        mis_params, norm_log_weights = glr_resampling(transformed_log_weights, generator_params, cluster_size=glr_cluster_size ,resampling_method=resampling_method)
    elif resampling_scheme == 'ir':
        mis_params, norm_log_weights = ir_resampling(transformed_log_weights, generator_params, resampling_method=resampling_method)
    else:
        raise NotImplementedError()

    mis_distr = model.create_distribution(mis_params, model.hparams.generator_distribution)

    # Sample missing values
    X_m = mis_distr.sample()

    # Compute average ESS
    log_weights_reshaped = rearrange(log_weights, 'i b k -> b (i k)')
    log_norm_weights = log_weights_reshaped - torch.logsumexp(log_weights_reshaped, dim=-1, keepdim=True)
    ess = torch.exp(-torch.logsumexp(2 * log_norm_weights, -1))
    # avg_ess = ess.mean(dim=0).cpu()

    transformedlog_weights_reshaped = rearrange(transformed_log_weights, 'i b k -> b (i k)')
    transformed_log_norm_weights = transformedlog_weights_reshaped - torch.logsumexp(transformedlog_weights_reshaped, dim=-1, keepdim=True)
    transfomer_ess = torch.exp(-torch.logsumexp(2 * transformed_log_norm_weights, -1))
    # transformed_avg_ess = transfomer_ess.mean(dim=0).cpu()

    # Compute average perplexity
    perplexity = torch.exp(-torch.sum(log_norm_weights*torch.exp(log_norm_weights), dim=-1))

    M_expanded = expand_M_dim(M, data_channel_dim=data_channel_dim)
    M_not_expanded = ~M_expanded
    # Set imputed missing values
    return X*M_expanded + X_m*M_not_expanded, ess, transfomer_ess, perplexity, log_weights_reshaped.shape[-1], norm_log_weights, Z_imp, log_weights



if __name__ == '__main__':

    def test_independent_resampling_i_equal_k():
        # Verify the indexing for Independent resampling of parameters
        # I = K
        B, P = 6, 4
        I, K = 5, 5
        X = torch.randn(I, B, K, P)
        idx = torch.tensor([[0, 1, 3, 2, 3],
                            [4, 1, 1, 1, 1],
                            [0, 0, 1, 2, 0],
                            [0, 1, 3, 1, 1],
                            [2, 2, 4, 2, 3],
                            [1, 0, 3, 2, 0]])

        mis_params = select_gen_params_for_ir(X, idx, B, K)

        for b in range(mis_params.shape[0]):
            mis_params_b = mis_params[b]
            for k in range(mis_params.shape[1]):
                mis_params_bk = mis_params_b[k]

                assert torch.allclose(X[k, b, idx[b,k]], mis_params_bk)

    test_independent_resampling_i_equal_k()

    def test_independent_resampling_i_notequal_k():
        # Verify the indexing for Independent resampling of parameters
        # I % K = 0
        B, P = 6, 4
        K = 5
        I = 2*K
        X = torch.randn(I, B, K, P)
        idx = torch.tensor([[0, 8, 5, 2, 9],
                            [4, 1, 1, 1, 1],
                            [0, 0, 1, 2, 5],
                            [0, 1, 3, 7, 1],
                            [2, 8, 4, 2, 3],
                            [1, 6, 6, 2, 0]])

        mis_params = select_gen_params_for_ir(X, idx, B, K)

        X_rearranged = rearrange(X, '(i i2) b k p -> i b (i2 k) p', i2=I//K)
        # NOTE:
        # This rearranges tensor([[1, 2, 3],
        #                         [4, 5, 6]])
        # into tensor([[1, 2, 3, 4, 5, 6]])

        for b in range(mis_params.shape[0]):
            mis_params_b = mis_params[b]
            for k in range(mis_params.shape[1]):
                mis_params_bk = mis_params_b[k]

                assert torch.allclose(X_rearranged[k, b, idx[b,k]], mis_params_bk)

    test_independent_resampling_i_notequal_k()

    def test_systematic_sample(num_dependent_samples=None):
        torch.manual_seed(10112)
        size = 20
        probs = torch.randn(size).exp()
        probs /= probs.sum()

        log_weights = torch.log(probs)

        num_samples = 20000
        index = _systematic_sample(log_weights, num_dependent_samples=num_dependent_samples, sample_shape=(num_samples,))
        histogram = torch.zeros_like(probs)
        histogram.scatter_add_(-1, index.reshape(-1),
                            probs.new_ones(1).expand(num_samples * size))

        expected = probs * (num_dependent_samples if num_dependent_samples is not None else size)
        actual = histogram / num_samples
        assert torch.allclose(actual, expected, atol=0.01)

    test_systematic_sample()
    test_systematic_sample(num_dependent_samples=5)
    test_systematic_sample(num_dependent_samples=1)

    # def test_residual_sample():
    #     torch.manual_seed(10112)
    #     size = 20
    #     probs = torch.randn(size).exp()
    #     probs /= probs.sum()

    #     log_weights = torch.log(probs)

    #     num_samples = 20000
    #     index = _residual_sample(log_weights, num_samples=None, sample_shape=(num_samples,))
    #     histogram = torch.zeros_like(probs)
    #     histogram.scatter_add_(-1, index.reshape(-1),
    #                         probs.new_ones(1).expand(num_samples * size))

    #     expected = probs * size
    #     actual = histogram / num_samples
    #     assert torch.allclose(actual, expected, atol=0.01)

    # test_residual_sample()
