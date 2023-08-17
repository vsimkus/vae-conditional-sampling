import torch
from einops import reduce


def compute_log_unnormalised_importance_weights(X: torch.Tensor,
                                                M: torch.Tensor,
                                                Z: torch.Tensor,
                                                var_latent_distr: torch.distributions.Distribution,
                                                prior_distr: torch.distributions.Distribution,
                                                generator_distr: torch.distributions.Distribution,
                                                miss_model: torch.nn.Module = None):
    # Compute cross-entropy term of observed data
    generator_logprob = generator_distr.log_prob(X)*M
    generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

    # Compute prior latent probability
    prior_logprob = prior_distr.log_prob(Z)
    prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

    # Compute the log-prob of samples under the latent distribution
    latent_logprob = var_latent_distr.log_prob(Z)
    latent_logprob = reduce(latent_logprob, '... d -> ...', 'sum')

    # Compute the total IWELBO weights
    log_weights = generator_logprob + prior_logprob - latent_logprob

    # If Missingness model is given - compute probability of missingness pattern (for MNAR data)
    if miss_model is not None:
        X_m_for_miss_model = generator_distr.sample()
        X_for_miss_model = X*M + X_m_for_miss_model*~M
        log_prob_M = miss_model.log_prob(X_for_miss_model, M)

        log_weights += log_prob_M

    return log_weights

def compute_mis_log_unnormalised_importance_weights(X: torch.Tensor,
                                                    M: torch.Tensor,
                                                    Z: torch.Tensor,
                                                    var_latent_distr: torch.distributions.Distribution,
                                                    var_comp_neg_idx: int, # should be negative index
                                                    prior_distr: torch.distributions.Distribution,
                                                    generator_distr: torch.distributions.Distribution,
                                                    miss_model: torch.nn.Module = None):
    # Compute cross-entropy term of observed data
    generator_logprob = generator_distr.log_prob(X)*M
    generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

    # Compute prior latent probability
    prior_logprob = prior_distr.log_prob(Z)
    prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

    # Compute the log-prob of samples under the mixture latent distribution
    latent_logprob = multiple_importance_logprob(Z, var_latent_distr, var_comp_neg_idx=var_comp_neg_idx)

    # Compute the total IWELBO weights
    log_weights = generator_logprob + prior_logprob - latent_logprob

    # If Missingness model is given - compute probability of missingness pattern (for MNAR data)
    if miss_model is not None:
        X_m_for_miss_model = generator_distr.sample()
        X_for_miss_model = X*M + X_m_for_miss_model*~M
        log_prob_M = miss_model.log_prob(X_for_miss_model, M)

        log_weights += log_prob_M

    return log_weights

def multiple_importance_logprob(Z: torch.Tensor,
                                var_latent_distr: torch.distributions.Distribution,
                                var_comp_neg_idx: int): # should be negative index
    # Compute the log-prob of samples under the mixture var distribution
    # Z_aug = rearrange(Z, 'z i b k d -> k z i b 1 d')
    Z_aug = torch.swapaxes(torch.unsqueeze(Z, 0), 0, var_comp_neg_idx)

    latent_logprob = var_latent_distr.log_prob(Z_aug)
    latent_logprob = reduce(latent_logprob, '... d -> ...', 'sum')
    latent_logprob = torch.logsumexp(latent_logprob, dim=var_comp_neg_idx+1, keepdim=True)
    latent_logprob -= torch.log(torch.tensor(Z.shape[var_comp_neg_idx], device=Z.device))

    # Undo the swap axes above
    # latent_logprob = rearrange(latent_logprob, 'k ... -> ... k')
    latent_logprob = torch.swapaxes(latent_logprob, 0, var_comp_neg_idx+1).squeeze(0)

    return latent_logprob


if __name__ == '__main__':
    # Check that the MIS weight estimator is correct
    B, I, K, D, NZ, DZ = 7, 6, 5, 4, 3, 2
    X = torch.randn((B, K, D))
    Z = torch.randn((NZ, I, B, K, DZ))

    gen_mean = torch.randn((NZ, I, B, K, D))
    gen_scale = (torch.randn((NZ, I, B, K, D))*3)**2

    var_mean = torch.randn((B, K, DZ))
    var_scale = (torch.randn((B, K, DZ))*2)**2

    var_comp_neg_idx = -2

    prior_distr = torch.distributions.Normal(0, 1)

    log_weights = compute_mis_log_unnormalised_importance_weights(X, torch.ones_like(X), Z,
                                                                  var_latent_distr=torch.distributions.Normal(var_mean, var_scale),
                                                                  var_comp_neg_idx=var_comp_neg_idx,
                                                                  prior_distr=prior_distr,
                                                                  generator_distr=torch.distributions.Normal(gen_mean, gen_scale))

    for z in range(Z.shape[0]):
        for i in range(Z.shape[1]):
            for b in range(Z.shape[2]):
                for k in range(Z.shape[3]):
                    X_temp = X[b, k, :]
                    Z_temp = Z[z, i, b, k, :]

                    gen_distr = torch.distributions.Normal(gen_mean[z, i, b, k, :], gen_scale[z, i, b, k, :])
                    generator_logprob = gen_distr.log_prob(X_temp)
                    generator_logprob = reduce(generator_logprob, '... d -> ...', 'sum')

                    prior_logprob = prior_distr.log_prob(Z_temp)
                    prior_logprob = reduce(prior_logprob, '... d -> ...', 'sum')

                    latent_logprobs = []
                    for k2 in range(Z.shape[var_comp_neg_idx]):
                        var_latent_distr = torch.distributions.Normal(var_mean[b, k2, :], var_scale[b, k2, :])
                        latent_logprob = var_latent_distr.log_prob(Z_temp)
                        latent_logprob = reduce(latent_logprob, '... d -> ...', 'sum')
                        latent_logprobs.append(latent_logprob)
                    latent_logprob = torch.logsumexp(torch.tensor(latent_logprobs), dim=0) - torch.log(torch.tensor(K))

                    log_weight = generator_logprob + prior_logprob - latent_logprob

                    assert log_weights[z, i, b, k] == log_weight
