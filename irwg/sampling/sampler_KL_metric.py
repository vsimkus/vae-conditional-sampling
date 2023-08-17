import math

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from irwg.utils.test_step_iwelbo import compute_log_p_xm_given_xo_with_iwelbo


def pairwise_exp_hamming(X, M):
    M_not = ~M
    # We only want to compare the missing dims
    X = (X*M_not).float()

    # diffs = torch.cdist(X, X, p=0) / torch.sum(M_not, dim=-1).unsqueeze(-1)
    diffs = torch.cdist(X, X, p=0)

    # don't exp here, we exp later in KDE class
    return -diffs

class KDE(object):
    def __init__(self, kernel_fn):
        self.kernel = kernel_fn

    def compute_log_prob(self, X, M):
        K = self.kernel(X, M)

        # Set diagonal kernel values to neg inf
        diag_idx = torch.arange(K.shape[1])
        K[:, diag_idx, diag_idx] = float('-inf')

        logprobs = torch.logsumexp(K, dim=-1) - torch.log(torch.tensor(K.shape[-1]-1))

        # norm_logprobs = logprobs - torch.logsumexp(logprobs, dim=-1, keepdim=True)
        # breakpoint()
        # return norm_logprobs

        return logprobs

def compute_KL_div(model, X, M, *, num_importance_samples, kde_kernel, iwelbo_batchsize=1):
    if kde_kernel == 'exp_hamming':
        kernel_fn = pairwise_exp_hamming
    else:
        raise NotImplementedError()
    kde = KDE(kernel_fn=kernel_fn)

    log_q_xm_given_xo_kde = kde.compute_log_prob(X, M)

    B, K = X.shape[0], X.shape[1]
    X = rearrange(X, 'b k ... -> (b k) 1 ...')
    M = repeat(M.bool(), 'b 1 ... -> (b k) 1 ...', k=K)
    log_p_xm_given_xo_iwelbo = torch.zeros(X.shape[0], device=X.device)
    print('Estimating with IWELBO')
    for t in tqdm(range(math.ceil(len(X) / iwelbo_batchsize)), desc='Estimating with IWELBO'):
        X_t = X[t*iwelbo_batchsize:min(len(X), (t+1)*iwelbo_batchsize)].to(model.device).float()
        M_t = M[t*iwelbo_batchsize:min(len(X), (t+1)*iwelbo_batchsize)].to(model.device)

        log_p_xm_given_xo_iwelbo[t*iwelbo_batchsize:min(len(X), (t+1)*iwelbo_batchsize)] = \
            compute_log_p_xm_given_xo_with_iwelbo(model, X_t, M_t, num_importance_samples=num_importance_samples).squeeze(-1)

    log_p_xm_given_xo_iwelbo = rearrange(log_p_xm_given_xo_iwelbo, '(b k) -> b k', k=K, b=B).cpu()

    kl = log_q_xm_given_xo_kde.cpu() - log_p_xm_given_xo_iwelbo

    return kl.mean()


