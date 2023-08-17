
import math
import re
from einops import rearrange, repeat, reduce

import numpy as np
import torch
import torchmetrics
from tqdm import tqdm


def imputation_f1_metric(X_imp, X_true, M):#, quantiles=[0.5]):
    X_true = X_true.bool()
    X_imp = X_imp.bool()
    M_not = ~M

    # true_pos = ((X_true & X_imp)*M_not).sum()
    # false_pos = ((~X_true & X_imp)*M_not).sum()
    # false_neg = ((X_true & ~X_imp)*M_not).sum()

    # f1 = true_pos / (true_pos + 0.5*(false_pos + false_neg))

    true_pos = reduce((X_true & X_imp)*M_not, 'b k ... -> b k', 'sum')
    false_pos = reduce((~X_true & X_imp)*M_not, 'b k ... -> b k', 'sum')
    false_neg = reduce((X_true & ~X_imp)*M_not, 'b k ... -> b k', 'sum')

    f1 = true_pos / (true_pos + 0.5*(false_pos + false_neg))

    # f1 = f1[torch.isfinite(f1).all(dim=-1)]
    f1[~torch.isfinite(f1)] = 0.

    # f1_std = f1.std(unbiased=True)
    # f1_mean = f1.mean()
    # f1_quantiles = torch.quantile(f1, q=torch.tensor(quantiles, device=X_imp.device))

    return f1 #, f1_mean #, f1_std, f1_quantiles

def imputation_inv_f1_metric(X_imp, X_true, M):#, quantiles=[0.5]):
    X_true = ~(X_true.bool())
    X_imp = ~(X_imp.bool())
    M_not = ~M

    # true_pos = ((X_true & X_imp)*M_not).sum()
    # false_pos = ((~X_true & X_imp)*M_not).sum()
    # false_neg = ((X_true & ~X_imp)*M_not).sum()

    # f1 = true_pos / (true_pos + 0.5*(false_pos + false_neg))

    true_pos = reduce((X_true & X_imp)*M_not, 'b k ... -> b k', 'sum')
    false_pos = reduce((~X_true & X_imp)*M_not, 'b k ... -> b k', 'sum')
    false_neg = reduce((X_true & ~X_imp)*M_not, 'b k ... -> b k', 'sum')

    f1 = true_pos / (true_pos + 0.5*(false_pos + false_neg))

    # f1 = f1[torch.isfinite(f1).all(dim=-1)]
    f1[~torch.isfinite(f1)] = 0.

    # f1_std = f1.std(unbiased=True)
    # f1_mean = f1.mean()
    # f1_quantiles = torch.quantile(f1, q=torch.tensor(quantiles, device=X_imp.device))

    return f1 #, f1_mean, f1_std, f1_quantiles

def imputation_bin_accuracy_metric(X_imp, X_true, M):#, quantiles=[0.5]):
    X_true = X_true.bool()
    X_imp = X_imp.bool()
    M_not = ~M

    # acc = ((X_true == X_imp)*M_not).sum()/(M_not.sum()*X_imp.shape[1])

    correct = reduce((X_true == X_imp)*M_not, 'b k ... -> b k', 'sum')
    count = reduce(M_not, 'b 1 ... -> b 1', 'sum') #*X_imp.shape[1]
    count[count == 0] = 1
    acc = correct / count

    # acc = acc[torch.isfinite(acc).all(dim=-1)]

    # acc_mean = acc.mean()
    # acc_std = acc.std(unbiased=True)
    # acc_quantiles = torch.quantile(acc, q=torch.tensor(quantiles, device=X_imp.device))

    return acc #, acc_mean, acc_std, acc_quantiles


def imputation_rmse_metric(X_imp, X_true, M):#, quantiles=[0.5]):
    M_not = ~M

    # mse = (((X_imp - X_true)**2)*M_not).sum(-1)/(M_not.sum(-1))

    se = reduce(((X_imp - X_true)**2)*M_not, 'b k ... -> b k', 'sum')
    count = reduce(M_not, 'b 1 ... -> b 1', 'sum')
    count[count == 0] = 1
    rmse = (se / count)**0.5
    # rmse = rmse.mean(-1)

    # rmse = rmse[torch.isfinite(rmse).all(dim=-1)]

    # rmse_mean = rmse.mean()
    # rmse_std = rmse.std(unbiased=True)
    # rmse_quantiles = torch.quantile(rmse, q=torch.tensor(quantiles, device=X_imp.device))

    return rmse #, rmse_mean, rmse_std, rmse_quantiles

def imputation_rmse_metric_batched(X_imp, X_true, M, *, batch_size):#, quantiles=[0.5]):
    M_not = ~M

    # mse = (((X_imp - X_true)**2)*M_not).sum(-1)/(M_not.sum(-1))

    rmses = []
    for i in range(math.ceil(len(X_imp)/batch_size)):
        X_imp_i = X_imp[i*batch_size:min((i+1)*batch_size, len(X_imp))]
        X_true_i = X_true[i*batch_size:min((i+1)*batch_size, len(X_imp))]
        M_not_i = M_not[i*batch_size:min((i+1)*batch_size, len(X_imp))]

        se = reduce(((X_imp_i - X_true_i)**2)*M_not_i, 'b k ... -> b k', 'sum')
        count = reduce(M_not_i, 'b 1 ... -> b 1', 'sum')
        count[count == 0] = 1
        rmse = (se / count)**0.5
        rmses.append(rmse)

    rmse = np.concatenate(rmses, axis=0)

    # rmse = rmse.mean(-1)

    # rmse = rmse[torch.isfinite(rmse).all(dim=-1)]

    # rmse_mean = rmse.mean()
    # rmse_std = rmse.std(unbiased=True)
    # rmse_quantiles = torch.quantile(rmse, q=torch.tensor(quantiles, device=X_imp.device))

    return rmse #, rmse_mean, rmse_std, rmse_quantiles

def imputation_mae_metric(X_imp, X_true, M):
    M_not = ~M

    ae = reduce(torch.absolute((X_imp - X_true))*M_not, 'b k ... -> b k', 'sum')
    count = reduce(M_not, 'b 1 ... -> b 1', 'sum')
    count[count == 0] = 1
    mae = (ae / count)

    # mae = mae[torch.isfinite(mae).all(dim=-1)]

    return mae

def imputation_mae_metric_np(X_imp, X_true, M):
    M_not = ~M

    ae = reduce(np.absolute((X_imp - X_true))*M_not, 'b k ... -> b k', 'sum')
    count = reduce(M_not, 'b 1 ... -> b 1', 'sum')
    count[count == 0] = 1
    mae = (ae / count)

    # mae = mae[torch.isfinite(mae).all(dim=-1)]

    return mae

def imputation_mae_metric_batched_np(X_imp, X_true, M, *, batch_size):
    M_not = ~M

    aes = []
    for i in range(math.ceil(len(X_imp)/batch_size)):
        X_imp_i = X_imp[i*batch_size:min((i+1)*batch_size, len(X_imp))]
        X_true_i = X_true[i*batch_size:min((i+1)*batch_size, len(X_imp))]
        M_not_i = M_not[i*batch_size:min((i+1)*batch_size, len(X_imp))]

        ae = reduce(np.absolute((X_imp_i - X_true_i))*M_not_i, 'b k ... -> b k', 'sum')
        aes.append(ae)

    ae = np.concatenate(aes, axis=0)
    count = reduce(M_not, 'b 1 ... -> b 1', 'sum')
    count[count == 0] = 1
    mae = (ae / count)

    # mae = mae[torch.isfinite(mae).all(dim=-1)]

    return mae

def imputation_ssim_metric(X_imp, X_true, img_shape=(1, 28,28), data_range=1.0):#, quantiles=[0.5]):
    B, K = X_imp.shape[0], X_imp.shape[1]
    X_imp = rearrange(X_imp, 'b k ... -> (b k) ...')
    X_true = repeat(X_true, 'b 1 ... -> (b k) ...', k=K)

    X_imp = X_imp.reshape(len(X_imp), *img_shape)
    X_true = X_true.reshape(len(X_true), *img_shape)

    ssim = torchmetrics.functional.structural_similarity_index_measure(X_imp, X_true,
                                                                       gaussian_kernel=True,
                                                                       reduction='none',
                                                                       data_range=data_range)

    ssim = rearrange(ssim, '(b k) -> b k', b=B, k=K)
    # ssim = ssim.mean(dim=-1)

    # ssim_mean = ssim.mean()
    # ssim_std = ssim.std(unbiased=True)
    # ssim_quantiles = torch.quantile(ssim, q=torch.tensor(quantiles, device=X_imp.device))

    return ssim #, ssim_mean, ssim_std, ssim_quantiles

def imputation_latent_wass(X_imp, X_true, vae):#, quantiles=[0.5]):
    imp_var = vae.compute_var_distr(X_imp, torch.ones_like(X_imp))
    true_var = vae.compute_var_distr(X_true, torch.ones_like(X_true))

    W = torch.sum((imp_var.mean - true_var.mean)**2, dim=-1)#**0.5**2
    W += torch.sum(imp_var.variance + true_var.variance - 2*(imp_var.variance*true_var.variance)**0.5, dim=-1)

    # W = W.mean(dim=-1)

    # W_mean = W.mean()
    # W_std = W.std(unbiased=True)
    # W_quantiles = torch.quantile(W, q=torch.tensor(quantiles, device=X_imp.device))

    return W #, W_mean, W_std, W_quantiles

#
# MMD computation adapted from https://github.com/wgrathwohl/GWG_release/blob/HEAD/mmd.py#L1-L112
#

def assert_shape(x, s):
    assert x.size() == s


def pairwise_avg_hamming(x):
    # diffs = (x[None, :] != y[:, None, :]).float().mean(-1)
    # return diffs

    # p=0 computes Hamming distance
    diffs = torch.nn.functional.pdist(x, p=0) / x.shape[-1]
    # Returns upper triangular portion of the pairwise distances

    # Diagonal is always zero here, so initialise to zeros and replace else with diffs
    K = torch.zeros(x.shape[0], x.shape[0], device=x.device)

    idx_u = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    # idx_l = torch.tril_indices(x.shape[0], x.shape[0], offset=-1, device=x.device)
    K[idx_u[0], idx_u[1]] = diffs
    # K[idx_l] = K.T[idx_l] # Copy upper triangle to bottom triangle
    K = K + K.T # Copy upper triangle to bottom triangle, cheaper than above

    return K


def pairwise_exp_avg_hamming(x):
    diffs = pairwise_avg_hamming(x)
    return (-diffs).exp()


def pairwise_scaled_exp_avg_hamming(x, s):
    diffs = pairwise_avg_hamming(x) * s
    return (-diffs).exp()

def pairwise_rbf_kernel(x, h):
    diffs = torch.nn.functional.pdist(x, p=2)
    # Returns upper triangular portion of the pairwise distances

    # Diagonal is always zero here, so initialise to zeros and replace else with diffs
    K = torch.zeros(x.shape[0], x.shape[0], device=x.device)

    idx_u = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    # idx_l = torch.tril_indices(x.shape[0], x.shape[0], offset=-1, device=x.device)
    K[idx_u[0], idx_u[1]] = diffs
    # K[idx_l] = K.T[idx_l] # Copy upper triangle to bottom triangle
    K = K + K.T # Copy upper triangle to bottom triangle, cheaper than above

    return torch.exp(-(K**2)/(2*h**2))

def pairwise_poly_kernel(x, degree, gamma, coeff=1):
    K = ((x.unsqueeze(-2) @ x.unsqueeze(-1))*gamma + coeff)**degree

    return K


def pairwise_exp_ssim(x, batch_size=150000, img_shape=(28,28), channels=1):
    N = x.shape[0]

    pair_idxs = torch.combinations(torch.arange(N), with_replacement=False)

    ssim = torch.zeros(len(pair_idxs))
    for t in tqdm(range(math.ceil(len(pair_idxs)/batch_size)), desc='Computing Pairwise SSIM'):
        pair_idxs_t = pair_idxs[t*batch_size:min(len(pair_idxs), (t+1)*batch_size)]

        x_pairs_t = x[pair_idxs_t]

        X1, X2 = x_pairs_t[:, 0, ...].reshape(len(x_pairs_t), channels, *img_shape), x_pairs_t[:, 1, ...].reshape(len(x_pairs_t), channels, *img_shape)

        ssim[t*batch_size:min(len(pair_idxs), (t+1)*batch_size)] = \
            torchmetrics.functional.structural_similarity_index_measure(X1, X2,
                                                                        gaussian_kernel=True,
                                                                        reduction='none',
                                                                        data_range=1.0)

    # Diagonal is always one here, so initialise to ones and replace else with ssim
    K = torch.ones(x.shape[0], x.shape[0], device=x.device)

    K[pair_idxs[:, 0], pair_idxs[:, 1]] = ssim
    K = K * K.T # Copy upper triangle to bottom triangle

    return torch.exp(K)



class MMD(object):
    """
    Quadratic-time maximum mean discrepancy (MMD) test.
    Use the unbiased U-statistic.
    """
    def __init__(self, kernel_fun, use_ustat=False):
        """
        Args:
            kernel: function, kernel function.
            use_ustat: boolean, whether to compute U-statistic or V-statistic.
        """
        assert callable(kernel_fun)

        self.kernel = kernel_fun
        self.use_ustat = use_ustat

    def compute_gram(self, x, y):
        """
        Compute Gram matrices:
            K: array((m+n, m+n))
            kxx: array((m, m))
            kyy: array((n, n))
            kxy: array((m, n))
        """
        # NOTE: could be more efficient by not constructing the full kernel matrix (it is symmetric)

        (m, d1) = x.shape
        (n, d2) = y.shape
        assert d1 == d2

        xy = torch.cat([x, y], 0) #np.vstack([x, y])
        K = self.kernel(xy)  # kxyxy
        assert_shape(K, (m+n, m+n))
        #assert is_psd(K)  # TODO: Remove check

        kxx = K[:m, :m]
        assert_shape(kxx, (m, m))
        # assert is_psd(kxx)
        #assert is_symmetric(kxx)

        kyy = K[m:, m:]
        assert_shape(kyy, (n, n))
        # assert is_psd(kyy)
        #assert is_symmetric(kyy)

        kxy = K[:m, m:]
        assert_shape(kxy, (m, n))

        return K, kxx, kyy, kxy

    def compute_statistic(self, kxx, kyy, kxy):
        """
        Compute MMD test statistic.
        """
        m = kxx.size(0)
        n = kyy.size(0)
        assert_shape(kxx, (m, m))
        assert_shape(kyy, (n, n))
        assert_shape(kxy, (m, n))

        if self.use_ustat:  # Compute U-statistics estimate
            term_xx = (kxx.sum() - torch.diag(kxx).sum()) / (m*(m-1))
            term_yy = (kyy.sum() - torch.diag(kyy).sum()) / (n*(n-1))
            term_xy = kxy.sum() / (m*n)

        else:  # Compute V-statistics estimate
            term_xx = kxx.sum() / (m**2)
            term_yy = kyy.sum() / (n**2)
            term_xy = kxy.sum() / (m*n)

        res = term_xx + term_yy - 2*term_xy

        # NOTE: need to take square root.
        return res

    def compute_mmd(self, x, y):
        _, kxx, kyy, kxy = self.compute_gram(x, y)
        stat = self.compute_statistic(kxx, kyy, kxy)
        return stat


def mmd_metric(X_imp, X_ref, kernel=None):
    if kernel == 'rbf':
        kernel_fn = lambda x : pairwise_rbf_kernel(x, h=3)
    elif kernel == 'exp_avg_hamming':
        kernel_fn = pairwise_exp_avg_hamming
    elif kernel.startswith('exp_ssim'):
        m = re.search('(?<=_c)\d+(?=_)', kernel)
        chann = int(m.group(0))
        m = re.search('(?<=_img)\d+x\d+(?=_)', kernel)
        img_size = tuple(int(d) for d in m.group(0).split('x'))
        m = re.search('(?<=_b)\d+(?=$)', kernel)
        batch_size = int(m.group(0))

        kernel_fn = lambda x: pairwise_exp_ssim(x, img_shape=img_size, channels=chann, batch_size=batch_size)
    elif kernel.startswith('poly'):
        kernel_fn = lambda x: pairwise_poly_kernel(x, degree=3, gamma=(1/50), coeff=1)
    else:
        raise NotImplementedError()

    mmd = MMD(kernel_fun=kernel_fn, use_ustat=False)
    # mmd = MMD(kernel_fun=kernel_fn, use_ustat=True)
    mmd = mmd.compute_mmd(X_imp, X_ref)
    return mmd



def exp_avg_hamming_kernel(x,y):
    return torch.exp(-(x != y).float().mean(-1))

def poly_kernel(x, y, degree, gamma, coeff):
    return (((x.unsqueeze(-2) @ y.unsqueeze(-1))*gamma + coeff)**degree).squeeze(-1).squeeze(-1)

class MMDLinearTime(object):
    """
    Linear-time maximum mean discrepancy (MMD) test.
    """
    def __init__(self, kernel_fun):
        """
        Args:
            kernel: function, kernel function.
        """
        assert callable(kernel_fun)

        self.kernel = kernel_fun

    def compute_mmd(self, x, y):
        (m, d1) = x.shape
        (n, d2) = y.shape
        assert d1 == d2

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y)

        if m > n:
            k = math.ceil(m / n)
            y = torch.repeat_interleave(y, k, dim=0)[:m]
        elif n < m:
            k = math.ceil(n / m)
            x = torch.repeat_interleave(x, k, dim=0)[:n]

        y = y[torch.randperm(y.shape[0])]
        x = x[torch.randperm(x.shape[0])]

        x1, x2 = x[0::2], x[1::2]
        y1, y2 = y[0::2], y[1::2]

        term_xx = self.kernel(x1, x2).mean()
        term_yy = self.kernel(y1, y2).mean()
        term_xy = (self.kernel(x1, y2) + self.kernel(x2, y1)).mean()

        res = term_xx + term_yy - term_xy

        # NOTE: need to take square root
        return res

def mmd_linear_time_metric(X_imp, X_ref, kernel=None):
    if kernel == 'exp_avg_hamming':
        kernel_fn = exp_avg_hamming_kernel
    elif kernel == 'poly':
        kernel_fn = lambda x, y: poly_kernel(x, y, degree=3, gamma=(1/50), coeff=1)
    else:
        raise NotImplementedError()

    mmd = MMDLinearTime(kernel_fun=kernel_fn)
    mmd = mmd.compute_mmd(X_imp, X_ref)
    return mmd
