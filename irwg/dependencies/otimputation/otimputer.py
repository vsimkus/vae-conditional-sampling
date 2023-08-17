from collections import defaultdict

import numpy as np
import torch

from geomloss import SamplesLoss
from einops import asnumpy

from irwg.sampling.imputation_metrics import imputation_rmse_metric, imputation_mae_metric

# Adapted from https://github1s.com/BorisMuzellec/MissingDataOT/blob/master/utils.py

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult

class OTimputer():
    """
    'One parameter equals one imputed value' model (Algorithm 1. in the paper)

    Parameters
    ----------

    eps: float, default=0.01
        Sinkhorn regularization parameter.

    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.

    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"


    """
    def __init__(self,
                 eps=None,
                 lr=1e-2,
                 opt=torch.optim.RMSprop,
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=.9):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.scaling = scaling
        self.sk = None

    def fit_transform(self, X, mask, verbose=True, X_true=None, X_fullref=None):
        """
        Imputes missing values using a batched OT loss

        Parameters
        ----------
        X : torch.FloatTensor or torch.cuda.FloatTensor
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned
            (e.g. with NaNs).

        mask : torch.FloatTensor or torch.cuda.FloatTensor
            mask[i,j] == 0 if X[i,j] is missing, else mask[i,j] == 1.

        verbose: bool, default=True
            If True, output loss to log during iterations.

        X_true: torch.FloatTensor or None, default=None
            Ground truth for the missing values. If provided, will output a
            validation score during training, and return score arrays.
            For validation/debugging only.

        X_fullref: torch.FloatTensor or None, default=None
            If provided it is used as a reference distribution instead of the incomplete data from X.

        Returns
        -------
        X_filled: torch.FloatTensor or torch.cuda.FloatTensor
            Imputed missing data (plus unchanged non-missing data).


        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(X_true, np.ndarray):
            X_true = torch.from_numpy(X_true)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if X_fullref is not None and isinstance(X_fullref, np.ndarray):
            X_fullref = torch.from_numpy(X_fullref)

        X = X.clone()
        n, d = X.shape

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                print(f"Batchsize larger that half size = {n // 2}. Setting batchsize to {self.batchsize}.")

        mask = (~mask)
        X[mask] = float('nan')

        if self.eps is None:
            # Pick epsilon automatically
            eps_quantile = 0.5
            eps_quantile_multiplier = 0.05
            self.eps = pick_epsilon(X, eps_quantile, eps_quantile_multiplier)
            print(f"epsilon: {self.eps:.4f} "
                  f"({100 * eps_quantile}th percentile times "
                  f"{eps_quantile_multiplier})")
        self.sk = SamplesLoss("sinkhorn", p=2, blur=self.eps, scaling=self.scaling, backend="tensorized")

        imps = (self.noise * torch.randn(mask.shape) + nanmean(X, 0))[mask]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if verbose:
            print(f"batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        if X_true is not None:
            metrics = defaultdict(list)

        all_imputations = []
        for i in range(self.niter):

            X_filled = X.detach().clone()
            X_filled[mask] = imps
            all_imputations.append(X_filled.detach().clone())
            loss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batchsize, replace=False)
                X1 = X_filled[idx1]

                if X_fullref is None:
                    idx2 = np.random.choice(n, self.batchsize, replace=False)
                    X2 = X_filled[idx2]
                else:
                    idx2 = np.random.choice(X_fullref.shape[0], self.batchsize, replace=False)
                    X2 = X_fullref[idx2]

                loss = loss + self.sk(X1, X2)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                print("Nan or inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if X_true is not None:
                rmse = imputation_rmse_metric(X_imp=X_filled.unsqueeze(1), X_true=X_true.unsqueeze(1), M=(~mask.bool()).unsqueeze(1))
                metrics['rmse'].append(asnumpy(rmse))

                mae = imputation_mae_metric(X_imp=X_filled.unsqueeze(1), X_true=X_true.unsqueeze(1), M=(~mask.bool()).unsqueeze(1))
                metrics['mae'].append(asnumpy(mae))

            # if verbose and (i % report_interval == 0):
            #     if X_true is not None:
            #         print(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}\t '
            #               f'Validation MAE: {maes[i]:.4f}\t'
            #               f'RMSE: {rmses[i]:.4f}')
            #     else:
            #         print(f'Iteration {i}:\t Loss: {loss.item() / self.n_pairs:.4f}')

        X_filled = X.detach().clone()
        X_filled[mask] = imps
        all_imputations.append(X_filled.detach())

        if X_true is not None:
            return X_filled, all_imputations, metrics
        else:
            return X_filled, all_imputations
