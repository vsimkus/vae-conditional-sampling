from collections import defaultdict

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from einops import asnumpy

from irwg.sampling.imputation_metrics import imputation_rmse_metric, imputation_mae_metric

# Adapted from https://github1s.com/BorisMuzellec/MissingDataOT/blob/master/softimpute.py

F32PREC = np.finfo(np.float32).eps

class SoftImputeImputer():
    def __init__(self, maxit=1000):
        self.maxit = maxit

    def fit_transform(self, X, mask, X_true=None, X_fullref=None):

        mask = (~mask)
        X = np.copy(X)
        X[mask] = float('nan')

        cv_error, grid_lambda = cv_softimpute(X, grid_len=15, X_fullref=X_fullref)
        lbda = grid_lambda[np.argmin(cv_error)]

        if X_true is not None:
            _, imp, all_imputations, metrics = softimpute((X), lbda, maxit=self.maxit, X_true=X_true, X_fullref=X_fullref)
            return imp, all_imputations, metrics
        else:
            _, imp, all_imputations = softimpute((X), lbda, maxit=self.maxit, X_true=X_true, X_fullref=X_fullref)
            return imp, all_imputations
# convergence criterion for softimpute
def converged(x_old, x, mask, thresh):
    x_old_na = x_old[mask]
    x_na = x[mask]
    rmse = np.sqrt(np.sum((x_old_na - x_na) ** 2))
    denom = np.sqrt((x_old_na ** 2).sum())

    if denom == 0 or (denom < F32PREC and rmse > F32PREC):
      return False
    else:
      return (rmse / denom) < thresh


def softimpute(x, lamb, maxit = 1000, thresh = 1e-5, X_true=None, X_fullref=None):
    """
    x should have nan values (the mask is not provided as an argument)
    """
    mask = ~np.isnan(x)
    imp = x.copy()
    imp[~mask] = 0

    if X_fullref is not None:
        imp = np.concatenate([imp, X_fullref], axis=0)
        mask = np.concatenate([mask, np.ones_like(X_fullref, dtype=bool)], axis=0)

    if X_true is not None:
        metrics = defaultdict(list)

    all_imputations = []
    for i in range(maxit):
        if x.shape[0]*x.shape[1] > 1e6:
            U, d, V = randomized_svd(imp, n_components = np.minimum(200, x.shape[1]))
        else:
            U, d, V = np.linalg.svd(imp, compute_uv = True, full_matrices=False)
        d_thresh = np.maximum(d - lamb, 0)
        rank = (d_thresh > 0).sum()
        d_thresh = d_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        D_thresh = np.diag(d_thresh)
        res = np.dot(U_thresh, np.dot(D_thresh, V_thresh))
        # if converged(imp, res, mask, thresh):
        #     break
        imp[~mask] = res[~mask]

        if X_true is not None:
            imps = imp
            imps_mask = mask
            if X_fullref is not None:
                imps = imps[:-len(X_fullref)]
                imps_mask = imps_mask[:-len(X_fullref)]

            rmse = imputation_rmse_metric(X_imp=torch.from_numpy(imps).unsqueeze(1), X_true=torch.from_numpy(X_true).unsqueeze(1), M=(torch.from_numpy(imps_mask).bool()).unsqueeze(1))
            metrics['rmse'].append(asnumpy(rmse))

            mae = imputation_mae_metric(X_imp=torch.from_numpy(imps).unsqueeze(1), X_true=torch.from_numpy(X_true).unsqueeze(1), M=(torch.from_numpy(imps_mask).bool()).unsqueeze(1))
            metrics['mae'].append(asnumpy(mae))

        if X_fullref is not None:
            all_imputations.append(imp[:-len(X_fullref)].copy())
        else:
            all_imputations.append(imp.copy())


    if X_fullref is not None:
        imp = imp[:-len(X_fullref)]

    if X_true is not None:
        return U_thresh, imp, all_imputations, metrics
    else:
        return U_thresh, imp, all_imputations


def test_x(x, mask):
    # generate additional missing values
    # such that each row has at least 1 observed value (assuming also x.shape[0] > x.shape[1])
    save_mask = mask.copy()
    for i in range(x.shape[0]):
      idx_obs = np.argwhere(save_mask[i, :] == 1).reshape((-1))
      if len(idx_obs) > 0:
          j = np.random.choice(idx_obs, 1)
          save_mask[i, j] = 0
    mmask = np.array(np.random.binomial(np.ones_like(save_mask), save_mask * 0.1), dtype=bool)
    xx = x.copy()
    xx[mmask] = np.nan
    return xx, mmask

def cv_softimpute(x, grid_len = 15, maxit = 1000, thresh = 1e-5, X_fullref=None):
    # impute with constant
    mask = ~np.isnan(x)
    x0 = x.copy()
    #x0 = copy.deepcopy(x)
    x0[~mask] = 0
    # svd on x0
    if x.shape[0]*x.shape[1] > 1e6:
        _, d, _ = randomized_svd(x0, n_components = np.minimum(200, x.shape[1]))
    else:
        d = np.linalg.svd(x0, compute_uv=False, full_matrices=False)
    # generate grid for lambda values
    lambda_max = np.max(d)
    lambda_min = 0.001*lambda_max
    grid_lambda = np.exp(np.linspace(np.log(lambda_min), np.log(lambda_max), grid_len).tolist())

    cv_error = []
    for lamb in grid_lambda:
        xx, mmask = test_x(x, mask)
        mmask = ~np.isnan(xx)
        _, res, _ = softimpute(xx, lamb, maxit, thresh, X_fullref=X_fullref)
        cv_error.append(np.sqrt(np.nanmean((res.flatten() - x.flatten())**2)))

    return cv_error, grid_lambda
