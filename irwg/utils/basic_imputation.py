import torch

def impute_with_rand_binary(X, M):
    X_imp = torch.rand_like(X) > 0.5

    return X*M + X_imp*(~M)

def impute_with_std_gauss(X, M):
    X_imp = torch.rand_like(X)

    return X*M + X_imp*(~M)

imputation_fn = {
    'rand_bin': impute_with_rand_binary,
    'std_gauss': impute_with_std_gauss
}

