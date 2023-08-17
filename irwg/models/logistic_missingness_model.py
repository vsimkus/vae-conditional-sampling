
import numpy as np
import torch
from scipy import optimize

"""
Adapted from <https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py>
"""

class MNARLogisticModel(torch.nn.Module):
    def __init__(self, p, idxs_params, idxs_miss, coeffs, intercepts, exclude_inputs=True):
        super().__init__()
        self.p = p
        assert self.p < 1
        self.exclude_inputs = exclude_inputs

        self.idxs_params = idxs_params
        self.idxs_miss = torch.from_numpy(idxs_miss)
        self.coeffs = coeffs
        self.intercepts = intercepts

    @staticmethod
    def initialise_from_data(X, total_miss, frac_input_vars, exclude_inputs, rng):
        assert total_miss < 1

        idxs_params, idxs_miss, coeffs, intercepts = \
            create_MNAR_logistic_mechanism(X,
                                           p=total_miss,
                                           p_params=frac_input_vars,
                                           exclude_inputs=exclude_inputs,
                                           rng=rng)

        return MNARLogisticModel(total_miss, idxs_params, idxs_miss, coeffs, intercepts, exclude_inputs=exclude_inputs)

    def forward(self, X):
        raise NotImplementedError()

    def log_prob(self, X, M):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        inputs = X[..., self.idxs_params]
        cond_logits = inputs @ (self.coeffs.to(X.device)) + self.intercepts.to(X.device)
        logits = torch.ones_like(X, dtype=torch.float)*float('-inf')
        if self.exclude_inputs:
            logits[..., self.idxs_params] = torch.logit(torch.tensor(self.p))
        logits[..., self.idxs_miss] = cond_logits

        logits = -logits

        # Bernoulli does not like inf logits...
        # bern = torch.distributions.Bernoulli(logits=logits)
        bern = torch.distributions.Bernoulli(probs=torch.sigmoid(logits))
        log_prob = bern.log_prob(M.to(logits.dtype))
        return log_prob.sum(-1)

    def sample_mask(self, X, rng=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        inputs = X[..., self.idxs_params]

        cond_probs = torch.sigmoid(inputs @ (self.coeffs) + self.intercepts)

        probs = torch.zeros_like(X, dtype=torch.float)
        if self.exclude_inputs:
            probs[..., self.idxs_params] = self.p
        probs[..., self.idxs_miss] = cond_probs

        probs = 1-probs

        mask = torch.bernoulli(probs, generator=rng).bool()
        return mask

class MNARSelfLogisticModel(torch.nn.Module):
    def __init__(self, coeffs, intercepts):
        super().__init__()
        self.coeffs = coeffs
        self.intercepts = intercepts

    @staticmethod
    def initialise_from_data(X, total_miss, rng):
        assert total_miss < 1

        coeffs, intercepts = \
            create_MNAR_self_logistic_mechanism(X,
                                                p=total_miss,
                                                rng=rng)

        return MNARSelfLogisticModel(coeffs, intercepts)

    def forward(self, X):
        raise NotImplementedError()

    def log_prob(self, X, M):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        logits = X * self.coeffs.to(X.device) + self.intercepts.to(X.device)
        logits = -logits

        # Bernoulli does not like inf logits...
        # bern = torch.distributions.Bernoulli(logits=logits)
        bern = torch.distributions.Bernoulli(probs=torch.sigmoid(logits))
        log_prob = bern.log_prob(M.to(logits.dtype))
        return log_prob.sum(-1)

    def sample_mask(self, X, rng=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        probs = torch.sigmoid(X * (self.coeffs) + self.intercepts)
        probs = 1-probs

        mask = torch.bernoulli(probs, generator=rng).bool()
        return mask

class MARLogisticModel(torch.nn.Module):
    def __init__(self, idxs_params, idxs_miss, coeffs, intercepts):
        super().__init__()
        self.idxs_params = idxs_params
        self.idxs_miss = torch.from_numpy(idxs_miss)
        self.coeffs = coeffs
        self.intercepts = intercepts

    @staticmethod
    def initialise_from_data(X, total_miss, frac_input_vars, rng):
        p = total_miss / (1-frac_input_vars)
        assert p < 1

        idxs_params, idxs_miss, coeffs, intercepts = \
            create_MAR_logistic_mechanism(X,
                                           p=p,
                                           p_obs=frac_input_vars,
                                           rng=rng)

        return MARLogisticModel(idxs_params, idxs_miss, coeffs, intercepts)

    def forward(self, X):
        raise NotImplementedError()

    def log_prob(self, X, M):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        inputs = X[..., self.idxs_params]
        cond_logits = inputs @ (self.coeffs.to(X.device)) + self.intercepts.to(X.device)
        logits = torch.ones_like(X, dtype=torch.float)*float('-inf')
        logits[..., self.idxs_miss] = cond_logits

        logits = -logits

        # Bernoulli does not like inf logits...
        # bern = torch.distributions.Bernoulli(logits=logits)
        bern = torch.distributions.Bernoulli(probs=torch.sigmoid(logits))
        log_prob = bern.log_prob(M.to(logits.dtype))
        return log_prob.sum(-1)

    def sample_mask(self, X, rng=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        inputs = X[..., self.idxs_params]

        cond_probs = torch.sigmoid(inputs @ (self.coeffs) + self.intercepts)

        probs = torch.zeros_like(X, dtype=torch.float)
        probs[..., self.idxs_miss] = cond_probs

        probs = 1-probs

        mask = torch.bernoulli(probs, generator=rng).bool()
        return mask


def create_MAR_logistic_mechanism(X, p, p_obs, rng=None):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    parameters of the missingness model
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    # mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    # idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_obs = torch.randperm(d, generator=rng)[:d_obs]
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas, rng=rng)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    # ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    # ber = torch.rand(n, d_na)
    # mask[:, idxs_nas] = ber < ps

    return idxs_obs, idxs_nas, coeffs, intercepts

def create_MNAR_logistic_mechanism(X, p, p_params =.3, exclude_inputs=True, rng=None):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of inputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    parameters of the missingness model
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    # mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    # idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_params = torch.randperm(d, generator=rng)[:d_params] if exclude_inputs else torch.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas, rng=rng)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    # ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    # ber = torch.rand(n, d_na)
    # mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    # if exclude_inputs:
    #     mask[:, idxs_params] = torch.rand(n, d_params) < p

    return idxs_params, idxs_nas, coeffs, intercepts

def create_MNAR_self_logistic_mechanism(X, p, rng=None):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    Returns
    -------
    parameters of the mechanism
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True, rng=rng)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    # ps = torch.sigmoid(X * coeffs + intercepts)

    # ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    # mask = ber < ps if to_torch else ber < ps.numpy()

    return coeffs, intercepts

def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False, rng=None):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d, generator=rng)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
        # Workaround if there is a constant feature
        coeffs[torch.isinf(coeffs)] = 1.
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, generator=rng)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
        # Workaround if there is a constant feature
        coeffs[torch.isinf(coeffs)] = 1.
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        if p > 0:
            intercepts = torch.zeros(d)
            for j in range(d):
                def f(x):
                    return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
                intercepts[j] = optimize.bisect(f, -50, 50)
        else:
            intercepts = torch.ones(d)*float('-inf')
    else:
        d_obs, d_na = coeffs.shape
        if p > 0:
            intercepts = torch.zeros(d_na)
            for j in range(d_na):
                def f(x):
                    return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
                intercepts[j] = optimize.bisect(f, -50, 50)
        else:
            intercepts = torch.ones(d_na)*float('-inf')
    return intercepts
