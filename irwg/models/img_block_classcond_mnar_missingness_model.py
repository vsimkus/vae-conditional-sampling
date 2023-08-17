import math

import torch
import numpy as np
import matplotlib as mpl

from irwg.utils.beta_binomial import BetaBinomial


class MNARBlockImgClassConditionalMissingnessModel(torch.nn.Module):
    def __init__(self, patterns, classcond_params):
        super().__init__()
        self.register_buffer('patterns', patterns)
        self.register_buffer('classcond_params', classcond_params)

    @staticmethod
    def initialise(num_classes, img_shape, num_patterns, max_concentration=60, rng=None):
        patterns = generate_patterns(num_patterns, img_shape, rng=rng)
        classcond_params = generate_classconditional_distributions(num_classes,
                                                                   max_concentration=max_concentration)
        return MNARBlockImgClassConditionalMissingnessModel(patterns, classcond_params)

    def set_classifier(self, classifier):
        self.classifier = classifier

    def forward(self, X):
        raise NotImplementedError()

    def log_prob(self, X, M):
        """
        NOTE: using the classifier instead of true class C here
        """
        assert M.shape[1] == 1 and len(M.shape) <= 3
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        # NOTE: this can be slow. Can we optimise this?
        pattern_idx = torch.empty(M.shape[0], device=X.device)
        for i in range(M.shape[0]):
            M_i = M[i, 0]
            idx = torch.where((self.patterns == M_i).all(dim=-1))[0]
            pattern_idx[i] = idx

        bb = BetaBinomial(self.classcond_params[:, 0], self.classcond_params[:, 1], len(self.patterns)-1)
        mask_logprobs = bb.log_prob(pattern_idx.unsqueeze(-1)).unsqueeze(1)

        classifier_logprobs = self.classifier(X)[0]

        log_prob = torch.logsumexp(mask_logprobs + classifier_logprobs, dim=-1)

        return log_prob

    def sample_mask(self, C, rng=None):
        """
        NOTE: using the true class C here
        """
        if isinstance(C, np.ndarray):
            C = torch.from_numpy(C)

        params = self.classcond_params[C]
        bb = BetaBinomial(params[:, 0], params[:, 1], len(self.patterns)-1)
        pattern_idx = bb.sample(generator=rng).long()
        mask = self.patterns[pattern_idx]

        return mask


def generate_patterns(num_patterns, img_shape, rng=None):
    block_size_num_blocks = torch.tensor([[5, 10], [5, 12], [7, 5], [7, 6], [9, 3], [9, 4], [11, 2], [11, 3], [13, 1], [15, 1]])

    # For each pattern select #size_blocks and #num_blocks from the selection above
    block_size_num_blocks = torch.tile(block_size_num_blocks, (math.ceil(num_patterns / len(block_size_num_blocks)), 1))
    perm = torch.randperm(len(block_size_num_blocks), generator=rng)
    block_size_num_blocks = block_size_num_blocks[perm]
    block_size_num_blocks = block_size_num_blocks[:num_patterns]

    # Sample locations from a Beta distribution
    location_distr = BetaBinomial(torch.tensor([3., 3.]), torch.tensor([3., 3.]), total_count=torch.tensor(img_shape)-1,)

    patterns = torch.zeros(num_patterns, torch.prod(torch.tensor(img_shape)), dtype=torch.bool)
    for i in range(num_patterns):
        size_blocks, num_blocks = block_size_num_blocks[i]

        locs = location_distr.sample((num_blocks,), generator=rng).long()

        mask = torch.ones(img_shape, dtype=torch.bool)
        for j in range(num_blocks):
            x, y = locs[j]
            x_min = max(x - torch.div(size_blocks, 2, rounding_mode='floor'), 0)
            x_max = min(x + torch.div(size_blocks, 2, rounding_mode='floor'), img_shape[0])
            y_min = max(y - torch.div(size_blocks, 2, rounding_mode='floor'), 0)
            y_max = min(y + torch.div(size_blocks, 2, rounding_mode='floor'), img_shape[0])
            mask[x_min:x_max, y_min:y_max] = 0.0

        patterns[i] = mask.flatten()

    return patterns

def generate_classconditional_distributions(num_classes, max_concentration=60):
    # BetaBinomial params
    params = torch.zeros(num_classes, 2)

    # First half of classes, beta=max_concentration
    params[:num_classes//2, 1] = max_concentration
    # alpha = (gamma*(beta - 2)+1)/(1-gamma), where gamma = c/(num_classes-1)
    gamma = torch.arange(num_classes//2)/(num_classes-1)
    params[:num_classes//2, 0] = (gamma*(max_concentration - 2) + 1)/(1-gamma)
    # Avoid inf
    params[params[:, 0] < 0, 0] = 1

    # Second half of classes, alpha=max_concentration
    params[num_classes//2:, 0] = max_concentration
    # beta = (alpha-1)/gamma - alpha + 2, where gamma = c/(num_classes-1)
    gamma = torch.arange(num_classes//2, num_classes)/(num_classes-1)
    params[num_classes//2:, 1] = (max_concentration-1)/gamma - max_concentration + 2
    # Avoid negative
    params[params[:, 1] < 0, 1] = 1

    return params

def plot_classconditional_distributions(params, num_patterns):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    for i in range(0, len(params)):
        bb = BetaBinomial(params[i, 0], params[i, 1], num_patterns-1)

        x = torch.arange(num_patterns)
        probs = bb.log_prob(x).exp()
        # assert(torch.allclose(probs.sum(), torch.tensor(1.), rtol=1e-04, atol=1e-04)), f'Assert error at {i}'

        ax.plot(x, probs, marker='x', label=i)

    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.legend()
    plt.show()


if __name__ == '__main__':
    num_classes = 20
    # max_concentration = 60
    max_concentration = 10000
    num_patterns=1000
    params = generate_classconditional_distributions(num_classes=num_classes, max_concentration=max_concentration)

    plot_classconditional_distributions(params, num_patterns)
