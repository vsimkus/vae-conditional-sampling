import numpy as np
import torch
from einops import rearrange, asnumpy, repeat, reduce

class ImputationHistoryQueue(object):
    def __init__(self, max_history_length, batch_shape, dtype):
        self.max_history_length = max_history_length
        self.tail_idx = -1
        self.full = False

        # t b k d
        self.history_queue = torch.empty((max_history_length,) + tuple(batch_shape), dtype=dtype)

    def enqueue_batch(self, X):
        self._increment_tail()
        self.history_queue[self.tail_idx] = X

    def _increment_tail(self):
        self.tail_idx = (self.tail_idx + 1) % self.max_history_length

        if self.tail_idx == self.max_history_length-1:
            self.full = True

    def __len__(self):
        assert self.tail_idx > -1, 'Queue is empty.'
        return self.tail_idx+1 if not self.full else self.max_history_length

    def sample_history(self, num_historical_proposals):
        """
        Implements stratified-sampling from the history
        """
        # Stratify the history by the number of proposals wanted
        hist_window_size = len(self) // num_historical_proposals
        # Sample an index within the history window size
        # NOTE: Using a common random number for all samples in the batch
        hist_proposal_idx = torch.randint(0, hist_window_size, size=(num_historical_proposals,))
        hist_proposal_idx += torch.arange(0, num_historical_proposals*hist_window_size, hist_window_size)
        # Shift the indices by the tail_idx
        hist_proposal_idx = (hist_proposal_idx + self.tail_idx) % len(self)

        # Select the proposals from the history, note it still have K (i.e. num chains) proposals for each index
        hist_proposals = self.history_queue[hist_proposal_idx]

        # Select one of the chains for each index
        B, K = self.history_queue.shape[1], self.history_queue.shape[2]
        chain_idx = torch.randint(0, K, size=(B,))
        hist_proposals = hist_proposals[:, torch.arange(B), chain_idx]

        return hist_proposals

    def sample_history_with_chain_idx_for_each_sample(self, num_historical_proposals, chain_idx):
        # Stratify the history by the number of proposals wanted
        hist_window_size = len(self) // num_historical_proposals
        # Sample an index within the history window size
        # NOTE: Using a common random number for all samples in the batch
        hist_proposal_idx = torch.randint(0, hist_window_size, size=(num_historical_proposals,))
        hist_proposal_idx += torch.arange(0, num_historical_proposals*hist_window_size, hist_window_size)
        # Shift the indices by the tail_idx
        hist_proposal_idx = (hist_proposal_idx + self.tail_idx) % len(self)

        # Select the proposals from the history, note it still have K (i.e. num chains) proposals for each index
        hist_proposals = self.history_queue[hist_proposal_idx]

        # Select one of the chains for each index
        B, K = self.history_queue.shape[1], self.history_queue.shape[2]
        hist_proposals = hist_proposals[torch.arange(num_historical_proposals), :, chain_idx]

        return hist_proposals

    def sample_history_nonstratified_for_each_chain(self, num_samples_per_chain):
        B, K = self.history_queue.shape[1], self.history_queue.shape[2]

        num_timesteps = num_samples_per_chain * K
        # Sample an index for each chain
        hist_proposal_idx = torch.randint(0, len(self), size=(num_timesteps,))

        # Select the proposals from the history for each chain
        hist_proposals = self.history_queue[hist_proposal_idx, :, torch.arange(K)]

        return hist_proposals


class ImputationHistoryQueue_with_restrictedavailability(object):
    def __init__(self, max_history_length, batch_shape, dtype):
        self.max_history_length = max_history_length
        self.tail_idx = -1
        self.full = False

        # t b k d
        self.history_queue = torch.empty((max_history_length,) + tuple(batch_shape), dtype=dtype)
        # b k
        self.available_timesteps = torch.zeros(tuple(batch_shape)[:2], dtype=torch.int64)

    def update_available_timesteps(self, make_full_history_available):
        """
        make_full_history_available:    BxK boolean tensor that sets all history of the chains with 1 to be available
                                        up to the current step
        """
        self.available_timesteps[make_full_history_available] = len(self)

    def enqueue_batch(self, X):
        if self.full:
            self.available_timesteps -= 1
            self.available_timesteps[self.available_timesteps < 0] = 0

        self._increment_tail()
        self.history_queue[self.tail_idx] = X

    def _increment_tail(self):
        self.tail_idx = (self.tail_idx + 1) % self.max_history_length

        if self.tail_idx == self.max_history_length-1:
            self.full = True

    def __len__(self):
        assert self.tail_idx > -1, 'Queue is empty.'
        return self.tail_idx+1 if not self.full else self.max_history_length

    # def _get_sampling_probabilities(self):
    #     # Return uniform sampling probabilities out of the available timesteps for each chain
    #     # If there are no available timesteps, return nan
    #     probs = torch.ones((self.max_history_length,) + self.available_timesteps.shape)
    #     probs[(torch.arange(self.max_history_length)[:, None, None] >= self.available_timesteps[None, :, :])] = 0.

    #     # Sum adds a non-negligible overhead
    #     # probs = probs / probs.sum(dim=0, keepdim=True)
    #     probs = probs / self.available_timesteps

    #     # Shift by the tail_idx to align with the history queue
    #     probs = torch.roll(probs, (self.tail_idx+1 - len(self)) % self.max_history_length, dims=0)

    #     return probs

    # def sample_history_nonstratified_for_each_chain(self, num_samples_per_chain):
    #     B, K = self.history_queue.shape[1], self.history_queue.shape[2]

    #     probs = self._get_sampling_probabilities()
    #     # TODO: undo this (temporary for debuging)
    #     # probs = torch.ones((len(self.max_history_length),) + self.available_timesteps.shape) / self.max_history_length
    #     # T B K -> (B K) T
    #     no_available_samples = ~torch.isfinite(probs.sum(dim=0))
    #     assert torch.all(no_available_samples == (self.available_timesteps == 0))
    #     assert torch.all((probs > 0.).sum(0) <= self.available_timesteps)
    #     # Set some probabilities to avoid issues with nan in multinomial
    #     probs[:, no_available_samples] = 1/self.max_history_length

    #     # Sample proposal indices
    #     probs = rearrange(probs, 't b k -> (b k) t')
    #     hist_proposal_idx = torch.multinomial(probs, num_samples_per_chain, replacement=True)
    #     # breakpoint()
    #     # hist_proposal_idx = custom_torch_multinomial_with_replacement_using_Gumbel_trick(probs, num_samples_per_chain)
    #     # breakpoint()
    #     hist_proposal_idx = rearrange(hist_proposal_idx, '(b k) t -> t b k', b=B, k=K)

    #     # Select the proposals from the history for each chain
    #     hist_proposals = self.history_queue[hist_proposal_idx, torch.arange(B)[None, :, None], torch.arange(K)[None, None, :]]

    #     # TODO undo this
    #     # num_timesteps = num_samples_per_chain * K
    #     # # Sample an index for each chain
    #     # hist_proposal_idx = torch.randint(0, len(self), size=(num_timesteps,))
    #     # # Select the proposals from the history for each chain
    #     # hist_proposals = self.history_queue[hist_proposal_idx, :, torch.arange(K)]
    #     #######################

    #     # Set the hist proposals to nan if there are no available samples
    #     hist_proposals = hist_proposals.clone()
    #     hist_proposals[:, no_available_samples] = float('nan')
    #     # Rearrange to conform to the other functions return order
    #     hist_proposals = rearrange(hist_proposals, 't b k d -> t k b d')

    #     # If only one sample per chain, remove the dimension
    #     if num_samples_per_chain == 1:
    #         hist_proposals = hist_proposals.squeeze(0)

    #     return hist_proposals

    # def sample_history_nonstratified_for_each_chain(self, num_samples_per_chain):
    #     B, K = self.history_queue.shape[1], self.history_queue.shape[2]

    #     num_timesteps = num_samples_per_chain * K
    #     # Sample an index for each chain
    #     hist_proposal_idx = torch.randint(0, len(self), size=(num_timesteps,))

    #     # Select the proposals from the history for each chain
    #     hist_proposals = self.history_queue[hist_proposal_idx, :, torch.arange(K)]

    #     return hist_proposals

    def sample_history_nonstratified_for_each_chain(self, num_samples_per_chain):
        B, K = self.history_queue.shape[1], self.history_queue.shape[2]

        # Sample proposal indices
        hist_proposal_idx = self.sample_index_for_each_chain(self.available_timesteps,
                                                             B, K, num_samples_per_chain)
        hist_proposal_idx = rearrange(hist_proposal_idx, 'b k t -> t b k', b=B, k=K)
        no_available_samples = (hist_proposal_idx == -1)[0]

        # Select the proposals from the history for each chain
        hist_proposals = self.history_queue[hist_proposal_idx, torch.arange(B)[None, :, None], torch.arange(K)[None, None, :]]
        hist_proposals = hist_proposals.clone()
        hist_proposals[:, no_available_samples] = float('nan')
        # Rearrange to conform to the other functions return order
        hist_proposals = rearrange(hist_proposals, 't b k d -> t k b d')

        # If only one sample per chain, remove the dimension
        if num_samples_per_chain == 1:
            hist_proposals = hist_proposals.squeeze(0)

        return hist_proposals

    @torch.jit.export
    def sample_index_for_each_chain(self, available_timesteps: torch.Tensor,
                                    B:int , K: int, num_samples_per_chain: int):
        """
        Uniform sampling of available imputations for each chain.
        Uses a loop, but is often faster than the vectorised versions.
        """
        hist_proposal_idx = torch.empty(B, K, num_samples_per_chain, dtype=torch.int64)
        for b in range(B):
            for k in range(K):
                available_timesteps_bk = available_timesteps[b, k]
                if available_timesteps_bk == 0:
                    available_timesteps_bk = torch.tensor(1)
                idx = torch.randint(0, available_timesteps_bk, size=(num_samples_per_chain,))

                if self.full:
                    idx = (idx + self.tail_idx+1) % self.max_history_length
                else:
                    # idx = (idx + self.tail_idx+1) % len(self)
                    idx = idx
                hist_proposal_idx[b, k] = idx

                if available_timesteps_bk == 0:
                    hist_proposal_idx[b, k] = torch.tensor(-1)

        return hist_proposal_idx

# def np_multinomial(batched_probs, num_samples, *, replacement=False, normalise_probs=False):
#     """
#     Adapted from <https://github.com/rubencart/self-critical.pytorch/pull/1/files>

#     Faster when batch size is small.

#     Faster surrogate for torch.multinomial, API is the same
#     :param batched_probs: tensor of bs x dim, every row is a dim-dimensional multinomial distribution
#                         from which num_samples are taken
#     :param num_samples: int
#     :return: tensor of bs x num_samples
#     """
#     norm_probs = asnumpy(batched_probs)
#     if normalise_probs:
#         norm_probs = asnumpy(batched_probs / batched_probs.sum(dim=-1, keepdim=True))
#     result = np.empty((batched_props.shape[0], num_samples), dtype=np.int64)
#     length = batched_props.shape[-1]
#     for i, probs in enumerate(norm_probs):
#         sample = np.random.choice(length, num_samples, p=probs, replace=replacement)
#         result[i] = sample
#     return torch.from_numpy(result).to(batched_probs.device)

# def custom_torch_multinomial_with_replacement(probs, num_samples):
#     # TODO verify correctness
#     cdf_boundaries = torch.cumsum(probs, dim=-1)

#     idx = torch.searchsorted(cdf_boundaries, torch.rand((probs.shape[0], num_samples), device=probs.device), right=True)
#     idx = torch.clamp_max_(idx, max=probs.shape[-1]-1)

#     return idx

# def custom_torch_multinomial_with_replacement_using_Gumbel_trick(probs, num_samples):
#     log_probs = torch.log(probs)

#     gumbel_noise = -torch.log(-torch.log(torch.rand(probs.shape + (num_samples,), device=probs.device)))
#     logits_with_noise = log_probs[..., None] + gumbel_noise
#     return torch.argmax(logits_with_noise, dim=-2)

if __name__ == '__main__':
    history = ImputationHistoryQueue_with_restrictedavailability(8, (2, 3, 4), torch.float32)

    # Test restricted history availability

    # B K D
    history.enqueue_batch(torch.ones((2, 3, 4))*1.)
    history.enqueue_batch(torch.ones((2, 3, 4))*2.)
    history.update_available_timesteps(torch.tensor([[1, 0, 1],
                                                     [0, 1, 1]], dtype=torch.bool))
    true_available_timesteps = torch.tensor([[2, 0, 2], [0, 2, 2]])
    assert torch.all(true_available_timesteps == history.available_timesteps)
    history.enqueue_batch(torch.ones((2, 3, 4))*3.)
    history.enqueue_batch(torch.ones((2, 3, 4))*4.)
    history.enqueue_batch(torch.ones((2, 3, 4))*5.)
    history.update_available_timesteps(torch.tensor([[0, 0, 1],
                                                     [0, 1, 0]], dtype=torch.bool))
    true_available_timesteps = torch.tensor([[2, 0, 5], [0, 5, 2]])
    assert torch.all(true_available_timesteps == history.available_timesteps)
    history.enqueue_batch(torch.ones((2, 3, 4))*6.)
    assert torch.all(true_available_timesteps == history.available_timesteps)
    history.enqueue_batch(torch.ones((2, 3, 4))*7.)
    history.enqueue_batch(torch.ones((2, 3, 4))*8.)
    history.update_available_timesteps(torch.tensor([[1, 0, 1],
                                                     [0, 0, 1]], dtype=torch.bool))
    true_available_timesteps = torch.tensor([[8, 0, 8], [0, 5, 8]])
    assert torch.all(true_available_timesteps == history.available_timesteps)

    history.enqueue_batch(torch.ones((2, 3, 4))*9.)
    history.enqueue_batch(torch.ones((2, 3, 4))*11.)
    true_available_timesteps = torch.tensor([[6, 0, 6], [0, 3, 6]])
    assert torch.all(true_available_timesteps == history.available_timesteps)

    history.sample_history_nonstratified_for_each_chain(2)
