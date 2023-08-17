
import torch
import torch.utils.data as data


class VAEDataset(data.Dataset):
    """
    A dataset wrapper that samples a VAE to create data
    """

    def __init__(self, root: str, # NOTE: ignored
                 split: str = 'train', # NOTE: ignored
                 num_samples: int = None,
                 vae: torch.nn.Module = None,
                 return_latents_as_targets: bool = False,
                 rng: torch.Generator = None):
        """
        Args:
            root:       root directory that contains all data
            filename:   filename of the data file
            split:      data split, e.g. train, val, test
            return_latents_as_targets: whether to return latents as targets
            rng:        random number generator
        """
        super().__init__()
        self.return_latents_as_targets = return_latents_as_targets

        # Sample a dataset
        with torch.no_grad():
            if return_latents_as_targets:
                samples, latents = vae.sample(num_samples, return_latents=True, rng=rng)
                self.targets = latents.cpu()
            else:
                samples = vae.sample(num_samples, rng=rng)

        self.data = samples.cpu()

    def __getitem__(self, index):
        """
        Args:
            index: index of sample
        Returns:
            image: dataset sample
        """
        if self.return_latents_as_targets:
            return self.data[index], self.targets[index]
        else:
            return self.data[index]

    def __setitem__(self, key, value):
        """
        Args:
            key: index of sample
        """
        self.data[key] = value

    def __len__(self):
        return self.data.shape[0]
