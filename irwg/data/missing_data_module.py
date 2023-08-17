
import os
from enum import Enum, auto
from typing import Optional, List

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from irwg.data.fully_missing_data_filter import FullyMissingDataFilter

from irwg.data.uci import UCIDataset
from irwg.utils.transforms import Int8bToFloatStandardTransform, ImgSpatialFlattenTransform

from .cifar10 import CIFAR10
from .imagenet import ImageNet
from .ffhq import FFHQ
from .toy import ToyDataset, ToyCustomTestDataset
from .mnist import MNIST
from .omniglot import Omniglot
from .vae import VAEDataset
from .dataset_with_indices import DatasetWithIndices
from .uniform_missingness_provider import UniformMissingnessDataset
from .half_missingness_provider import HalfMissingnessDataset
from .quadrant_missingness_provider import QuadrantMissingnessDataset
from .logistic_missingness_provider import LogisticMissingnessDataset
from .mnar_classconditional_squares_missingness_provider import MNARClassConditionalSquaresMissingness
from .mar_squares_missingness_provider import MARSquaresMissingness
from .nan_missingness_provider import NanMissingnessDataset
from irwg.dependencies.NVAE.lmdb_datasets import LMDBDataset
from irwg.dependencies.NVAE.datasets import CropCelebA64
from .mnist_gmm import MNIST_GMM, MNIST_GMM_customtest
from .uci_gas import UCI_GAS
from .uci_hepmass import UCI_HEPMASS
from .uci_miniboone import UCI_MINIBOONE
from .uci_power import UCI_POWER
from .bsds300 import BSDS300

from einops import rearrange

# class DATASET(Enum):
#     ffhq256 = auto()
#     cifar10 = auto()
#     imagenet32 = auto()
#     imagenet64 = auto()
#     toy_mog2 = auto()
#     toy_mog_500d = auto()
#     toy_mog_10d = auto()
#     toy_mog_10d_large = auto()
#     mnist = auto()
#     mnist_fbin = auto()
#     mnist_sbin = auto()
#     omniglot28x28_fbin = auto()
#     omniglot28x28_sbin = auto()
#     vae_samples = auto()
from irwg.data.fetch_uci import DATASETS as UCI_DATASETS
datasets = 'ffhq256 cifar10 cifar10_float_flat imagenet32 imagenet64 celeba64 toy_mog2 toy_mog_500d toy_mog_10d toy_mog_10d_large toy_grid2dmog toy_grid2dmog2 toy_grid2dmog3 toy_grid2dmog4 toy_grid2dmog5 toy_grid2dmog_custtest0 mnist mnist_fbin mnist_fbin_padded mnist_sbin omniglot28x28_fbin omniglot28x28_sbin mnist_gmm mnist_gmm_wtargets mnist_gmm_custtest0 mnist_gmm_custtest1 vae_samples vae_samples_with_latents'
datasets += ' ' + ' '.join([f'uci_{d}' for d in  UCI_DATASETS])
datasets += ' uci_gas uci_power uci_hepmass uci_miniboone bsds300 uci_gas_wtrainnoise uci_power_wtrainnoise uci_hepmass_wtrainnoise uci_miniboone_wtrainnoise bsds300_wtrainnoise'
DATASET = Enum('DATASET', datasets, module=__name__)

class MISSINGNESS(Enum):
    uniform = 'uniform'
    top_half = 'top_half'
    bot_half = 'bot_half'
    quadrants = 'quadrants'
    MNAR_self_logistic = 'MNAR_self_logistic'
    MNAR_logistic_03fracinpt_excludeinpts = 'MNAR_logistic_03fracinpt_excludeinpts'
    MNAR_logistic_03fracinpt = 'MNAR_logistic_03fracinpt'
    MAR_logistic_03fracinpt = 'MAR_logistic_03fracinpt'
    MNAR_classcond_blocks = 'MNAR_classcond_blocks'
    MAR_blocks = 'MAR_blocks'
    MAR_blocks_2 = 'MAR_blocks_2'
    nan_miss = 'nan_miss'

train_missingness_fn = {
    "uniform": lambda dataset, hparams, rng: \
        UniformMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "top_half": lambda dataset, hparams, rng: \
        HalfMissingnessDataset(dataset, top_miss=True, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "bot_half": lambda dataset, hparams, rng: \
        HalfMissingnessDataset(dataset, top_miss=False, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "quadrants": lambda dataset, hparams, rng: \
        QuadrantMissingnessDataset(dataset, target_idx=0, img_dims=hparams.img_dims, total_miss=hparams.total_miss_train, rng=rng),
    "MNAR_self_logistic": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx,
                                    miss_type='MNAR_self', frac_input_vars=None, exclude_inputs=None, rng=rng),
    "MNAR_logistic_03fracinpt_excludeinpts": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MNAR', frac_input_vars=0.3, exclude_inputs=True, rng=rng),
    "MNAR_logistic_03fracinpt": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MNAR', frac_input_vars=0.3, exclude_inputs=False, rng=rng),
    "MAR_logistic_03fracinpt": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_train, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MAR', frac_input_vars=0.3, rng=rng),
    "MNAR_classcond_blocks": lambda dataset, hparams, rng: \
        MNARClassConditionalSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                                               max_concentration=hparams.classcond_concentration,
                                               num_patterns=hparams.classcond_maxpatterns,
                                               num_classes=hparams.classcond_numclasses,
                                               rng=rng),
    "MAR_blocks": lambda dataset, hparams, rng: \
        MARSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                              rng=rng),
    "MAR_blocks_2": lambda dataset, hparams, rng: \
        MARSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                              rng=rng, block_types=2),
    "nan_miss": lambda dataset, hparams, rng: \
        NanMissingnessDataset(dataset, target_idx=0, rng=rng),
}

val_missingness_fn = {
    "uniform": lambda dataset, hparams, rng: \
        UniformMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "top_half": lambda dataset, hparams, rng: \
        HalfMissingnessDataset(dataset, top_miss=True, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "bot_half": lambda dataset, hparams, rng: \
        HalfMissingnessDataset(dataset, top_miss=False, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "quadrants": lambda dataset, hparams, rng: \
        QuadrantMissingnessDataset(dataset, target_idx=0, img_dims=hparams.img_dims, total_miss=hparams.total_miss_val, rng=rng),
    "MNAR_self_logistic": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx,
                                    miss_type='MNAR_self', frac_input_vars=None, exclude_inputs=None, rng=rng),
    "MNAR_logistic_03fracinpt_excludeinpts": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MNAR', frac_input_vars=0.3, exclude_inputs=True, rng=rng),
    "MNAR_logistic_03fracinpt": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MNAR', frac_input_vars=0.3, exclude_inputs=False, rng=rng),
    "MAR_logistic_03fracinpt": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_val, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MAR', frac_input_vars=0.3, rng=rng),
    "MNAR_classcond_blocks": lambda dataset, hparams, rng: \
        MNARClassConditionalSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                                               max_concentration=hparams.classcond_concentration,
                                               num_patterns=hparams.classcond_maxpatterns,
                                               num_classes=hparams.classcond_numclasses,
                                               rng=rng),
    "MAR_blocks": lambda dataset, hparams, rng: \
        MARSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                              rng=rng),
    "MAR_blocks_2": lambda dataset, hparams, rng: \
        MARSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                              rng=rng, block_types=2),
    "nan_miss": lambda dataset, hparams, rng: \
        NanMissingnessDataset(dataset, target_idx=0, rng=rng),
}

test_missingness_fn = {
    "uniform": lambda dataset, hparams, rng: \
        UniformMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "top_half": lambda dataset, hparams, rng: \
        HalfMissingnessDataset(dataset, top_miss=True, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "bot_half": lambda dataset, hparams, rng: \
        HalfMissingnessDataset(dataset, top_miss=False, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx, rng=rng),
    "quadrants": lambda dataset, hparams, rng: \
        QuadrantMissingnessDataset(dataset, target_idx=0, img_dims=hparams.img_dims, total_miss=hparams.total_miss_test, rng=rng),
    "MNAR_self_logistic": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx,
                                    miss_type='MNAR_self', frac_input_vars=None, exclude_inputs=None, rng=rng),
    "MNAR_logistic_03fracinpt_excludeinpts": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MNAR', frac_input_vars=0.3, exclude_inputs=True, rng=rng),
    "MNAR_logistic_03fracinpt": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MNAR', frac_input_vars=0.3, exclude_inputs=False, rng=rng),
    "MAR_logistic_03fracinpt": lambda dataset, hparams, rng: \
        LogisticMissingnessDataset(dataset, target_idx=0, total_miss=hparams.total_miss_test, dims_missing_idx=hparams.dims_missing_idx,
                                       miss_type='MAR', frac_input_vars=0.3, rng=rng),
    "MNAR_classcond_blocks": lambda dataset, hparams, rng: \
        MNARClassConditionalSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                                               max_concentration=hparams.classcond_concentration,
                                               num_patterns=hparams.classcond_maxpatterns,
                                               num_classes=hparams.classcond_numclasses,
                                               rng=rng),
    "MAR_blocks": lambda dataset, hparams, rng: \
        MARSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                              rng=rng),
    "MAR_blocks_2": lambda dataset, hparams, rng: \
        MARSquaresMissingness(dataset, target_idx=0, dims_missing_idx=hparams.dims_missing_idx, img_dims=hparams.img_dims,
                              rng=rng, block_types=2),
    "nan_miss": lambda dataset, hparams, rng: \
        NanMissingnessDataset(dataset, target_idx=0, rng=rng),
}

class MissingDataModule(pl.LightningDataModule):
    """
    Missing Data provider module adding synthetic missingness to complete data.

    Args:
        dataset:                        Which dataset to load.
        batch_size:                     Size of the training and validation mini-batches.
        missingess:                     Missingness type.
        total_miss_train:               Total missing fraction in training data.
        total_miss_val:                 Total missing fraction in validation data.
        use_test_instead_val:           Loads test data instead of val data in "validation" loader.
        dims_missing_idx:               Which dimensions are missing, should be set depending on the dataset.
        img_dims:                       Used in some missingness providers.
        data_root:                      Root directory of all datasets.
        setup_seed:                     Random seed for the setup.
        num_workers:                    The number of dataloader workers. If None, chooses the smaller
                                        between 8 and the number of CPU cores on the machine.

        total_miss_test:                 Total missing fraction in test data.
        filter_fully_missing_test:       Remove fully missing datapoints from the test data.
        num_model_samples:               How many samples to generate from the model as a dataset

        classcond_concentration:        Class-conditional beta-binomial concentration param
        classcond_maxpatterns:          Class-conditional missingness max patterns
        classcond_numclasses:           Class-conditional missingness total num of classes

        num_first_datapoints_test:      Only uses the first num_first_datapoints_test datapoints

        return_uci_targets: If true returns the targets of UCI datasets as well
    """
    def __init__(self,
                 dataset: DATASET,
                 batch_size: int,
                 missingness: MISSINGNESS, # TODO: load by subclass instead?
                 total_miss_train: float,         # TODO: load by subclass instead?
                 total_miss_val: float,         # TODO: load by subclass instead?
                 use_test_instead_val: bool = False,
                 use_train_instead_test: bool = False,
                 dims_missing_idx: List[int] = [-1,],
                 img_dims: List[int] = None,
                 data_root: str = "./data",
                 setup_seed: int = None,
                 num_workers: int = None,
                 total_miss_test: float = 0.,         # TODO: load by subclass instead?
                 filter_fully_missing_train: bool = False,
                 filter_fully_missing_val: bool = False,
                 filter_fully_missing_test: bool = False,
                 num_model_samples: int = None,  # Only necessary when sampling the data from a model
                 num_first_datapoints_test: int = None,

                 classcond_concentration: float = None,
                 classcond_maxpatterns: int = None,
                 classcond_numclasses: int = None,

                 return_uci_targets: bool = False,
                 ):
        super().__init__()
        self.save_hyperparameters()

    def dataset(self, *args, **kwargs):
        if self.hparams.dataset.name == 'ffhq256':
            return FFHQ(*args, dataset='ffhq256', **kwargs)
        elif self.hparams.dataset.name == 'cifar10':
            return CIFAR10(*args, **kwargs)
        elif self.hparams.dataset.name == 'cifar10_float_flat':
            return CIFAR10(*args,
                           transform=torchvision.transforms.Compose([Int8bToFloatStandardTransform(to_num_bits=8),
                                                                     ImgSpatialFlattenTransform()]),
                           **kwargs)
        elif self.hparams.dataset.name == 'imagenet32':
            return ImageNet(*args, dataset='imagenet32', **kwargs)
        elif self.hparams.dataset.name == 'imagenet64':
            return ImageNet(*args, dataset='imagenet64', **kwargs)
        elif self.hparams.dataset.name == 'celeba64':
            root = kwargs['root'] + '/celeba64_lmdb/'
            kwargs['root'] = root
            train = None
            if kwargs['split'] == 'train':
                train=True
            elif kwargs['split'] == 'val' or kwargs['split'] == 'test':
                train=False
            del kwargs['split']

            class FlattenTransform:
                def __call__(self, x):
                    return rearrange(x, '... c h w -> ... c (h w)')

            size=64
            transform = torchvision.transforms.Compose([
                CropCelebA64(),
                torchvision.transforms.Resize(size),
                torchvision.transforms.ToTensor(),
                FlattenTransform(),
            ])

            del kwargs['rng']
            return LMDBDataset(*args, name='celeba64', train=train, **kwargs, transform=transform, is_encoded=True)
        elif self.hparams.dataset.name == 'toy_mog2':
            return ToyDataset(*args, filename='data_mog2', **kwargs)
        elif self.hparams.dataset.name == 'toy_mog_500d':
            return ToyDataset(*args, filename='data_mog_500d_no_params', **kwargs)
        elif self.hparams.dataset.name == 'toy_mog_10d':
            return ToyDataset(*args, filename='data_mog_10d', **kwargs)
        elif self.hparams.dataset.name == 'toy_mog_10d_large':
            return ToyDataset(*args, filename='data_mog_10d_large', **kwargs)
        elif self.hparams.dataset.name == 'toy_grid2dmog':
            return ToyDataset(*args, filename='data_grid2dmog', return_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'toy_grid2dmog2':
            return ToyDataset(*args, filename='data_grid2dmog2', return_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'toy_grid2dmog3':
            return ToyDataset(*args, filename='data_grid2dmog3', return_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'toy_grid2dmog4':
            return ToyDataset(*args, filename='data_grid2dmog4', return_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'toy_grid2dmog5':
            return ToyDataset(*args, filename='data_grid2dmog5', return_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'toy_grid2dmog_custtest0':
            return ToyCustomTestDataset(*args, dataset_idx=0, **kwargs)
        elif self.hparams.dataset.name == 'mnist':
            return MNIST(*args, **kwargs)
        elif self.hparams.dataset.name == 'mnist_fbin':
            return MNIST(*args, binarise_fixed=True, **kwargs)
        elif self.hparams.dataset.name == 'mnist_fbin_padded':
            return MNIST(*args, binarise_fixed=True, **kwargs,
                         transform=torchvision.transforms.Pad(padding=2),)
        elif self.hparams.dataset.name == 'mnist_sbin':
            return MNIST(*args, binarise_stochastic=True, **kwargs)
        elif self.hparams.dataset.name == 'omniglot28x28_fbin':
            return Omniglot(*args, binarise_fixed=True,
                            transform=torchvision.transforms.Resize((28,28),
                                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                            **kwargs)
        elif self.hparams.dataset.name == 'omniglot28x28_sbin':
            return Omniglot(*args, binarise_stochastic=True,
                            transform=torchvision.transforms.Resize((28,28),
                                                                    interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                            **kwargs)
        elif self.hparams.dataset.name == 'mnist_gmm':
            return MNIST_GMM(*args, **kwargs)
        elif self.hparams.dataset.name == 'mnist_gmm_wtargets':
            return MNIST_GMM(*args, return_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'mnist_gmm_custtest0':
            return MNIST_GMM_customtest(*args, version=0, **kwargs)
        elif self.hparams.dataset.name == 'mnist_gmm_custtest1':
            return MNIST_GMM_customtest(*args, version=1, **kwargs)
        elif self.hparams.dataset.name == 'vae_samples':
            return VAEDataset(*args, vae=self.model, num_samples=self.hparams.num_model_samples, **kwargs)
        elif self.hparams.dataset.name == 'vae_samples_with_latents':
            return VAEDataset(*args, vae=self.model, num_samples=self.hparams.num_model_samples, return_latents_as_targets=True, **kwargs)
        elif self.hparams.dataset.name == 'uci_gas':
            return UCI_GAS(*args, **kwargs)
        elif self.hparams.dataset.name == 'uci_gas_wtrainnoise':
            return UCI_GAS(*args, train_noise=0.001, **kwargs)
        elif self.hparams.dataset.name == 'uci_power':
            return UCI_POWER(*args, **kwargs)
        elif self.hparams.dataset.name == 'uci_power_wtrainnoise':
            return UCI_POWER(*args, train_noise=0.001, **kwargs)
        elif self.hparams.dataset.name == 'uci_hepmass':
            return UCI_HEPMASS(*args, **kwargs)
        elif self.hparams.dataset.name == 'uci_hepmass_wtrainnoise':
            return UCI_HEPMASS(*args, train_noise=0.001, **kwargs)
        elif self.hparams.dataset.name == 'uci_miniboone':
            return UCI_MINIBOONE(*args, **kwargs)
        elif self.hparams.dataset.name == 'uci_miniboone_wtrainnoise':
            return UCI_MINIBOONE(*args, train_noise=0.001, **kwargs)
        elif self.hparams.dataset.name.startswith('uci_'):
            dataset=self.hparams.dataset.name.split('uci_')[1]
            return UCIDataset(*args, dataset=dataset, return_targets=self.hparams.return_uci_targets, **kwargs)
        elif self.hparams.dataset.name == 'bsds300':
            return BSDS300(*args, **kwargs)
        elif self.hparams.dataset.name == 'bsds300_wtrainnoise':
            return BSDS300(*args, train_noise=0.001, **kwargs)
        else:
            raise NotImplementedError()

    def set_model(self, model):
        self.model = model

    def setup(self, stage: Optional[str] = None):
        rng = torch.Generator()
        rng = rng.manual_seed(self.hparams.setup_seed)

        if stage == 'fit':
            # Load train and validation splits
            rng_dataset = torch.Generator()
            rng_dataset.manual_seed(int(self.hparams.setup_seed*2.33/2))
            self.train_data = self.dataset(root=self.hparams.data_root, split='train', rng=rng_dataset)
            self.orig_train_data = self.train_data
            val_split = 'val'
            if self.hparams.use_test_instead_val:
                print('Using test data in val dataloader!')
                val_split = 'test'
            self.val_data = self.dataset(root=self.hparams.data_root, split=val_split, rng=rng_dataset)

            # Initialise missingness
            train_init_miss = train_missingness_fn[self.hparams.missingness.value]
            self.train_data = train_init_miss(self.train_data, self.hparams, rng=rng)
            val_init_miss = val_missingness_fn[self.hparams.missingness.value]
            self.val_data = val_init_miss(self.val_data, self.hparams, rng=rng)

            # # Filter fully missing datapoints
            if self.hparams.filter_fully_missing_train:
                self.train_data = FullyMissingDataFilter(self.train_data, miss_mask_idx=1)
            if self.hparams.filter_fully_missing_val:
                self.val_data = FullyMissingDataFilter(self.val_data, miss_mask_idx=1)

            # Augment datapoints
            self.train_data = DatasetWithIndices(self.train_data)
            self.val_data = DatasetWithIndices(self.val_data)

            print('Train data size:', len(self.train_data))
            print('Validation data size:', len(self.val_data))
        elif stage == 'test' and self.hparams.use_train_instead_test:
            # Load train and validation splits
            rng_dataset = torch.Generator()
            rng_dataset.manual_seed(int(self.hparams.setup_seed*2.33/2))
            train_data = self.dataset(root=self.hparams.data_root, split='train', rng=rng_dataset)
            val_split = 'val'
            if self.hparams.use_test_instead_val:
                print('Using test data in val dataloader!')
                val_split = 'test'
            val_data = self.dataset(root=self.hparams.data_root, split=val_split, rng=rng_dataset)

            # Initialise missingness
            train_init_miss = train_missingness_fn[self.hparams.missingness.value]
            train_data = train_init_miss(train_data, self.hparams, rng=rng)

            # Filter fully missing datapoints
            if self.hparams.filter_fully_missing_train:
                self.train_data = FullyMissingDataFilter(self.train_data, miss_mask_idx=1)
            if self.hparams.filter_fully_missing_val:
                val_data = FullyMissingDataFilter(val_data, miss_mask_idx=1)

            # Augment datapoints
            train_data = DatasetWithIndices(train_data)

            self.test_data = train_data
        elif stage == 'test':
            # Load test split
            rng_dataset = torch.Generator()
            rng_dataset.manual_seed(int(self.hparams.setup_seed*2.33/2))
            self.test_data = self.dataset(root=self.hparams.data_root, split='test', rng=rng_dataset)
            self.test_data_core = self.test_data

            # Initialise missingness
            test_init_miss = test_missingness_fn[self.hparams.missingness.value]
            self.test_data = test_init_miss(self.test_data, self.hparams, rng=rng)
            if hasattr(self.test_data, 'miss_model'):
                self.test_miss_model = self.test_data.miss_model

            if self.hparams.filter_fully_missing_test:
                self.test_data = FullyMissingDataFilter(self.test_data, miss_mask_idx=1)

            # Augment datapoints
            self.test_data = DatasetWithIndices(self.test_data)

            if self.hparams.num_first_datapoints_test is not None:
                self.test_data = Subset(self.test_data, indices=torch.arange(0, min(len(self.test_data), self.hparams.num_first_datapoints_test)))
            print('Test data size:', len(self.test_data))
        else:
            raise NotImplementedError(f'{stage=} is not implemented.')

    def train_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.hparams.num_workers is None else self.hparams.num_workers
        return DataLoader(self.train_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=True,
        )

    def val_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.hparams.num_workers is None else self.hparams.num_workers
        return DataLoader(self.val_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=False,
        )

    def test_dataloader(self):
        num_workers = min(8, os.cpu_count()) if self.hparams.num_workers is None else self.hparams.num_workers
        return DataLoader(self.test_data,
                          batch_size=self.hparams.batch_size,
                          num_workers=num_workers,
                          shuffle=False,
        )
