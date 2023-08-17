# Conditional Sampling of Variational Autoencoders via Iterated Approximate Ancestral Sampling

This repository contains the research code for

> Vaidotas Simkus, Michael U. Gutmann. Conditional Sampling of Variational Autoencoders via Iterated Approximate Ancestral Sampling. Transactions on Machine Learning Research, 2023.

The paper can be found here: <https://openreview.net/forum?id=I5sJ6PU6JN>.

The code is shared for reproducibility purposes and is not intended for production use. It should also serve as a reference implementation for anyone wanting to use LAIR or AC-MWG for conditional sampling of VAEs (for e.g. missing data imputation using pre-trained VAEs).

## Abstract

Conditional sampling of variational autoencoders (VAEs) is needed in various applications, such as missing data imputation, but is computationally intractable. A principled choice for asymptotically exact conditional sampling is Metropolis-within-Gibbs (MWG). However, we observe that the tendency of VAEs to learn a structured latent space, a commonly desired property, can cause the MWG sampler to get “stuck” far from the target distribution. This paper mitigates the limitations of MWG: we systematically outline the pitfalls in the context of VAEs, propose two original methods that address these pitfalls, and demonstrate an improved performance of the proposed methods on a set of sampling tasks.

## Dependencies

Install python dependencies from conda and the `irwg` project package with

```bash
conda env create -f environment.yml
conda activate irwg
python setup.py develop
```

If the dependencies in `environment.yml` change, update dependencies with

```bash
conda env update --file environment.yml
```

## Organisation of the code

* `./irwg/data/` contains data loaders and missingness generators.
* `./irwg/models/` contains the neural network model implementations.
* `./irwg/sampling/` contains the code related to VAE sampling.
  * `test_step_vae_sampling.py` contains the implementations of the methods in the paper.
  (Note: some method names are different from the paper)
  * LAIR is implemented in a class called `TestVAELatentAdaptiveImportanceResampling`
  * AC-MWG is implemented in a class called `TestVAEAdaptiveCollapsedMetropolisWithinGibbs`
* `./configs/` contains the yaml configuration files containing all the information about each experiment.
* `./helpers/` directory contains various helper scripts for the analysis of the imputations.
  * `compute_mnist_mog_posterior_probs.py` computes the metrics on MNIST-GMM data.
  * `eval_large_uci_joint_imputed_dataset_divergences.py` computes the metrics on UCI data and stores into a file.
  * `eval_omniglot_joint_imputed_dataset_fids.py` computes the metrics on Omniglot data and stores into a file.
  * `create_marginal_vae_imputations.py` creates imputations by sampling the marginal of the VAE (i.e. unconditional imputation baseline)
  * Configs for the helper scripts are also located in `./configs/` directory.
* `./notebooks/` contain analysis notebooks that produce the figures in the paper, using the outputs from the helper scripts.

## Running the code

Activate the conda environment

```bash
conda activate irwg
```

### VAE training

To train the VAE, which we use for sampling run e.g. 

```bash
python train.py --config=configs/mnist_gmm/vae_convresnet3.yaml
```

### VAE sampling

Then, to sample a VAE using one of the methods run

```bash
python test.py --config=configs/mnist_gmm/samples/vae_convresnet3_k4_irwg_i1_dmis_gr_mult_replenish1_finalresample.yaml
```

### Analysis helper scripts

Then, use `./helpers/compute_mnist_mog_posterior_probs.py` to compute the metrics and store them in a file, and then plot them in a notebook.

Similarly, for UCI data use `./helpers/eval_large_uci_joint_imputed_dataset_divergences.py` to compute the metrics, and then plot them in a notebook.
