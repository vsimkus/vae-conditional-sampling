""" Setup script for irwg package. """

from setuptools import setup, find_packages

long_description = ('Code for "Conditional Sampling of Variational Autoencoders via Iterated Approximate Ancestral Sampling". '
                    'Implementing LAIR and AC-MWG samplers (plus all the baselines).')

setup(
    name="irwg",
    author="Vaidotas Simkus",
    description=("Conditional sampling of VAEs using LAIR and AC-MWG samplers."),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vsimkus/vae-conditional-sampling",
    packages=find_packages()
)
