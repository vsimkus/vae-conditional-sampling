#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Adapted from https://github.com/BorisMuzellec/MissingDataOT
#

import os

import numpy as np
import pandas as pd
import wget
from sklearn.datasets import fetch_california_housing, load_iris, load_wine

DATASETS = ['iris', 'wine', 'california', 'banknote', #'parkinsons',
            'climate_model_crashes', 'concrete_compression',
            'yacht_hydrodynamics', 'airfoil_self_noise',
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation',
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax',
            'blood_transfusion', 'breast_cancer_diagnostic',
            'connectionist_bench_vowel', 'concrete_slump',
            'wine_quality_red', 'wine_quality_white']


def dataset_loader(root, dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes
    datasets are located in './datasets'. If the called for dataset is not in
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    root: root directory
    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.

    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS , f"Dataset not supported: {dataset}"

    if dataset in DATASETS:
        if dataset == 'iris':
            # https://archive.ics.uci.edu/ml/datasets/iris
            iris = load_iris()
            X, Y = iris['data'], iris['target']
        elif dataset == 'wine':
            # https://archive.ics.uci.edu/ml/datasets/wine
            wine = load_wine()
            X, Y = wine['data'], wine['target']
        elif dataset == 'california':
            # https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
            california = fetch_california_housing(data_home=root)
            X, Y = california['data'], california['target']
        elif dataset == 'parkinsons':
            # https://archive.ics.uci.edu/ml/datasets/parkinsons
            parkinsons = fetch_parkinsons(root)
            X, Y = parkinsons['data'], parkinsons['target']
        elif dataset == 'banknote':
            # https://archive.ics.uci.edu/ml/datasets/banknote+authentication
            banknote = fetch_banknote(root)
            X, Y = banknote['data'], banknote['target']
        elif dataset == 'climate_model_crashes':
            # https://archive.ics.uci.edu/ml/datasets/climate+model+simulation+crashes
            climate = fetch_climate_model_crashes(root)
            X, Y = climate['data'], climate['target']
        elif dataset == 'concrete_compression':
            # https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
            concrete = fetch_concrete_compression(root)
            X, Y = concrete['data'], concrete['target']
        elif dataset == 'yacht_hydrodynamics':
            # https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
            yacht = fetch_yacht_hydrodynamics(root)
            X, Y = yacht['data'], yacht['target']
        elif dataset == 'airfoil_self_noise':
            # https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
            airfoil = fetch_airfoil_self_noise(root)
            X, Y = airfoil['data'], airfoil['target']
        elif dataset == 'connectionist_bench_sonar':
            # https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
            conn_bench_sonar = fetch_connectionist_bench_sonar(root)
            X, Y = conn_bench_sonar['data'], conn_bench_sonar['target']
        elif dataset == 'ionosphere':
            # https://archive.ics.uci.edu/ml/datasets/ionosphere
            ionosphere = fetch_ionosphere(root)
            X, Y = ionosphere['data'], ionosphere['target']
        elif dataset == 'qsar_biodegradation':
            # https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation
            qsar = fetch_qsar_biodegradation(root)
            X, Y = qsar['data'], qsar['target']
        elif dataset == 'seeds':
            # https://archive.ics.uci.edu/ml/datasets/seeds
            seeds = fetch_seeds(root)
            X, Y = seeds['data'], seeds['target']
        elif dataset == 'glass':
            # https://archive.ics.uci.edu/ml/datasets/glass+identification
            glass = fetch_glass(root)
            X, Y = glass['data'], glass['target']
        elif dataset == 'ecoli':
            # https://archive.ics.uci.edu/ml/datasets/ecoli
            ecoli = fetch_ecoli(root)
            X, Y = ecoli['data'], ecoli['target']
        elif dataset == 'yeast':
            # https://archive.ics.uci.edu/ml/datasets/yeast
            yeast = fetch_yeast(root)
            X, Y = yeast['data'], yeast['target']
        elif dataset == 'libras':
            # https://archive.ics.uci.edu/ml/datasets/Libras+Movement
            libras = fetch_libras(root)
            X, Y = libras['data'], libras['target']
        elif dataset == 'planning_relax':
            # https://archive.ics.uci.edu/ml/datasets/Planning+Relax
            planning = fetch_planning_relax(root)
            X, Y = planning['data'], planning['target']
        elif dataset == 'blood_transfusion':
            # https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
            blood = fetch_blood_transfusion(root)
            X, Y = blood['data'], blood['target']
        elif dataset == 'breast_cancer_diagnostic':
            # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
            cancer = fetch_breast_cancer_diagnostic(root)
            X, Y = cancer['data'], cancer['target']
        elif dataset == 'connectionist_bench_vowel':
            # https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Vowel+Recognition+-+Deterding+Data)
            conn_bench_vowel = fetch_connectionist_bench_vowel(root)
            X, Y = conn_bench_vowel['data'], conn_bench_vowel['target']
        elif dataset == 'concrete_slump':
            # https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
            slump = fetch_concrete_slump(root)
            X, Y = slump['data'], slump['target']
        elif dataset == 'wine_quality_red':
            # https://archive.ics.uci.edu/ml/datasets/wine+quality
            wine_red = fetch_wine_quality_red(root)
            X, Y = wine_red['data'], wine_red['target']
        elif dataset == 'wine_quality_white':
            wine_white = fetch_wine_quality_white(root)
            X, Y = wine_white['data'], wine_white['target']

        return X, Y


def fetch_banknote(root):
    dir = f'{root}/banknote/'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'data_banknote_authentication.txt'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('bool')

    return Xy


def fetch_parkinsons(root):
    dir = f'{root}/parkinsons/'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'parkinsons.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] =  df.values[:, 17].astype('bool')
        Xy['data'] = np.delete(Xy['data'], 16, axis=1)

    return Xy


def fetch_climate_model_crashes(root):
    dir = f'{root}/climate_model_crashes'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'pop_failures.dat'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = 0)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1]
        Xy['target'] =  df.values[:, -1].astype('bool')

    return Xy


def fetch_concrete_compression(root):
    dir = f'{root}/concrete_compression'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'Concrete_Data.xls'), 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-2]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_yacht_hydrodynamics(root):
    dir = f'{root}/yacht_hydrodynamics'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'yacht_hydrodynamics.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy

def fetch_airfoil_self_noise(root):
    dir = f'{root}/airfoil_self_noise'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'airfoil_self_noise.dat'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] =  df.values[:, -1]

    return Xy


def fetch_connectionist_bench_sonar(root):
    dir = f'{root}/connectionist_bench_sonar'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'sonar.all-data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c.astype('bool')

    return Xy


def fetch_ionosphere(root):
    dir = f'{root}/ionosphere'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'ionosphere.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c.astype('bool')

    return Xy


def fetch_qsar_biodegradation(root):
    dir = f'{root}/qsar_biodegradation'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'biodeg.csv'), 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c.astype('bool')

    return Xy


def fetch_seeds(root):
    dir = f'{root}/seeds'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'seeds_dataset.txt'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  (df.values[:, -1]-1).astype('int64')

    return Xy


def fetch_glass(root):
    dir = f'{root}/glass'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'glass.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  (df.values[:, -1]-1).astype('int64')

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c

    return Xy


def fetch_ecoli(root):
    dir = f'{root}/ecoli'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'ecoli.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c

    return Xy

def fetch_yeast(root):
    dir = f'{root}/yeast'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'yeast.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c

    return Xy


def fetch_libras(root):
    dir = f'{root}/libras'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'movement_libras.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  (df.values[:, -1]-1).astype('int64')

    return Xy

def fetch_planning_relax(root):
    dir = f'{root}/planning_relax'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'plrx.txt'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header = None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  (df.values[:, -1]-1).astype('bool')

    return Xy


def fetch_blood_transfusion(root):
    dir = f'{root}/blood_transfusion'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'transfusion.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('bool')

    return Xy

def fetch_breast_cancer_diagnostic(root):
    dir = f'{root}/breast_cancer_diagnostic'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'wdbc.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] =  df.values[:, 1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c.astype('bool')

    return Xy


def fetch_connectionist_bench_vowel(root):
    dir = f'{root}/connectionist_bench_vowel'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'vowel-context.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] =  df.values[:, -1].astype('int64')

    return Xy


def fetch_concrete_slump(root):
    dir = f'{root}/concrete_slump'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'slump_test.data'), 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, 1:-3].astype('float')
        Xy['target'] =  df.values[:, -3:]

        # Use only one of the target variables
        Xy['target'] = Xy['target'][:, 0]

    return Xy


def fetch_wine_quality_red(root):
    dir = f'{root}/wine_quality_red'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out=dir)

    with open(os.path.join(dir, 'winequality-red.csv'), 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c

    return Xy


def fetch_wine_quality_white(root):
    dir = f'{root}/wine_quality_white'
    if not os.path.isdir(dir):
        os.mkdir(dir)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out=dir)

    with open(os.path.join(dir,'winequality-white.csv'), 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] =  df.values[:, -1]

        b, c = np.unique(Xy['target'], return_inverse=True)
        Xy['target'] = c

    return Xy

def plot_histograms(data):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, data.shape[-1])

    for i in range(data.shape[-1]):
        axes[i].hist(data[:, i])

    plt.show()
