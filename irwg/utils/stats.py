import os
import numpy as np
from einops import rearrange

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_stats(log_dir, stat_keys, version=None, *, verbose=False):
    # log_dir = f'../lightning_logs/{experiment}/lightning_logs'
    if version is None:
        # Gather latest stats
        versions = os.listdir(log_dir)
        versions = sorted([int(v.split('version_')[1]) for v in versions], reverse=True)
        if len(versions) > 1:
            print(f'Multiple versions in {log_dir}')
        version = versions[0]
        if verbose:
            print(f'Gathering stats for version {version}')

    log_dir = f'{log_dir}/version_{version}'
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    stats = {}
    for k in stat_keys:
        _, step_nums, vals = zip(*event_acc.Scalars(k))
        stats[k] = np.array(vals)
        stats[k + '_step'] = np.array(step_nums)

    return stats


def load_imputations(log_dir, version=None, *, verbose=False, filename_prefix='imputations_'):
    if version is None:
        # Gather latest stats
        versions = os.listdir(log_dir)
        versions = sorted([int(v.split('version_')[1]) for v in versions], reverse=True)
        if len(versions) > 1:
            print(f'Multiple versions in {log_dir}')
        version = versions[0]
        if verbose:
            print(f'Gathering stats for version {version}')

    log_dir = f'{log_dir}/version_{version}'
    files = [file for file in os.listdir(log_dir) if file.startswith(filename_prefix) and file.endswith('.npz')]
    file_idx = sorted([int(f.split(filename_prefix)[1].split('.npz')[0]) for f in files])
    files = [f'{filename_prefix}{f_idx}.npz' for f_idx in file_idx]

    all_imputations = []
    all_true = []
    all_masks = []
    for file in files:
        imps = np.load(os.path.join(log_dir, file))
        all_imputations.append(imps['imputations'])
        all_true.append(imps['true_X'])
        all_masks.append(imps['masks'])

    all_imputations = np.concatenate(all_imputations, axis=1)
    all_imputations = rearrange(all_imputations, 't n ... -> n t ...')
    all_true = np.concatenate(all_true, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    return all_imputations, all_true, all_masks
