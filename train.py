import os
import os.path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

from irwg.data import MissingDataModule
from irwg.utils.arg_utils import construct_experiment_subdir


def build_argparser():
    parser = LightningArgumentParser('IRWG training experiment runner',
                                     parse_as_dict=False)

    # Add general arguments
    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)
    parser.add_argument('--experiment_subdir_base', type=str, required=True,
        help='Experiment subdirectory.')
    parser.add_argument('--add_checkpoint_callback', type=bool, default=False,
                        help='Adds additional checkpointing callback.')

    # Add class arguments
    parser.add_lightning_class_args(MissingDataModule, 'data')
    # Note use `python train.py --model=vgiwae.models.VAE --print_config`
    # to print a config for a specific model class
    parser.add_lightning_class_args(pl.LightningModule, 'model', subclass_mode=True)
    parser.add_argument('--pretrained_model_path', type=str, required=False, default=None,
                        help=('Path to the pretrained model.'))
    parser.add_argument('--freeze_generator', type=bool, required=False, default=False,
                        help=('Whether to freeze the generator parameters or not.'))
    parser.add_lightning_class_args(pl.Trainer, 'trainer')

    return parser

def run(hparams):
    # Set random seed
    # NOTE: this must be done before any class initialisation,
    # hence also before the call to parser.instantiate_classes()
    seed_everything(hparams.seed_everything, workers=True)

    # Construct the experiment directory
    experiment_subdir = construct_experiment_subdir(hparams)
    if hparams.trainer.default_root_dir is None:
        experiment_dir = f'./{experiment_subdir}'
    else:
        experiment_dir = f'{hparams.trainer.default_root_dir}/{experiment_subdir}'

    # Instantiate dynamic object classes
    hparams = parser.instantiate_classes(hparams)

    # Get the instantiated data
    datamodule = hparams.data

    # Get the instantiated model
    model = hparams.model
    if hparams.pretrained_model_path is not None:
        versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.pretrained_model_path.split('version_')[0])))
        if len(versions) > 1:
            print('More than one version is available:', versions, '. Loading ', versions[-1])
        version = versions[-1]
        pretrained_model_path = hparams.pretrained_model_path.format(version)
        if os.path.isdir(pretrained_model_path):
            models = os.listdir(pretrained_model_path)
            if hparams.load_best is not None and hparams.load_best:
                models.remove('last.ckpt')
            if len(models) > 1:
                raise ValueError(f'Too many models in path: {pretrained_model_path}')
            pretrained_model_path = os.path.join(pretrained_model_path, models[0])
        print('Pretrained model path:', pretrained_model_path)
        model.load_state_dict(torch.load(pretrained_model_path)['state_dict'])

        # In case we want to generate data from the model
        datamodule.set_model(model)
    if hparams.freeze_generator:
        model.freeze_generator_params()

    # Instantiate the trainer
    trainer_args = { **(hparams.trainer.as_dict()), "default_root_dir": experiment_dir }
    if hparams.add_checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              save_last=True,
                                              monitor="loss/val")
        if trainer_args['callbacks'] is not None:
            trainer_args['callbacks'].append(checkpoint_callback)
        else:
            trainer_args['callbacks'] = [checkpoint_callback]

    trainer = pl.Trainer(**trainer_args)

    # The instantiation steps might be different for different models.
    # Hence we reset the seed before training such that the seed at the start of lightning setup is the same.
    reset_seed()

    # Begin fitting
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
