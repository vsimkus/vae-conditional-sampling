import os
import os.path

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.utilities.seed import seed_everything, reset_seed
from pytorch_lightning.profiler import AdvancedProfiler

from irwg.data import MissingDataModule
from irwg.models.resnet_classifier import ResNetClassifier
from irwg.utils.arg_utils import construct_experiment_subdir
from irwg.models import VDVAELightning, VAE, NVAELightning
from irwg.utils.test_step_base import TestBase


def build_argparser():
    parser = LightningArgumentParser('IRWG test experiment runner',
                                     parse_as_dict=False)

    # Add general arguments
    parser.add_argument("--seed_everything", type=int, required=True,
        help="Set to an int to run seed_everything with this value before classes instantiation",)
    parser.add_argument('--experiment_subdir_base', type=str, required=True,
        help='Experiment subdirectory.')

    # Add class arguments
    parser.add_lightning_class_args(MissingDataModule, 'data')
    # Note use `python test.py --model=irwg.models.VAE --print_config`
    # to print a config for a specific model class
    # parser.add_lightning_class_args(pl.LightningModule, 'model', subclass_mode=True)
    parser.add_argument('--model', type=str, required=True,
                        help=('Model to load.'))
    parser.add_argument('--model_path', type=str, required=False,
                        help=('Path to the model, except for VDVAE, where this is just the name of one of the pre-trained models'))
    parser.add_argument('--load_best', type=bool, default=None,
                        help=('If model_path is directoy use this flag to load the `epoch=*step=*.ckpt`. Throws error if there are multiple.'))
    parser.add_subclass_arguments(TestBase, 'test_class', fail_untyped=False, required=False)
    parser.add_lightning_class_args(pl.Trainer, 'trainer')

    parser.add_argument('--inception_model', type=str, required=False,
                        help=('Model to use for extracting features for e.g. Inception score or FID.'))
    parser.add_argument('--inception_model_path', type=str, required=False,
                        help=('Path to the feature extractor model weights.'))

    parser.add_argument('--classifier_model', type=str, required=False,
                        help=('Model to use for missing model.'))
    parser.add_argument('--classifier_model_path', type=str, required=False,
                        help=('Path to the classifier.'))

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

    # Get the instantiated data
    # datamodule = hparams.data
    datamodule = MissingDataModule(**hparams.data)

    inception_model = None
    if hparams.inception_model == 'ResNetClassifier':
        versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.inception_model_path.split('version_')[0])))
        if len(versions) > 1:
            print('More than one version is available:', versions, '. Loading ', versions[-1])
        version = versions[-1]
        model_path = hparams.inception_model_path.format(version)
        print('Inception model path:', model_path)

        inception_model = ResNetClassifier.load_from_checkpoint(checkpoint_path=model_path)

    classifier_model = None
    if hparams.classifier_model == 'ResNetClassifier':
        versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.classifier_model_path.split('version_')[0])))
        if len(versions) > 1:
            print('More than one version is available:', versions, '. Loading ', versions[-1])
        version = versions[-1]
        model_path = hparams.classifier_model_path.format(version)
        print('Classifer model path:', model_path)

        classifier_model = ResNetClassifier.load_from_checkpoint(checkpoint_path=model_path)

    def dynamic_import(name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    # Get the instantiated model
    # model = hparams.model
    if hparams.model == 'VDVAE':
        model = VDVAELightning(dataset=hparams.model_path)
    elif hparams.model == 'NVAE':
        TestClass = dynamic_import(hparams.test_class.class_path)
        tester = TestClass(**hparams.test_class.init_args)

        tester.set_datamodule(datamodule)

        model = tester

    elif hparams.model == 'VAE':
        versions = sorted(list(int(f.split('_')[1]) for f in os.listdir(hparams.model_path.split('version_')[0])))
        if len(versions) > 1:
            print('More than one version is available:', versions, '. Loading ', versions[-1])
        version = versions[-1]
        model_path = hparams.model_path.format(version)
        if os.path.isdir(model_path):
            models = os.listdir(model_path)
            if hparams.load_best is not None and hparams.load_best:
                models.remove('last.ckpt')
            if len(models) > 1:
                raise ValueError(f'Too many models in path: {model_path}')
            model_path = os.path.join(model_path, models[0])
        print('Model path:', model_path)

        TestClass = dynamic_import(hparams.test_class.class_path)
        tester = TestClass(**hparams.test_class.init_args)
        model = VAE.load_from_checkpoint(checkpoint_path=model_path)
        tester.set_model(model)
        model = tester

        tester.set_datamodule(datamodule)

        if inception_model is not None:
            tester.set_inception_model(inception_model)
        if classifier_model is not None:
            tester.set_classifier_model(classifier_model)

        # In case we want to generate data from the model
        datamodule.set_model(tester.model)
    elif hparams.model == 'MoG':
        def dynamic_import(name):
            components = name.split('.')
            mod = __import__(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod

        TestClass = dynamic_import(hparams.test_class.class_path)
        tester = TestClass(**hparams.test_class.init_args)

        tester.set_datamodule(datamodule)

        model = tester

        # In case we want to generate data from the model
        # datamodule.set_model(tester.model)
    else:
        raise NotImplementedError()

    if hparams.trainer.profiler is not None and hparams.trainer.profiler == 'advanced':
        print('Using profiler')
        profiler = AdvancedProfiler(
            filename='profiler.out'
        )
    elif hparams.trainer.profiler is not None:
        raise NotImplementedError()
    else:
        profiler = None
    del hparams.trainer['profiler']

    # Instantiate the trainer
    trainer_args = { **(hparams.trainer.as_dict()), 'default_root_dir': experiment_dir }
    trainer = pl.Trainer(**trainer_args, profiler=profiler)

    # The instantiation steps might be different for different models.
    # Hence we reset the seed before training such that the seed at the start of lightning setup is the same.
    reset_seed()

    # Begin testing
    trainer.test(model, dataloaders=datamodule)


if __name__ == '__main__':
    parser = build_argparser()

    # Parse arguments
    hparams = parser.parse_args()

    run(hparams)
