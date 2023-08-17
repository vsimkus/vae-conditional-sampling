from typing import List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from irwg.models.vae_resnet import ResNetEncoder


class ResNetClassifier(pl.LightningModule):
    def __init__(self, learning_rate: float,
                 input_shape: Tuple[int, int, int], layers: List[int], layer_widths: List[int],
                 num_classes: int,
                 first_conv: bool=False, maxpool1: bool=False, dropout_prob:float=0.):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResNetEncoder(input_shape=input_shape,
                                   layers=layers,
                                   layer_widths=layer_widths,
                                   latent_dim=num_classes,
                                   first_conv=first_conv,
                                   maxpool1=maxpool1,
                                   dropout_prob=dropout_prob)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        return optimizer

    def forward(self, x):
        out = self.model(x)
        feats = out
        out = F.log_softmax(out, dim=-1)

        return out, feats

    def training_step(self, batch):
        X, target = batch[0], batch[2]

        output, _ = self.forward(X)

        loss = F.nll_loss(output, target)

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        acc = 100 * correct / len(X)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('acc/train', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            output, _ = self.forward(X)

            loss = F.nll_loss(output, target)

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('acc/val', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def test_step(self, batch):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            output, _ = self.forward(X)

            loss = F.nll_loss(output, target)

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/test', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('acc/test', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return loss
