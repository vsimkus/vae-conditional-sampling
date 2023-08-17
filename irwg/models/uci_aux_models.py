import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy import stats
from einops import rearrange

from irwg.models.neural_nets import ResidualFCNetwork


class ResNetMulticlassClassifier(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 input_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 num_classes: int,
                 dropout_prob: float=0.):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResidualFCNetwork(input_dim=input_dim,
                                       output_dim=num_classes,
                                       num_residual_blocks=num_residual_blocks,
                                       residual_block_dim=residual_block_dim,
                                       dropout_probability=dropout_prob,
                                       use_batch_norm=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        return optimizer

    def forward(self, x):
        out = self.model(x)

        return out

    # def bootstrap_empirical_ci(self, X, Y, num_boostrap_resamples=1000):
    #     """
    #     Bootstrap empirical confidence intervals.
    #     """
    #     raise NotImplementedError()

    def model_output_to_prediction(self, out):
        return F.softmax(out, dim=-1)

    def confidence_interval_target_coverage(self, cis, targets):
        # TODO: this is not really correct, but there is no known correct way to do this.

        # Get the CI of the target class
        target_cis = cis[torch.arange(len(cis)), targets.long(), :]

        # If the CI of the target class is greater than the CI of the other classes,
        # including the lower and upper bound, then we consider that the target is covered.
        coverage = rearrange([cis[:, :, 0] <= target_cis[:, None, 0], cis[:, :, 1] <= target_cis[:, None, 1]], 'ci ... -> ... ci')
        coverage = torch.all(torch.all(coverage, dim=-1), dim=-1)

        return coverage

    def prediction_confidence_interval(self, X, *, ci_level=0.95):
        # NOTE (Option 1) Assuming that the class probabilities are independent, we compute
        # the confidence intervals for each class probability separately.
        logit_predictions = self(X)
        prob_predictions = self.model_output_to_prediction(logit_predictions)

        ci = torch.quantile(prob_predictions, q=torch.tensor([(1-ci_level)/2, 1-(1-ci_level)/2]), dim=-2)

        # # Using Goodman method, copied from https://www.statsmodels.org/dev/_modules/statsmodels/stats/proportion.html#multinomial_proportions_confint
        # # NOTE (Option 2): should probably sample, but to align with other methods we just use argmax
        # # predictions = torch.argmax(prob_predictions, dim=-1)
        # # NOTE (Option 3): Sampling should be better then the argmax above
        # gumbel_noise = -torch.log(-torch.log(torch.rand(size=logit_predictions.shape)))
        # logits_with_noise = logit_predictions + gumbel_noise
        # predictions = torch.argmax(logits_with_noise, dim=-1)

        # counts = torch.nn.functional.one_hot(predictions).sum(dim=-2)
        # n = counts.sum()
        # k = len(counts)
        # proportions = counts / n

        # chi2 = stats.chi2.ppf(ci_level / k, 1)
        # delta = chi2 ** 2 + (4 * n * proportions * chi2 * (1 - proportions))
        # ci = rearrange([- torch.sqrt(delta), torch.sqrt(delta)], 'ci b classes -> ci b classes')
        # ci = ((2 * n * proportions[None, :, :] + chi2 + ci) / (2 * (chi2 + n)))

        return ci

    def training_step(self, batch):
        X, target = batch[0], batch[2]
        target = target.to(torch.long)

        logits = self.forward(X)

        loss = F.cross_entropy(logits, target)

        pred = logits.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        acc = 100 * correct / len(X)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/train', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, target = batch[0], batch[2]
        target = target.to(torch.long)

        with torch.inference_mode():
            logits = self.forward(X)

            loss = F.cross_entropy(logits, target)

            pred = logits.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/val', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch):
        X, target = batch[0], batch[2]
        target = target.to(torch.long)

        with torch.inference_mode():
            logits = self.forward(X)

            loss = F.cross_entropy(logits, target)

            pred = logits.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/test', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/test', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

class ResNetBinaryClassifier(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 input_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 dropout_prob: float=0.):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResidualFCNetwork(input_dim=input_dim,
                                       output_dim=1,
                                       num_residual_blocks=num_residual_blocks,
                                       residual_block_dim=residual_block_dim,
                                       dropout_probability=dropout_prob,
                                       use_batch_norm=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        return optimizer

    def model_output_to_prediction(self, out):
        return torch.sigmoid(out)

    def confidence_interval_target_coverage(self, cis, targets):
        #TODO maybe threshold by avg of targets?
        threshold = 0.5
        return ((cis[:,0] <= threshold) & (targets == 0)) | ((cis[:,1] >= threshold) & (targets == 1))

    def prediction_confidence_interval(self, X, *, ci_level=0.95):
        prob_predictions = self.model_output_to_prediction(self(X))

        ci = torch.quantile(prob_predictions, q=torch.tensor([(1-ci_level)/2, 1-(1-ci_level)/2]), dim=-1)

        return ci

    def forward(self, x):
        out = self.model(x).squeeze(-1)

        return out

    def training_step(self, batch):
        X, target = batch[0], batch[2]

        logit = self.forward(X)

        loss = F.binary_cross_entropy_with_logits(logit, target)

        pred = torch.sigmoid(logit.data) > 0.5
        correct = pred.eq(target.data.view_as(pred)).sum()
        acc = 100 * correct / len(X)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/train', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            logit = self.forward(X)

            loss = F.binary_cross_entropy_with_logits(logit, target)

            pred = torch.sigmoid(logit.data) > 0.5
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/val', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            logit = self.forward(X)

            loss = F.binary_cross_entropy_with_logits(logit, target)

            pred = torch.sigmoid(logit.data) > 0.5
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/test', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/test', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

class ResNetOrdinalClassifier(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 input_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 num_classes: int,
                 dropout_prob: float=0.):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResidualFCNetwork(input_dim=input_dim,
                                       output_dim=1,
                                       num_residual_blocks=num_residual_blocks,
                                       residual_block_dim=residual_block_dim,
                                       dropout_probability=dropout_prob,
                                       use_batch_norm=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        return optimizer

    def model_output_to_prediction(self, out):
        return out*self.hparams.num_classes

    def confidence_interval_target_coverage(self, cis, targets):
        cis = cis.floor()
        return (targets >= cis[:, 0]) * (targets <= cis[:,1])

    def prediction_confidence_interval(self, X, *, ci_level=0.95):
        predictions = self.model_output_to_prediction(self(X))

        ci = torch.quantile(predictions, q=torch.tensor([(1-ci_level)/2, 1-(1-ci_level)/2]), dim=-1)

        return ci

    def forward(self, x):
        out = self.model(x).squeeze(-1)

        # Force the output to be in the range (0, 1)
        out = torch.sigmoid(out)

        return out

    def ordinal_classification_loss(self, ord_predictor, targets):
        bin_size = 1/self.hparams.num_classes

        targets = targets.float()
        targets = targets / self.hparams.num_classes + bin_size / 2

        # Using MSE loss in this transformed domain.
        return F.mse_loss(ord_predictor, targets)

    def predictions_to_classes(self, ord_predictor):
        bin_size = 1/self.hparams.num_classes

        prediction = (ord_predictor - bin_size / 2) * self.hparams.num_classes
        prediction = torch.round(prediction).to(torch.int64)

        return prediction

    def training_step(self, batch):
        X, target = batch[0], batch[2]

        ord_predictor = self.forward(X)

        loss = self.ordinal_classification_loss(ord_predictor, target)

        pred = self.predictions_to_classes(ord_predictor)
        correct = pred.eq(target.data.view_as(pred)).sum()
        acc = 100 * correct / len(X)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/train', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            ord_predictor = self.forward(X)

            loss = self.ordinal_classification_loss(ord_predictor, target)

            pred = self.predictions_to_classes(ord_predictor)
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/val', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            ord_predictor = self.forward(X)

            loss = self.ordinal_classification_loss(ord_predictor, target)

            pred = self.predictions_to_classes(ord_predictor)
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/test', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/test', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


class ResNetPositiveRegressor(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 input_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 dropout_prob: float=0.):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResidualFCNetwork(input_dim=input_dim,
                                       output_dim=1,
                                       num_residual_blocks=num_residual_blocks,
                                       residual_block_dim=residual_block_dim,
                                       dropout_probability=dropout_prob,
                                       use_batch_norm=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        return optimizer

    def model_output_to_prediction(self, out):
        return out

    def confidence_interval_target_coverage(self, cis, targets):
        return (targets >= cis[:, 0]) * (targets <= cis[:,1])

    def prediction_confidence_interval(self, X, *, ci_level=0.95):
        predictions = self.model_output_to_prediction(self(X))

        ci = torch.quantile(predictions, q=torch.tensor([(1-ci_level)/2, 1-(1-ci_level)/2]), dim=-1)

        return ci

    def forward(self, x):
        out = self.model(x).squeeze(-1)

        # Make sure the output is positive
        out = F.softplus(out)

        return out

    def training_step(self, batch):
        X, target = batch[0], batch[2]

        prediction = self.forward(X)

        loss = F.mse_loss(prediction, target)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            prediction = self.forward(X)

            loss = F.mse_loss(prediction, target)

        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            prediction = self.forward(X)

            loss = F.mse_loss(prediction, target)

        self.log('loss/test', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


class ResNetDensityRatioBinaryClassifier(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 input_dim: int,
                 num_residual_blocks: int,
                 residual_block_dim: int,
                 use_lr_scheduler: bool = False,
                 max_scheduler_steps: int = -1,
                 dropout_prob: float=0.):
        super().__init__()
        self.save_hyperparameters()

        self.model = ResidualFCNetwork(input_dim=input_dim,
                                       output_dim=1,
                                       num_residual_blocks=num_residual_blocks,
                                       residual_block_dim=residual_block_dim,
                                       dropout_probability=dropout_prob,
                                       use_batch_norm=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        opts = {
            'optimizer': optimizer
        }

        if self.hparams.use_lr_scheduler:
            max_steps = self.hparams.max_scheduler_steps if self.hparams.max_scheduler_steps != -1 else self.trainer.estimated_stepping_batches
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, eta_min=1e-9, last_epoch=-1)

            opts['lr_scheduler'] = {
                'scheduler': sched,
                'interval': 'step',
                'frequency': 1,
            }

        return opts

    def forward(self, x):
        out = self.model(x).squeeze(-1)

        return out

    def training_step(self, batch):
        X, target = batch[0], batch[1]

        logit = self.forward(X)

        loss = F.binary_cross_entropy_with_logits(logit, target)

        prob = torch.sigmoid(logit.data)
        pred = prob > 0.5
        correct = pred.eq(target.data.view_as(pred)).sum()
        acc = 100 * correct / len(X)

        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/train', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Avg loss corresponds to an approximate Jensen-Shannon divergence up to constants: loss = -2 * JS(p||q) + log(4)
        jsd = -0.5*(loss - torch.log(torch.tensor(4)))
        self.log('jsd/train', jsd, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # logit corresponds to an approximate density log-ratio, so compute avg ratio
        kld_forward = (logit * target).sum() / target.sum()
        not_target = 1 - target
        kld_reverse = -(logit * not_target).sum() / not_target.sum()
        self.log('kld_forward/train', kld_forward, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld_reverse/train', kld_reverse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            logit = self.forward(X)

            loss = F.binary_cross_entropy_with_logits(logit, target)

            pred = torch.sigmoid(logit.data) > 0.5
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/val', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Avg loss corresponds to an approximate Jensen-Shannon divergence up to constants: loss = -2 * JS(p||q) + log(4)
        jsd = -0.5*(loss - torch.log(torch.tensor(4)))
        self.log('jsd/val', jsd, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # logit corresponds to an approximate density log-ratio, so compute avg ratio
        kld_forward = (logit * target).sum() / target.sum()
        not_target = 1 - target
        kld_reverse = -(logit * not_target).sum() / not_target.sum()
        self.log('kld_forward/val', kld_forward, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld_reverse/val', kld_reverse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch):
        X, target = batch[0], batch[2]

        with torch.inference_mode():
            logit = self.forward(X)

            loss = F.binary_cross_entropy_with_logits(logit, target)

            pred = torch.sigmoid(logit.data) > 0.5
            correct = pred.eq(target.data.view_as(pred)).sum()
            acc = 100 * correct / len(X)

        self.log('loss/test', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/test', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Avg loss corresponds to an approximate Jensen-Shannon divergence up to constants: loss = -2 * JS(p||q) + log(4)
        jsd = -0.5*(loss - torch.log(torch.tensor(4)))
        self.log('jsd/test', jsd, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # logit corresponds to an approximate density log-ratio, so compute avg ratio
        kld_forward = (logit * target).sum() / target.sum()
        not_target = 1 - target
        kld_reverse = -(logit * not_target).sum() / not_target.sum()
        self.log('kld_forward/test', kld_forward, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kld_reverse/test', kld_reverse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
