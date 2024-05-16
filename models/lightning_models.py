import math
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import (BinaryAUROC, BinaryPrecision,
                                         BinaryRecall,
                                         MultilabelAveragePrecision)


class LightningModel(pl.LightningModule):
    """
    LightningModel is a PyTorch Lighning wrapper for custom models.

    Args:
        model: The underlying model.
        loss: The loss function used for training and testing.
        acc: The accuracy metric used for evaluation.
        optimizer_parameters: A dictionary of parameters for the optimizer.
        scheduler_parameters: A dictionary of parameters for the learning rate scheduler (optional).
        additional_metric: Additional metric or list of additional metrics used for evaluation of validation and test steps (optional).
        ema_decay: The decay factor for the Exponential Moving Average (EMA) of model parameters (optional).
    """

    def __init__(self,
                 model,
                 loss,
                 acc,
                 optimizer_parameters,
                 scheduler_parameters=None,
                 additional_metric=None,
                 ema_decay=None):
        super().__init__()
        self.model = model
        self.optimizer_parameters = optimizer_parameters
        self.scheduler_parameters = scheduler_parameters
        self.loss = loss
        self.acc = acc
        self.additional_metric = additional_metric
        if ema_decay:
            self.ema = ExponentialMovingAverage(self.model.cuda().parameters(),
                                                decay=ema_decay)
        else:
            self.ema = None

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: The input data.

        Returns:
            The output of the model.
        """
        return self.model(data)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The loss value for the training step.
        """
        data = batch
        x_hat = self.model(data)
        loss = self.loss(torch.squeeze(x_hat), data, "train")
        self.log("train_loss",
                 loss,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=False,
                 batch_size=_calculate_batch_size(data))
        return loss

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        """
        Performs an optimizer step.

        Args:
            epoch: The current epoch.
            batch_idx: The index of the current batch.
            optimizer: The optimizer.
            optimizer_closure: The closure function for the optimizer.

        """
        optimizer.step(closure=optimizer_closure)
        if self.ema is not None:
            self.ema.update(self.model.parameters())

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        """
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        """
        self._shared_eval(batch, batch_idx, "test")

    def on_save_checkpoint(self, checkpoint):
        """
        Callback function called when saving a checkpoint that saves ema state.

        Args:
            checkpoint: The checkpoint dictionary.

        """
        checkpoint['state_dict'] = self.state_dict()
        if self.ema:
            with self.ema.average_parameters():
                checkpoint['ema_state_dict'] = deepcopy(self.state_dict())

    def on_train_epoch_end(self):
        """
        Callback function called at the end of each training epoch.

        """
        if self.scheduler_parameters:  # log learning rate
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr)

    def additional_metrics(self, prefix, xhat, data):
        """
        Computes additional metrics for evaluation.

        Args:
            prefix: The prefix used for logging, i.e. "val" or "test".
            xhat: The predicted output of the model.
            data: The input data.

        """
        if self.additional_metric:
            if isinstance(self.additional_metric, list):
                metrics = {}
                for metric in self.additional_metric:
                    name = metric.__name__
                    metrics[f"{prefix}_{name}"] = metric(
                        torch.squeeze(xhat), data, prefix)
                self.log_dict(metrics)
            else:
                metric = self.additional_metric
                name = metric.__name__
                self.log_dict({f"{prefix}_{name}": metric(torch.squeeze(
                    xhat), data, prefix)}, batch_size=_calculate_batch_size(data))

    def _shared_eval(self, batch, batch_idx, prefix):
        """
        Shared evaluation step for validation and test.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.
            prefix: The prefix used for logging, i.e. "val" or "test".

        """
        data = batch
        with torch.no_grad():
            if self.ema:
                with self.ema.average_parameters():
                    x_hat = self.model(data)
                    loss = self.loss(torch.squeeze(x_hat), data, prefix)
                    acc = self.acc(torch.squeeze(x_hat), data, prefix)
                    self.log_dict(
                        {
                            f"{prefix}_loss": loss,
                            f"{prefix}_acc": acc
                        },
                        batch_size=_calculate_batch_size(data),
                        on_epoch=True,
                        on_step=False,
                        prog_bar=False)
                    self.additional_metrics(prefix, x_hat, data)
            else:
                x_hat = self.model(data)
                loss = self.loss(torch.squeeze(x_hat), data, prefix)
                acc = self.acc(torch.squeeze(x_hat), data, prefix)
                self.log_dict({
                    f"{prefix}_loss": loss,
                    f"{prefix}_acc": acc
                },
                    batch_size=_calculate_batch_size(data))
                self.additional_metrics(prefix, x_hat, data)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        Available Schedulers are CosineWithWarmup and ReduceLROnPlateau (defaul).
        Optimizer is AdamW.

        Returns:
            The optimizer and learning rate scheduler.

        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      **self.optimizer_parameters)
        scheduler_parameters = {
            k: v
            for k, v in self.scheduler_parameters.items() if k != "name"
        }
        if "name" in self.scheduler_parameters and self.scheduler_parameters[
                "name"] == "cosine_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=scheduler_parameters["num_warmup_epochs"],
                num_training_steps=scheduler_parameters["max_epochs"],
                min_lr=scheduler_parameters.get("min_lr", 0.),
                min_lr_mode=scheduler_parameters.get("min_lr_mode", "rescale"))
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.scheduler_parameters:
            scheduler = ReduceLROnPlateau(optimizer,
                                          **self.scheduler_parameters)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }
        return optimizer


def _calculate_batch_size(data):
    if hasattr(data, "batch"):
        batch_size = torch.max(data.batch) + 1
    elif hasattr(data, "x_batch"):
        batch_size = torch.max(data.x_batch) + 1
    else:
        batch_size = 1
    return batch_size


def ce_loss(xhat, data, mode: str):
    metric = torch.nn.CrossEntropyLoss()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], data.y[mask])
    else:
        loss = metric(xhat, data.y)
    return loss


def bce_loss(xhat, data, mode: str):
    metric = torch.nn.BCEWithLogitsLoss()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], torch.squeeze(data.y[mask]).float())
    else:
        loss = metric(xhat, torch.squeeze(data.y).float())
    return loss


def binary_classification_accuracy(xhat, data, mode: str):
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        acc = torch.mean((torch.where(xhat[mask] < 0, 0,
                                      1) == data.y[mask]).float(),
                         dtype=torch.float32)
    else:
        acc = torch.mean((torch.where(xhat < 0, 0, 1) == data.y).float(),
                         dtype=torch.float32)
    return acc


def classification_accuracy(xhat, data, mode: str):
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        acc = torch.mean((torch.argmax(xhat[mask],
                                       dim=1) == data.y[mask]).float(),
                         dtype=torch.float32)
    else:
        acc = torch.mean((torch.argmax(xhat, dim=1) == data.y).float(),
                         dtype=torch.float32)
    return acc


def mse_loss(xhat, data, mode: str):
    metric = F.mse_loss
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], data.y[mask])
    else:
        loss = metric(xhat, data.y)
    return loss


def mae_loss(xhat, data, mode: str):
    metric = torch.nn.L1Loss()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], data.y[mask])
    else:
        loss = metric(xhat, data.y)
    return loss


def auroc(xhat, data, mode: str):
    metric = BinaryAUROC()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], torch.squeeze(data.y[mask]))
    else:
        loss = metric(xhat, torch.squeeze(data.y))
    return loss


def regression_acc(xhat, data, mode: str):
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        acc = torch.mean((torch.round(xhat[mask]) == data.y[mask]).float(),
                         dtype=torch.float32)
    else:
        acc = torch.mean((torch.round(xhat) == data.y).float(),
                         dtype=torch.float32)
    return acc


def average_multilabel_precision(xhat, data, mode: str):
    num_labels = xhat.size(1)
    metric = MultilabelAveragePrecision(num_labels=num_labels).to(xhat.device)
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], torch.squeeze(data.y[mask]).long())
    else:
        loss = metric(xhat, torch.squeeze(data.y).long())
    return loss


def regression_precision(xhat, data, mode: str):
    metric = BinaryPrecision().to(xhat.device)
    xhat_greater = (xhat >= 0.5).long()
    y_greater = torch.squeeze((data.y >= 1)).long()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat_greater[mask], y_greater[mask])
    else:
        loss = metric(xhat_greater, y_greater)
    return loss


def regression_recall(xhat, data, mode: str):
    metric = BinaryRecall().to(xhat.device)
    xhat_greater = (xhat >= 0.5).long()
    y_greater = torch.squeeze((data.y >= 1)).long()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat_greater[mask], y_greater[mask])
    else:
        loss = metric(xhat_greater, y_greater)
    return loss


def num_true_positives(xhat, data, mode: str):
    return torch.sum(torch.squeeze(data.y)).item()


def get_cosine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1,
                                    min_lr: float = 0.,
                                    min_lr_mode: str = "rescale"):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6,
                       float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        lr = max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        if min_lr > 0.:
            if min_lr_mode == "clamp":
                lr = max(min_lr / base_lr, lr)
            elif min_lr_mode == "rescale":  # "rescale lr"
                lr = (1 - min_lr / base_lr) * lr + min_lr / base_lr

        return lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
