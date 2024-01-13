import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torchmetrics.classification import BinaryAUROC, MultilabelAveragePrecision
from torch_ema import ExponentialMovingAverage
from typing import Optional, Callable
from copy import deepcopy

class LightningModel(pl.LightningModule):
    def __init__(self, model, loss, acc, optimizer_parameters, scheduler_parameters = None, additional_metric = None, ema_decay = None):
        super().__init__()
        self.model = model
        self.optimizer_parameters = optimizer_parameters
        self.scheduler_parameters = scheduler_parameters
        self.loss = loss
        self.acc = acc
        self.additional_metric = additional_metric
        if ema_decay:
            self.ema = ExponentialMovingAverage(self.model.cuda().parameters(), decay=ema_decay)
        else:
            self.ema = None
    
    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        x_hat = self.model(data)
        loss = self.loss(torch.squeeze(x_hat), data, "train")
        self.log("train_loss", loss, on_epoch=False, batch_size = _calculate_batch_size(data))
        return loss

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        optimizer.step(closure=optimizer_closure)
        if self.ema is not None:
            self.ema.update(self.model.parameters())
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.state_dict()
        if self.ema:
            with self.ema.average_parameters():
                checkpoint['ema_state_dict'] = deepcopy(self.state_dict())
    
    def on_train_epoch_end(self):
        if self.scheduler_parameters: # log learning rate
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr)
    
    def _shared_eval(self, batch, batch_idx, prefix):
        data = batch
        with torch.no_grad():
            if self.ema:
                with self.ema.average_parameters():
                    x_hat = self.model(data)
                    loss = self.loss(torch.squeeze(x_hat), data, prefix)
                    acc = self.acc(torch.squeeze(x_hat), data, prefix)
                    self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": acc}, batch_size = _calculate_batch_size(data))
                    if self.additional_metric:
                        metric = self.additional_metric
                        name = self.additional_metric.__name__
                        self.log_dict({f"{prefix}_{name}": metric(torch.squeeze(x_hat), data, prefix)}, batch_size = _calculate_batch_size(data))
            else:
                x_hat = self.model(data)
                loss = self.loss(torch.squeeze(x_hat), data, prefix)
                acc = self.acc(torch.squeeze(x_hat), data, prefix)
                self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": acc}, batch_size = _calculate_batch_size(data))
                if self.additional_metric:
                    metric = self.additional_metric
                    name = self.additional_metric.__name__
                    self.log_dict({f"{prefix}_{name}": metric(torch.squeeze(x_hat), data, prefix)}, batch_size = _calculate_batch_size(data))
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_parameters)
        if self.scheduler_parameters:
            scheduler = ReduceLROnPlateau(optimizer, **self.scheduler_parameters)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
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
        acc = torch.mean((torch.where(xhat[mask] < 0, 0, 1) == data.y[mask]).float(), dtype = torch.float32)
    else:
        acc = torch.mean((torch.where(xhat < 0, 0, 1) == data.y).float(), dtype = torch.float32)
    return acc

def classification_accuracy(xhat, data, mode: str):
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        acc = torch.mean((torch.argmax(xhat[mask], dim = 1) == data.y[mask]).float(), dtype = torch.float32)
    else:
        acc = torch.mean((torch.argmax(xhat, dim = 1) == data.y).float(), dtype = torch.float32)
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
        acc = torch.mean((torch.round(xhat[mask]) == data.y[mask]).float(), dtype=torch.float32) 
    else:
        acc = torch.mean((torch.round(xhat) == data.y).float(), dtype=torch.float32) 
    return acc

def average_multilabel_precision(xhat, data, mode: str):
    num_labels = xhat.size(1)
    metric = MultilabelAveragePrecision(num_labels = num_labels)
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], torch.squeeze(data.y[mask]).long())
    else:
        loss = metric(xhat, torch.squeeze(data.y).long())
    print(loss)
    return loss
