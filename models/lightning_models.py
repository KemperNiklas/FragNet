import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class LightningModel(pl.LighningModule):
    def __init__(self, model, loss, acc, optimizer_parameters):
        super().__init__()
        self.model = model
        self.optimizer_parameters = optimizer_parameters
        self.loss = loss
        self.acc = acc
    
    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        x_hat = self.model(data)
        loss = self.loss(torch.squeeze(x_hat), data, "train")
        self.log("train_loss", loss, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")
    
    def _shared_eval(self, batch, batch_idx, prefix):
        data = batch
        x_hat = self.model(data)
        loss = self.loss(torch.squeeze(x_hat), data, prefix)
        acc = self.acc(torch.squeeze(x_hat), data, prefix)  
        self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": acc})
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_parameters)
        return optimizer


def classification_loss(xhat, data, mode: str):
    metric = torch.nn.CrossEntropyLoss()
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], data.y[mask])
    else:
        loss = metric(xhat, data.y)
    return loss

def classification_accuracy(xhat, data, mode: str):
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        acc = torch.mean((torch.argmax(xhat[mask], dim = 1) == data.y[mask]).int(), dtype = torch.float32)
    else:
        acc = torch.mean((torch.argmax(xhat, dim = 1) == data.y).int(), dtype = torch.float32)
    return acc

def regression_loss(xhat, data, mode: str):
    metric = F.mse_loss
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        loss = metric(xhat[mask], data.y[mask])
    else:
        loss = metric(xhat, data.y)
    return loss

def regression_acc(xhat, data, mode: str):
    if hasattr(data, f"{mode}_mask"):
        mask = getattr(data, f"{mode}_mask")
        acc = torch.mean((torch.round(xhat[mask]) == data.y[mask]).int(), dtype=torch.float32) 
    else:
        acc = torch.mean((torch.round(xhat) == data.y).int(), dtype=torch.float32) 
    return acc
