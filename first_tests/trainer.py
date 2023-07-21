import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class GraphNet(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.metric = F.mse_loss

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        loss = self.metric(x_hat[data.train_mask], data.y[data.train_mask])
        self.log("training_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        data = batch
        mask = data.val_mask if prefix == "val" else data.test_mask
        x_hat = torch.squeeze(self.model(data))
        loss = self.metric(x_hat[mask], data.y[mask])
        correct = torch.round(x_hat[mask]) == data.y[mask]  # Check against ground-truth labels.
        acc = int(correct.sum()) / mask.sum()  # Derive ratio of correct predictions.
        self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": acc})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        return optimizer

class GraphNetNodePredictions(pl.LightningModule):
    def __init__(self, model, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.model = model
        self.metric = F.mse_loss
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        loss = self.metric(x_hat, data.y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        loss = self.metric(x_hat, data.y)
        correct = torch.round(x_hat) == data.y  # Check against ground-truth labels.
        acc = int(correct.sum()) / x_hat.size(0)  # Derive ratio of correct predictions.
        self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": acc})
        self.log_dict({"triangle_attention": self.model.substructures_attention[0], "square_attention": self.model.substructures_attention[1]})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
class GraphNetGraphPredictions(pl.LightningModule):
    def __init__(self, model, lr = 0.01, weight_decay = 5e-4):
        super().__init__()
        self.model = model
        self.metric = F.mse_loss
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        y = torch.squeeze(data.y)
        loss = self.metric(x_hat, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        y = torch.squeeze(data.y)
        loss = self.metric(x_hat, y)
        correct = torch.mean((torch.round(x_hat) == y).int(), dtype=torch.float32)  # Check against ground-truth labels.
        self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": correct})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
class GraphNetGraphClassification(pl.LightningModule):
    def __init__(self, model, lr = 0.01, weight_decay = 5e-4):
        super().__init__()
        self.model = model
        self.metric = torch.nn.CrossEntropyLoss()
        self.lr = 0.01
        self.weight_decay = 5e-4

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        y = data.y
        loss = self.metric(x_hat, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        data = batch
        x_hat = torch.squeeze(self.model(data))
        y = data.y
        loss = self.metric(x_hat, y)
        correct = torch.mean(torch.argmax(x_hat, dim=1) == y, dtype=torch.float32)  # Check against ground-truth labels.
        self.log_dict({f"{prefix}_loss": loss, f"{prefix}_acc": correct})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer