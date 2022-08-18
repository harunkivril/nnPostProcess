import pytorch_lightning as pl
import torch.nn as nn
import torch
from datetime import datetime, timedelta

AVAIL_GPUS = min(1, torch.cuda.device_count())

class ModelBase(pl.LightningModule):
    def __init__(
        self,
        optimizer_name,
        learning_rate,
        weight_decay,
        test_prediction_prefix,
        test_start_year,
        loss_function_name="MSE",
        **kwargs
    ):

        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if loss_function_name == "MSE":
            self.loss_fn = nn.functional.mse_loss
        else:
            print("WARNING: Unknown loss function seting it to MSE")
            self.loss_fn = nn.functional.mse_loss


        self.test_prediction_prefix = test_prediction_prefix
        self.era5_dims = (3, 5, 9, 17, 19)
        self.gefs_dims = (5, 7, 17, 19)
        self.in_dim = 5*7*17*19
        self.out_dim = 3*5*9*17*19
        self.test_start = datetime(test_start_year, 1, 1)
        self.optimizer_name = optimizer_name

    def training_step(self, batch, batch_idx):
        x = batch["gefs"]
        y = batch["era5"]
        values = self(x)
        loss = self.loss_fn(values, y)
        self.log("Train Loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["gefs"]
        y = batch["era5"]
        values = self(x)
        loss = self.loss_fn(values, y)
        self.log("Val Loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        save_day = self.test_start + timedelta(days=batch_idx)
        save_day = save_day.strftime("%Y-%m-%d")
        save_path = f"{self.test_prediction_prefix}/{save_day}.pt"
        x = batch["gefs"]
        y = batch["era5"]
        values = self(x)
        torch.save(values.cpu(), save_path)
        loss = self.loss_fn(values, y)
        gefs_common = x[:, 1:, :, :, :]
        era5_common = y[:, 0, 1:, [0,1,2,3,4,6,8], :, :]
        values_common = values[:, 0, 1:, [0,1,2,3,4,6,8], :, :]
        gefs_loss = self.loss_fn(gefs_common, era5_common)
        reduced_test_loss = self.loss_fn(values_common, era5_common)
        self.log("Test MSE", loss, prog_bar=True)
        self.log("GEFS MSE", gefs_loss, prog_bar=True)
        self.log("Reduced Test MSE", reduced_test_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name=="Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            print("WARNING: Optimizer name is not valid using Adam")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return optimizer
