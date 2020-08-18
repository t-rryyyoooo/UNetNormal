from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from torch import nn
from .model import UNetModel
from .dataset import UNetDataset
from .transform import UNetTransform
from torch.utils.data import DataLoader
from .utils import DICE
from .loss import WeightedCategoricalCrossEntropy

class UNetSystem(pl.LightningModule):
    def __init__(self, dataset_path, criteria, in_channel, num_class, learning_rate, batch_size, checkpoint, num_workers):
        super(UNetSystem, self).__init__()
        use_cuda = torch.cuda.is_available() and True
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataset_path = dataset_path
        self.num_class = num_class
        self.model = UNetModel(in_channel, self.num_class).to(self.device, dtype=torch.float)
        self.criteria = criteria
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint = checkpoint
        self.num_workers = num_workers
        self.DICE = DICE(self.num_class, self.device)
        self.loss = WeightedCategoricalCrossEntropy(device=self.device)

    def forward(self, x):
        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        image = image.to(self.device, dtype=torch.float)
        label = label.to(self.device, dtype=torch.long)

        pred = self.forward(image).to(self.device)

        """ Onehot for loss. """
        pred_argmax = pred.argmax(dim=1)
        label_onehot = torch.eye(self.num_class)[label].to(self.device).permute((0, 4, 1, 2, 3))

        dice = self.DICE.compute(label, pred_argmax)
        loss = self.loss(pred, label_onehot)


        tensorboard_logs = {
                "train_loss" : loss, 
                "dice" : dice
                }
        progress_bar = {
                "dice" : dice
                }
        
        return {"loss" : loss, "log" : tensorboard_logs, "progress_bar" : progress_bar}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        image = image.to(self.device, dtype=torch.float)
        label = label.to(self.device, dtype=torch.long)

        pred = self.forward(image).to(self.device)

        """ Onehot for loss. """
        pred_argmax = pred.argmax(dim=1)
        label_onehot = torch.eye(self.num_class)[label].to(self.device).permute((0, 4, 1, 2, 3))

        dice = self.DICE.compute(label, pred_argmax)
        loss = self.loss(pred, label_onehot)

        tensorboard_logs = {
                "val_loss" : loss, 
                "val_dice" : dice, 
                }
        progress_bar= {
                "val_loss" : loss, 
                "val_dice" : dice
                }
 
        return {"val_loss" : loss, "log" : tensorboard_logs}#, "progress_bar" : progress_bar}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_dice = torch.stack([x["log"]["val_dice"] for x in outputs]).mean()

        self.checkpoint(avg_loss.item(), self.model)

        tensorboard_logs = {
                "val_loss" : avg_loss,
                "val_dice" : avg_dice, 
                }
        progress_bar = {
                "val_dice" : avg_dice
                }


        return {"avg_val_loss" : avg_loss, "log" : tensorboard_logs, "progress_bar" : progress_bar}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        translate = 0.0
        rotate = 0
        shear = 0.0
        scale = 0.0
        batch_size = self.batch_size

        train_dataset = UNetDataset(
                dataset_path = self.dataset_path, 
                phase = "train", 
                criteria = self.criteria,
                transform = UNetTransform(self.num_class, translate, rotate, shear, scale)
                )

        train_loader = DataLoader(
                train_dataset,
                shuffle=True, 
                batch_size = self.batch_size, 
                num_workers = self.num_workers
                )

        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = UNetDataset(
                dataset_path = self.dataset_path, 
                phase = "val", 
                criteria = self.criteria,
                transform = UNetTransform(self.num_class)
                )

        val_loader = DataLoader(
                val_dataset, 
                batch_size = self.batch_size,
                num_workers = self.num_workers
                )

        return val_loader








