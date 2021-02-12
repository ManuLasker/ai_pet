import torch
import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
from torchvision import models
import warnings
warnings.filterwarnings("ignore")
from torchvision.transforms import Normalize


class BSSModel(pl.LightningModule):
    """Binary semantic segmentation model using vgg16 pretrained weights"""
    def __init__(self, out_features):
        super(BSSModel, self).__init__()
        self.encoder = models.vgg16_bn(pretrained=True)
        
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, out_features, bias=True)
        )
        self.loss = nn.BCEWithLogitsLoss()
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr = 0.00003, weight_decay = 0)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=False,
                on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, logger=False,
                on_step=False, on_epoch=True)
        
def get_transforms():
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalize = Normalize(mean=mean, std=std)
    return normalize

def get_trainer():
    return pl.Trainer(gpus=0, num_nodes=1, max_epochs=150)