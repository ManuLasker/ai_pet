from numpy.core.fromnumeric import prod
import pytorch_lightning as pl
import torch
import numpy as np
import torch.optim as optim
from torchvision import models
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)
    
class VG11BackBoneSegmentation(nn.Module):
    def __init__(self, requires_grad:bool = False):
        super().__init__()
        self.vgg_pretrained_features = models.vgg11(pretrained=True).features
        self.avgpool = models.vgg11(pretrained=True).avgpool
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.out_features = np.prod(self(torch.zeros(1, 3, 224, 224)).shape)
    
    def forward(self, x):
        x = self.vgg_pretrained_features(x)
        x = self.avgpool(x)
        return x
    
class VG16BackBoneSegmentation(nn.Module):
    def __init__(self, requires_grad:bool = False):
        super().__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.avgpool = models.vgg16(pretrained=True).avgpool
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.out_features = np.prod(self(torch.zeros(1, 3, 224, 224)).shape)


    def forward(self, x):
        x = self.vgg_pretrained_features(x)
        x = self.avgpool(x)
        return x

class SegmentationNaive(nn.Module):
    def __init__(self, img_shape=(224, 224),
                 requires_grad_backbone:bool = False,
                 backbone:int = 11):
        super().__init__()
        if backbone == 11:
            self.backbone = VG11BackBoneSegmentation(requires_grad=requires_grad_backbone)
        elif backbone == 16:
            self.backbone = VG16BackBoneSegmentation(requires_grad=requires_grad_backbone)
            
        self.decoder = nn.Sequential(
            Lambda(reshape_batch),
            nn.Linear(in_features=self.backbone.out_features,
                      out_features=np.prod(img_shape)),
        )
    
    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.decoder(x)
        return x

def reshape_batch(x:torch.Tensor):
    return x.view(x.size(0), -1)

class SegmentationFCN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x:torch.Tensor):
        pass
    
class SegmentationModule(pl.LightningModule):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam([filter(lambda p: p.requires_grad), self.parameters()])
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        batch = train_batch
        x, y = batch['source'], batch['mask']
        bs, _, _, _ = x.size()
        y_pred = self(x)
        loss = self.loss(y_pred, y.view(bs, -1))
        self.log('train_loss', loss, prog_bar=True, logger=False,
                on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        batch = val_batch
        x, y = batch['source'], batch['mask']
        bs, _, _, _ = x.size()
        y_pred = self(x)
        loss = self.loss(y_pred, y.view(bs, -1))
        self.log('val_loss', loss, prog_bar=True, logger=False,
                on_step=False, on_epoch=True)
    
class Predictor:
    model = None
    model_path = None
    
    @classmethod
    def set_config(cls, model_path:str):
        cls.model_path = model_path
        
    @classmethod
    def load_model(cls, model_path:str):
        if cls.model is None:
            cls.model = torch.jit.load(model_path)
        return cls.model
    
    @classmethod
    def predict(cls, x: torch.Tensor):
        model = cls.load_model(cls.model_path)
        model.eval()
        with torch.no_grad():
            prediction: torch.Tensor = model(x)
        return prediction.softmax(dim=1).argmax(dim=1)