from numpy.core.fromnumeric import prod
import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision import models
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)
    
class VG11BackBoneSegmentation(nn.Module):
    def __init__(self, img_shape = (224, 224)):
        super().__init__()
        self.vgg_pretrained_features = models.vgg11(pretrained=True).features
        self.avgpool = models.vgg11(pretrained=True).avgpool
        self.decoder = nn.Sequential(
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(in_features=4096, out_features=np.prod(img_shape))
        )
    
    def forward(self, x):
        x = self.vgg_pretrained_features(x)
        x = self.avgpool(x)
        x = self.decoder(x)
        return x
    
class SegmentationModule(pl.LightningModule):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.model = VG11BackBoneSegmentation(img_shape=self.img_shape)
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return None
    
    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

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