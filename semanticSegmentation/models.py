import torch
import torch.nn as nn
import pytorch_lightning as pl

class VGG11(nn.Module):
  def __init__(self, cfg):
    self.features = nn.Sequential()
    for i in range(len(cfg)):
      pass