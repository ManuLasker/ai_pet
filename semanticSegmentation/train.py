import argparse
from models import BSSModel, get_transforms, get_trainer
from utils.dataset import create_dataloaders, hyp
import torch

path = "../data"

if __name__ == "__main__":
    transforms = get_transforms()
    train_dataloader, val_dataloader = create_dataloaders(path=path, transforms=transforms)
    out_features = hyp['img_size']**2
    model = BSSModel(out_features)
    trainer = get_trainer()
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("models/model.ckpt")
    script = model.to_torchscript()
    torch.jit.save(script, "models/model.pt")
