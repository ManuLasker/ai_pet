import json
import os
import torch
from pathlib import Path
from torch.functional import norm
from tqdm.auto import tqdm
from typing import Dict, Tuple
from torch.utils.data import Dataset
from .image import load_image

def _get_name_file(full_path: Path):
    return full_path.name
        
class ImageDataBlending(Dataset):
    def __init__(self, root_path,
                 device:torch.device = torch.device('cuda') 
                                    if torch.cuda.is_available() else torch.device('cpu'),
                 normalize:bool = True):
        # Read all the files
        path_files = [Path(os.path.join(root_path, path)) 
                 for path in os.listdir(root_path)]
        # Initialize files needed for blend
        self.image_data = {"dims": {},
                           "source": {},
                           "target": {},
                           "mask": {}}
        self.normalize = normalize
        self.device = device
        # Getting files paths
        for path in tqdm(path_files, total=len(path_files),
                         desc="Getting Files"):
            name_file, index = _get_name_file(path).split("_")
            index = int(index.split(".")[0])
            self.image_data[name_file][index] = path
                
    def __len__(self) -> int:
        return len(self.image_data["dims"])
    
    def __getitem__(self, index) -> Dict:
        # Assert error in order to stop the indexing
        list(self.image_data["source"].values())[index]
        
        # Load images
        source = load_image(self.image_data["source"][index + 1],
                            normalize=self.normalize, device=self.device)
        target = load_image(self.image_data["target"][index + 1],
                            normalize=self.normalize, device=self.device)
        
        dims = json.load(open(self.image_data["dims"][index + 1], "r"))
        mask = load_image(self.image_data["mask"][index + 1], is_mask=True,
                          normalize=self.normalize, device=self.device)
        
        return {"source": source, "target": target,
                "dims": [dims["h0"], dims["h0"] + mask.shape[1],
                         dims["w0"], dims["w0"] + mask.shape[2]],
                "mask": mask}