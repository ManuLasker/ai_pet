import json
import os
from pathlib import Path
from torch.functional import norm
from tqdm.auto import tqdm
from typing import Dict, Tuple
from torch.utils.data import Dataset
from .image import load_image

def _get_name_file(full_path: Path):
    return full_path.name
        
class ImageDataBlending(Dataset):
    def __init__(self, root_path, normalize:bool = True):
        # Read all the files
        path_files = [Path(os.path.join(root_path, path)) 
                 for path in os.listdir(root_path)]
        # Initialize files needed for blend
        self.image_data = {"dims": {},
                           "source": {},
                           "target": {},
                           "mask": {}}
        self.normalize = normalize
        # Getting files paths
        for path in tqdm(path_files, total=len(path_files),
                         desc="Getting Files"):
            name_file, index = _get_name_file(path).split("_")
            index = int(index.split(".")[0])
            self.image_data[name_file][index] = path
                
    def __len__(self) -> int:
        return len(self.image_data["dims"])
    
    def __getitem__(self, index) -> Dict:
        source = load_image(list(self.image_data["source"].values())[index],
                            normalize=self.normalize)
        target = load_image(list(self.image_data["target"].values())[index],
                            normalize=self.normalize)
        
        dims = json.load(open(list(self.image_data["dims"].values())[index], "r"))
        mask = load_image(list(self.image_data["mask"].values())[index], is_mask=True,
                          normalize=self.normalize)
        
        return {"source": source, "target": target,
                "dims": [dims["h0"], dims["h0"] + mask.shape[1],
                         dims["w0"], dims["w0"] + mask.shape[2]],
                "mask": mask}