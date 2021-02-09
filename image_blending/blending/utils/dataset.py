import json
import os
import torch
from pathlib import Path
from torch.functional import norm
from tqdm.auto import tqdm
from typing import Dict, Tuple
from torch.utils.data import Dataset
from .image import load_image, _numpy, prepare_images_arrays

IMG_EXTENSIONS = ['jpg', 'png', 'jpeg']
def _get_name_file(full_path: Path):
    return full_path.name
        

class ImageDataBlending(Dataset):
    def __init__(self, blending_dir, source_dir = None, target_dir = None,
                 save = True, device:torch.device = torch.device('cuda') 
                                    if torch.cuda.is_available() 
                                    else torch.device('cpu')):
        # Initialize files needed for blend
        self.image_data = {"dims": {},
                           "source": {},
                           "target": {},
                           "mask": {}}
        cache = source_dir is None and target_dir is None
        # Cache files or recache
        if cache:
            # Read all the files
            path_files = [Path(os.path.join(blending_dir, path)) 
                    for path in os.listdir(blending_dir) if path.split(".")[-1] in IMG_EXTENSIONS]
            
            # Getting files paths
            for path in tqdm(path_files, total=len(path_files),
                         desc="Getting Files"):
                name_file, index = _get_name_file(path).split("_")
                index = int(index.split(".")[0])
                self.image_data[name_file][index] = path
        else:
            source_files = [Path(os.path.join(source_dir, path))
                            for path in os.listdir(source_dir) if path.split(".")[-1] in IMG_EXTENSIONS]
            target_files = [Path(os.path.join(target_dir, path))
                            for path in os.listdir(target_dir) if path.split(".")[-1] in IMG_EXTENSIONS]
            path_files = source_files + target_files
            # Getting files paths
            for path in tqdm(path_files, total=len(path_files),
                             desc="Getting Files"):
                name_file, index = _get_name_file(path).split("_")
                index = int(index.split(".")[0])
                self.image_data[name_file][index] = path
            self.prepare_data_for_blending()
                    
        self.device = device
        self.cache = cache
        self.save = save
    
    def prepare_data_for_blending(self):
        cpu_device = torch.device('cpu')
        for index in tqdm(range(len(self.image_data['mask'])),
                          total = len(self.image_data['mask']), desc="preprocess data"):
            mask = _numpy(load_image(self.image_data['mask'][index + 1],
                                     is_mask=True, device=cpu_device))
            source = _numpy(load_image(self.image_data['source'][index + 1],
                                       device=cpu_device))
            target = _numpy(load_image(self.image_data['target'][index + 1],
                                       device=cpu_device))
            mask, source, target = prepare_images(mask, source, target)

    
    def __len__(self) -> int:
        return len(self.image_data["dims"])
    
    def __getitem__(self, index) -> Dict:
        # Assert error in order to stop the indexing
        list(self.image_data["source"].values())[index]
        
        # Load images
        source = load_image(self.image_data["source"][index + 1],
                            device=self.device)
        target = load_image(self.image_data["target"][index + 1],
                            device=self.device)
        
        dims = json.load(open(self.image_data["dims"][index + 1], "r"))
        mask = load_image(self.image_data["mask"][index + 1], is_mask=True,
                          device=self.device)
        
        return {"source": source, "target": target,
                "dims": [dims["h0"], dims["h0"] + mask.shape[1],
                         dims["w0"], dims["w0"] + mask.shape[2]],
                "mask": mask}