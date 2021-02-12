import json
import os
from os import path
import torch
import numpy as np
from pathlib import Path
from torch.functional import norm
from tqdm.auto import tqdm
from typing import Dict, Tuple
from torch.utils.data import Dataset
from .image import (load_image, _numpy, _to_tensor,
                    normalize_image, resize_pad_image,
                    prepare_images_arrays, _pil_image)

IMG_EXTENSIONS = ['jpg', 'png', 'jpeg', 'json']
def _get_name_file(full_path: Path):
    return full_path.name
        

class ImageDataBlending(Dataset):
    def __init__(self, blending_dir, source_dir = None, target_dir = None,
                 device:torch.device = torch.device('cuda') 
                                    if torch.cuda.is_available() 
                                    else torch.device('cpu')):
        self.blending_dir = blending_dir
        # Initialize files needed for blend
        self.image_data = {"dims": {},
                           "source": {},
                           "target": {},
                           "mask": {}}
        cache = source_dir is None and target_dir is None
        # Cache files or recache
        if cache:
            self._cache()
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
            self._cache()
                    
        self.device = device

    def _cache(self):
        # Read all the files
        path_files = [Path(os.path.join(self.blending_dir, path)) 
                for path in os.listdir(self.blending_dir) if path.split(".")[-1] in IMG_EXTENSIONS]
        
        # Getting files paths
        for path in tqdm(path_files, total=len(path_files),
                        desc="Getting Files"):
            name_file, index = _get_name_file(path).split("_")
            index = int(index.split(".")[0])
            self.image_data[name_file][index] = path
            
    def prepare_data_for_blending(self):
        cpu_device = torch.device('cpu')
        for index in tqdm(range(len(self.image_data['mask'])),
                          total = len(self.image_data['mask']), desc="preprocess data"):
            dims = np.array(json.load(open(self.image_data["dims"][index + 1], "r")),
                            dtype=np.float32)
            mask = _numpy(load_image(self.image_data['mask'][index + 1],
                                     is_mask=True, device=cpu_device))
            source = _numpy(load_image(self.image_data['source'][index + 1],
                                       device=cpu_device))
            target = _numpy(load_image(self.image_data['target'][index + 1],
                                       device=cpu_device))
            mask, source, target, dims = prepare_images_arrays(mask, source, target, dims)
            # save each mask, source, target
            tensors = {"mask":mask,
                       "source":source,
                       "target":target}
            for keys, t in tensors.items():
                _pil_image(t).save(os.path.join(self.blending_dir,
                                                keys+"_"+str(index + 1)+".jpg"))
            json.dump([int(xy) for xy in dims], open(os.path.join(self.blending_dir,
                                                "dims_"+str(index + 1)+".json"), "w"))

    
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
                "dims": dims,
                "mask": mask}
        
class SegmentationData(Dataset):
    def __init__(self, image_dir, mask_dir=None,
                 img_size=(224, 224),
                 device=torch.device("cpu"),
                 preprocess = True) -> None:
        self.img_size = img_size
        self.device = device
        self.preprocess = preprocess
        img = [Path(os.path.join(image_dir, path)) 
                for path in os.listdir(image_dir)
                if path.split(".")[-1] in IMG_EXTENSIONS and "mask" not in path]
        img = sorted(img, key=lambda path: int(_get_name_file(path).split("_")[1].split(".")[0]))
        
        if mask_dir is None:
            self.train = False
            self.img_paths = {
                "img": img
            }
        else:
            self.train = True
            mask = [Path(os.path.join(mask_dir, path))
                    for path in os.listdir(mask_dir)
                    if path.split(".")[-1] in IMG_EXTENSIONS and "source" not in path]
            mask = sorted(mask, key=lambda path: int(_get_name_file(path).split("_")[1].split(".")[0]))
            self.img_paths = {
                "img": img,
                "mask": mask
            }
        
    def __len__(self) -> int:
        return len(self.img_paths["img"])
    
    def preprocess_img(self, img: torch.Tensor):
        if self.preprocess:
            img = img/255.0
            img = normalize_image(img)
            img = resize_pad_image(img, new_shape=self.img_size)
        return img
    
    def preprocess_mask(self, mask: torch.Tensor):
        if self.preprocess:
            mask = resize_pad_image(mask, new_shape=self.img_size)
        return mask
    
    def __getitem__(self, index) -> list:
        if self.train:
            mask = load_image(self.img_paths['mask'][index], is_mask=True, 
                              device=self.device)
            img = load_image(self.img_paths["img"][index], device=self.device)
            original_shape = tuple(img.shape)
            img = self.preprocess_img(img)
            return {"source":img, "mask":self.preprocess_mask(mask),
                    "original_shape": original_shape[1:],
                    "new_shape": self.img_size}
        else:
            img = load_image(self.img_paths["img"][index], device=self.device)
            original_shape = tuple(img.shape)
            img = self.preprocess_img(img)
            return {"source":img,
                    "original_shape": original_shape[1:],
                    "new_shape": self.img_size}