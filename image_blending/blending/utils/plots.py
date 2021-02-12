import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple
from .image import _numpy, unormalize_image

IMAGE_DATA_KEYS = ['source', 'target', 'mask']

def plots_multiple_image_data(*images_data, normalize=True, figsize=(10, 5)):    
    fig, ax = plt.subplots(nrows=len(images_data),
                           ncols=len(IMAGE_DATA_KEYS),
                           figsize=figsize,
                           tight_layout=True)
    if len(images_data) == 1:
        ax = ax.reshape(1, -1)
        
    for nrow_plot, image_data in enumerate(images_data):
        source = _numpy(image_data['source'])
        mask = _numpy(image_data['mask'])
        target = _numpy(image_data['target'])
        
        if not normalize:
            source = source.astype(np.uint8)
            target = target.astype(np.uint8)
            mask = mask.astype(np.uint8)
            
        ax[nrow_plot][0].set_title("source image")
        ax[nrow_plot][0].imshow(source)
        
        ax[nrow_plot][1].set_title("mask image")
        ax[nrow_plot][1].imshow(mask, cmap="gray")
        
        ax[nrow_plot][2].set_title("target image")
        ax[nrow_plot][2].imshow(target)
    plt.show()
    
def plots_multiple_segmentation_data(*images_data, preprocess=False,
                                     normalize=True, figsize=(10, 5)):
    fig, ax = plt.subplots(nrows=len(images_data),
                           ncols=len(images_data[0].keys()),
                           figsize=figsize,
                           tight_layout=True)
    if len(images_data) == 1:
        if not isinstance(ax, np.ndarray):
            ax = np.array(ax)
        ax = ax.reshape(1, -1)
        
    for nrow_plot, image_data in enumerate(images_data):
        if preprocess:
            source = _numpy(unormalize_image(image_data['source']))
        else:
            source = _numpy(image_data['source'])
            
        mask = image_data.get("mask", None)
        if mask is not None:
            mask = _numpy(mask)
            
        if not normalize:
            source = source.astype(np.uint8)
            if mask is not None:
                mask = mask.astype(np.uint8)
            
        ax[nrow_plot][0].set_title("source image")
        ax[nrow_plot][0].imshow(source)
        
        if mask is not None:
            ax[nrow_plot][1].set_title("mask image")
            ax[nrow_plot][1].imshow(mask, cmap="gray")
        
    plt.show()
        
def get_subplots_config(total_images:int,
                        max_ncols:int = None,
                        max_nrows:int = None) -> Tuple[int, int, int]:
    use_cols = True
    
    if max_ncols is None and max_nrows is None:
        max_ncols = 5
    elif max_ncols is not None and max_nrows is None:
        use_cols = True
    elif max_ncols is None and max_nrows is not None:
        use_cols = False
    elif max_ncols is not None and max_nrows is not None:
        if total_images > (max_ncols*max_nrows):
            total_images = max_ncols*max_nrows
        
    if use_cols:
        if total_images%max_ncols == 0:
            nrows = total_images // max_ncols
        else:
            nrows = total_images // max_ncols + 1
            
        if total_images - max_ncols >= 0:
            ncols = max_ncols
        else:
            ncols = total_images
    else:
        if total_images%max_nrows == 0:
            ncols = total_images // max_nrows
        else:
            ncols = total_images // max_nrows + 1
            
        if total_images - max_nrows >= 0:
            nrows = max_nrows
        else:
            nrows = total_images
            
    return nrows, ncols, total_images

def plots_multiple_tensor_image(*tensor_images, title_name:list=[None],
                                ncols:int=None, nrows:int=None,
                                figsize:Tuple[int, int]=(14, 10),
                                normalize:bool = True):
    total_images = len(tensor_images)
    
    if total_images > len(title_name):
        extras = total_images - len(title_name)
        title_name.extend(range(extras))
        
    if ncols is None and nrows is None:
        nrows, ncols, total_images = get_subplots_config(total_images)
    elif ncols is not None and nrows is None:
        nrows, ncols, total_images = get_subplots_config(total_images,
                                                         max_ncols=ncols)
    elif ncols is None and nrows is not None:
        nrows, ncols, total_images = get_subplots_config(total_images,
                                                         max_nrows=nrows)
    elif ncols is not None and nrows is not None:
        nrows, ncols, total_images = get_subplots_config(total_images,
                                                         max_nrows=nrows,
                                                         max_ncols=ncols)
    
    if title_name[0] is None:
        title_name = list(range(total_images))
        
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=figsize, tight_layout=True)
    if nrows == 1 and ncols == 1:
        ax = [[ax]]
    elif nrows == 1:
        ax = ax.reshape(1, -1)
    elif ncols == 1:
        ax = ax.reshape(-1, 1)
    
    idx_image = 0
    for i in range(nrows):
        for j in range(ncols):
            if idx_image >= total_images:
                return
            else:
                if isinstance(tensor_images[idx_image], np.ndarray):
                    image_array = tensor_images[idx_image]
                else:  
                    image_array = _numpy(tensor_images[idx_image])
                if not normalize:
                    image_array = image_array.astype(np.uint8)
                    
                ax[i][j].set_title(title_name[idx_image])
                if len(image_array.shape) == 2:
                    ax[i][j].imshow(image_array, cmap="gray")
                else:
                    ax[i][j].imshow(image_array)
            idx_image += 1
    plt.show()