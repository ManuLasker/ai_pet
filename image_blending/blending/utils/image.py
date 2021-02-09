import PIL
import torch
import numpy as np
import torch.nn as nn

from typing import Tuple
import torchvision.transforms as T 
from PIL import Image

def load_image(path, is_mask = False,
               device: torch.device = torch.device('cpu')):
    if not is_mask:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        image = _to_tensor(image, device=device)
        return image
    else:
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        image = _to_tensor(image, device=device)
        image[image > 0] = 1
        return image
    
def _pil_image(tensor_image: torch.Tensor):
    if isinstance(tensor_image, np.ndarray):
        tensor_image = _to_tensor(tensor_image)
        return _pil_image(tensor_image)
    else:
        transform = T.ToPILImage()
        if len(tensor_image.shape) > 3:
            tensor_image = tensor_image.squeeze(0).cpu()
        return transform(tensor_image)

def _numpy(tensor_image: torch.Tensor) -> np.ndarray:
    if len(tensor_image.shape) > 3:
        if tensor_image.shape[0] != 1:
            raise Exception("can't plots batch of tensor "
                            "just one tensor at a time")
        else:
            tensor_image = tensor_image.squeeze(0)
            
    array_image = np.transpose(tensor_image.cpu().numpy(), (1, 2, 0))
    _, _, c = array_image.shape
    if c == 1:
        return array_image.squeeze(2)
    else:
        return array_image
    
def _to_tensor(image, 
               device:torch.device = torch.device('cpu')) -> torch.Tensor:
    # Transfor pil image or numpy array image to torch.image
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, 
                                 dtype=torch.float32, device=device)
            return image
        elif len(image.shape) == 2:
            image = torch.tensor(image, 
                                 dtype=torch.float32, device=device).unsqueeze(0)
            return image
    else:
        # is a pil image
        image = np.array(image, dtype=np.float32)
        return _to_tensor(image, device)

def get_laplacian_kernel(device:torch.device = torch.device('cpu')) -> torch.Tensor:
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=device)
    return laplacian_kernel.view(1, 1, *laplacian_kernel.shape)

    
def get_image_laplacian_operator(tensor_image: torch.Tensor, 
                                 device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
    laplacian_kernel = get_laplacian_kernel(device)
    laplacian_conv = nn.Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
    laplacian_conv.weight = nn.Parameter(laplacian_kernel, requires_grad=False)
    rgb_channel_gradients = [laplacian_conv(tensor_image[:, ch, :, :].unsqueeze(1)) 
                             for ch in range(3)]
    return rgb_channel_gradients

def get_target_subimg(target: torch.Tensor,
                      mask: torch.Tensor,
                      dims: list):
    target_subimg = target[:, :, dims[0]:dims[1], dims[2]:dims[3]]
    return target_subimg

def get_mixing_gradients(image_data:dict,
                         device:torch.device = torch.device('cpu'),
                         alpha=0.5):
    source = image_data['source']
    mask = image_data['mask']
    dims = image_data['dims']
    target = image_data['target']
    # target = get_target_subimg(image_data['target'], mask, dims)
    
    rgb_source_gradients = [channel_gradient * mask 
                            for channel_gradient in get_image_laplacian_operator(source, device=device)]
    rgb_target_gradients = [channel_gradient * (1 - mask)
                            for channel_gradient in get_image_laplacian_operator(target, device=device)]
    # rgb_channel_target = [target[:, ch, :, :].unsqueeze(1) * (1 - mask) 
    #                       for ch in range(3)]
    # gradients_mix = [ source_channel * alpha + target_channel * (1 - alpha)
    #                  for source_channel, target_channel in zip(rgb_source_gradients,
    #                                                             rgb_channel_target)]
    gradients_mix = [ source_channel * alpha + target_channel * (1 - alpha)
                     for source_channel, target_channel in zip(rgb_source_gradients,
                                                                rgb_target_gradients)]
    return gradients_mix

def get_blending_gradients(tensor_image:torch.Tensor,
                           device:torch.device = torch.device('cpu')):
    # mask = image_data['mask']
    # dims = image_data['dims']
    # target = get_target_subimg(image_data['target'], mask, dims)
    
    # tensor_image_gradient = [channel_gradient * mask
    #                          for channel_gradient in get_image_laplacian_operator(tensor_image,
    #                                                                               device=device)]
    # rgb_channel_target = [target[:, ch, :, :].unsqueeze(1) * (1 - mask) 
    #                       for ch in range(3)]
    # gradient_blend = [ ch_img_gradient + ch_target_gradient
    #                   for ch_img_gradient, ch_target_gradient in zip(tensor_image_gradient,
    #                                                                  rgb_channel_target)]
    # gradient_blend = [channel_gradient 
    #                   for channel_gradient in get_image_laplacian_operator(tensor_image,
    #                                                                        device=device)]
    return get_image_laplacian_operator(tensor_image,
                                        device=device)

def normalize_image(tensor_image: torch.Tensor):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    return normalize(tensor_image)

def resize_pad_image(tensor_image: torch.Tensor,
                     new_shape: tuple) -> torch.Tensor:
    """Resize and pad if is needed

    Args:
        tensor_image (torch.Tensor): tensor image from load (ch, h, w)
        new_shape (tuple[int, int]): tuple int (h_new, w_new)
    """
    if isinstance(tensor_image, np.ndarray):
        tensor_image = _to_tensor(tensor_image)
    # ratio
    old_h, old_w = tensor_image.shape[1:]
    new_h, new_w = new_shape
    r = min(new_h/old_h, new_w/old_w)
    
    # Get padding
    new_unpad_h, new_unpad_w = round(old_h * r), round(old_w * r)
    dh, dw = (new_h - new_unpad_h)/2, (new_w - new_unpad_w)/2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    
    transform = T.Compose([
        T.Resize(size=(new_unpad_h, new_unpad_w)),
        T.Pad(padding=(left, top, right, bottom))
    ])
    return transform(tensor_image)

def prepare_images_arrays(mask: np.ndarray, 
                   source:np.ndarray,
                   target:np.ndarray):
    
    pass