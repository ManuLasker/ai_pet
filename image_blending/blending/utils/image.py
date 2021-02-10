import PIL
import cv2
import torch
import numpy as np
import torch.nn as nn

from typing import Tuple
from .general import get_dimensions_box, scale_coordinates, get_middel_point, xywh2xyxy, xyxy2xywh
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
                   target:np.ndarray,
                   dims: np.ndarray):
    # resize target
    target_old_shape = target.shape[:-1]
    target = resize_pad_image(target, new_shape=(640, 640))
    tm_coords = scale_coordinates(new_shape=(640, 640),
                                  coordinates=dims,
                                  old_shape=target_old_shape)
    target_mp = get_middel_point(tm_coords)
    # mask and source information
    sm_coords, source_mp = get_bbInfo_from_mask(source=source,
                                                mask=mask)
    # get new mask with grabcut algorithm
    mask = get_grabcut(source=source, mask=mask, rect=sm_coords)
    # crop mask and source
    sm_coords = sm_coords.astype(int)
    mask = mask[sm_coords[1]:sm_coords[3], sm_coords[0]:sm_coords[2]]
    source = source[sm_coords[1]:sm_coords[3], sm_coords[0]:sm_coords[2]]
    # resize crops to match w, h of target membrane
    tmw, tmh = get_dimensions_box(tm_coords)
    mask = resize_pad_image(mask, new_shape=(tmh, tmw))
    source = resize_pad_image(source, new_shape=(tmh, tmw))
    return mask, source/255.0, target/255.0, tm_coords
    
def get_bbInfo_from_mask(source:np.ndarray, mask:np.ndarray):
    # Get Contours
    contours, _ = cv2.findContours(image=mask.astype(np.uint8),
                                   mode=cv2.RETR_TREE,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Get bounding box from contours
    c = contours[0]
    sm_coords = np.array(cv2.boundingRect(c), dtype=np.int32)
    sm_coords = xywh2xyxy(sm_coords.astype(np.float32))
    source_mp = get_middel_point(sm_coords.astype(np.float32))
    
    return sm_coords, source_mp

def get_grabcut(source, mask, rect):
    # Numpy configuration
    bgdModel = np.zeros(shape=(1, 65), dtype=np.float64)
    fgdModel = np.zeros(shape=(1, 65), dtype=np.float64)
    
    # Grab cut
    mask_grabcut, _, _ = cv2.grabCut(source.astype(np.uint8),
                                     mask.astype(np.uint8), xyxy2xywh(rect),
                                     bgdModel=bgdModel, fgdModel=fgdModel,
                                     iterCount=10, mode=cv2.GC_INIT_WITH_MASK)
    
    # process grab cut
    mask_grabcut = np.where((mask_grabcut == 2)|(mask_grabcut == 0),
                            0, 1).astype(np.uint8)
    return mask_grabcut