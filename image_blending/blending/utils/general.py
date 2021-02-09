from typing import Tuple
import torch
import numpy as np

def scale_coordinates(new_shape: tuple, coordinates: np.ndarray,
                      old_shape: tuple):
    """Scale corrdinates down

    Args:
        new_shape (tuple): (h_new, w_new)
        coordinates (np.ndarray): [x0, y0, x1, y1]
        old_shape (tuple): (h_old, w_old)
    """
    scale_coords = coordinates.copy()
    r = min(new_shape[0]/old_shape[0], new_shape[1]/old_shape[1])
    unpad = (round(old_shape[0] * r),
            round(old_shape[1] * r))
    dw, dh = ((new_shape[1] - unpad[1])//2,
              (new_shape[0] - unpad[0])//2)
    # Scale down all coordinates
    scale_coords *= r
    # Apply pad
    scale_coords[[0, 2]] += dw  # x padding
    scale_coords[[1, 3]] += dh  # y padding
    return scale_coords.round()

def get_middel_point(xyxy_coordinates: np.ndarray):
    x0, y0, x1, y1 = xyxy_coordinates
    w, h = (np.abs(x0 - x1),
            np.abs(y0 - y1))
    return (int(x0 + w//2), int(y0 + h//2))

def load_target_head_dimensions(path:str, image_shape: list):
    lines = open(path, "r").readlines()
    coords = np.array([line.strip().split(" ")[1:]
                       for line in lines],
                      dtype=np.float32)
    return x_my_m_wh2xyxy(coords.reshape(-1) * (image_shape[::-1] * 2))

def x_my_m_wh2xyxy(coords:np.ndarray)->np.ndarray:
    """Transform coordinates from xywh, (middel points, width and height) to
    xyxy, (top-left, bottom-right)

    Args:
        coords (np.ndarray): Numpy array with xywh coordinates

    Returns:
        new_coords (np.ndarray): Numpy array with xyxy coordinates
    """
    new_coords = np.zeros_like(coords)
    new_coords[0] = coords[0] - coords[2]/2
    new_coords[1] = coords[1] - coords[3]/2
    new_coords[2] = coords[0] + coords[2]/2
    new_coords[3] = coords[1] + coords[3]/2
    return new_coords.round()

def xywh2xyxy(coords: np.ndarray) -> np.ndarray:
    """Transform coordinates from xywh, (top-left, width and height)
    to xyxy, (top-left, bottom-right)

    Args:
        coords (np.ndarray): Numpy array with xywh coordinates

    Returns:
        new_coords (np.ndarray): Numpy array with xyxy coordinates
    """
    new_coords = np.zeros_like(coords)
    new_coords[0] = coords[0]
    new_coords[1] = coords[1]
    new_coords[2] = coords[0] + coords[2]
    new_coords[3] = coords[1] + coords[3]
    return new_coords

def xyxy2xywh(coords: np.ndarray) -> np.ndarray:
    """Trandform coordinates from xyxy, (top-left, bottom-right) 
    to xywh, (top-left, width and height)

    Args:
        coords (np.ndarray): Numpy array with xyxy coordinates

    Returns:
        np.ndarray: Numpy array with xywh coordinates
    """
    new_coords = np.zeros_like(coords)
    new_coords[0] = coords[0]
    new_coords[1] = coords[1]
    new_coords[2] = np.abs(coords[0] - coords[2])
    new_coords[3] = np.abs(coords[1] - coords[3])
    return new_coords

def get_dimensions_box(coords: np.ndarray) -> Tuple[int, int]:
    w, h = (np.abs(coords[0] - coords[2]), 
            np.abs(coords[1] - coords[3]))
    return int(w), int(h)