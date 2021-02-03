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
    scale_coords[:, [0, 2]] += dw  # x padding
    scale_coords[:, [1, 3]] += dh  # y padding
    return scale_coords.round()