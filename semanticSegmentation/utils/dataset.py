from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

import torch
import glob
import cv2
import numpy as np

class LoadImagesAndLabels(Dataset): # for train/test
  def __init__(self, path, img_size=320, transforms=None, augment = False):
    path = Path(path)
    if path.is_dir():
      temp_data = glob.glob(str(path/'**'/'*.*'), recursive=True)
      self.labels_paths = [data for data in temp_data if 'SegmentationObject' in data]
      self.labels_paths.sort()
      # label_cfg_path = [data for data in temp_data if 'labelmap.txt' in data]
      self.imgs_paths = [data for data in temp_data if 'ImageObject' in data]
      self.imgs_paths.sort()
    self.img_size = img_size
    self.transforms = transforms
    self.augment = augment
  
  def __getitem__(self, idx):
    img = load_img(self.imgs_paths[idx], self.img_size, augment=self.augment)
    mask_img = load_img(self.labels_paths[idx], self.img_size, mask=True, augment=self.augment)
    if self.transforms:
      img = self.transforms(img)
    return img, mask_img.view(-1)
  
  def __len__(self):
    return len(self.imgs_paths)

def load_img(path, img_size, mask=False, augment=False):
  img = cv2.imread(path)  # BGR
  assert img is not None, 'Image Not Found ' + path
  h0, w0 = img.shape[:2]  # orig hw
  r = img_size / max(h0, w0)  # resize image to img_size
  if r != 1:  # always resize down, only resize up if training with augmentation
    interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
  h, w = img.shape[:2] # hw_resized
  # img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
  # letterbox
  shape = img_size
  img, ratio, (dw, dh) = letterbox(img, shape, auto=False)
  if mask:
    img = img[:,:,::-1]
    img = np.ascontiguousarray(img)
    img = preprocess_mask(img)
  else:
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
  img = img/255.0
  img = img.astype(np.float32)
  return torch.from_numpy(img)


def preprocess_mask(mask):
  new_mask = mask.copy()
  new_mask[~ np.all(mask == [0,0,0],  axis=2)] = [255, 255, 255]
  new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2GRAY)
  return new_mask
  
def letterbox(img, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True):
  # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
  shape = img.shape[:2]  # current shape [height, width]
  if isinstance(new_shape, int):
      new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
  if not scaleup:  # only scale down, do not scale up (for better test mAP)
      r = min(r, 1.0)

  # Compute padding
  ratio = r, r  # width, height ratios
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
  if auto:  # minimum rectangle
      dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
  elif scaleFill:  # stretch
      dw, dh = 0.0, 0.0
      new_unpad = (new_shape[1], new_shape[0])
      ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

  dw /= 2  # divide padding into 2 sides
  dh /= 2

  if shape[::-1] != new_unpad:  # resize
      img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
  return img, ratio, (dw, dh)