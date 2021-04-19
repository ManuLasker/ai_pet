import argparse
import logging
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from blending.models import MeanShift, VGG16_Model
from blending.utils import image as I
from blending.utils import dataset as D
from blending.utils import general as G
from blending.utils import plots as P
from blending.segmentation import (Predictor, SegmentationModule, VGGFCN)

paths = ['E:\\Koombea\\ai_pet\\image_blending\\source_data\\source_2.jpg',
         'E:\\Koombea\\ai_pet\\image_blending\\target_data\\target_1.jpg',
         'E:\\Koombea\\ai_pet\\image_blending\\target_data\\dims_1.json']

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()

def pretty_print(obj):
  return json.dumps(obj=obj, indent=3)

def predict_mask(source:torch.Tensor):
  # Preprocess source
  logger.info('preprocessing source for mask prediction')
  old_shape = tuple(source.shape[1:])
  source = source/255.0
  source = I.normalize_image(source)
  source_resize = I.resize_pad_image(tensor_image=source.clone(), new_shape=(224, 224))
  mask = Predictor.predict(source_resize.unsqueeze(0)).sigmoid()
  mask[mask > 0.8] = 1
  mask = mask.round().squeeze(0)
  return I.resize_up(mask, down_shape=(224, 224), up_shape=old_shape)

def plots_results(input_img:torch.Tensor, mask:torch.Tensor,
                  target:torch.Tensor, naive_copy:torch.Tensor, results_path:str):
  with torch.no_grad():
    blend_img = (input_img * mask + target * (1 - mask))
    fig, ax = plt.subplots(1, 2, figsize=(12, 12), tight_layout=True)

    ax[0].set_axis_off()
    ax[0].set_title("Blend Image Algorithm")
    ax[0].imshow(np.asarray(I._pil_image(blend_img)))
    
    ax[1].set_axis_off()
    ax[1].set_title("Copy and Paste")
    ax[1].imshow(np.asarray(I._pil_image(naive_copy)))
        
    fig.savefig(os.path.join(results_path, 'comparation.jpg'))

def blend(source_temp: torch.Tensor, mask_temp: torch.Tensor,
          target_temp: torch.Tensor, dims: np.ndarray,
          results_path:str, device:torch.device):
  target = target_temp
  h, w = target.shape[2], target.shape[3]
  x0, y0, x1, y1 = dims
  
  source = torch.zeros_like(target)
  source[:, :, y0:y1, x0:x1] = source_temp
  
  mask = torch.zeros(1, 1, h, w)
  mask[:, :, y0:y1, x0:x1] = mask_temp
  
  input_img = torch.randn(*source.shape, device=device).contiguous()
  input_img.requires_grad = True
  
  # Pass all tensors to device
  target = target.to(device=device)
  mask = mask.to(device=device)
  source = source.to(device=device)
  
  # blend_img = input_img * mask + target * (1 - mask)
  naive_copy = source * mask + target * (1 - mask)
  
  new_image_data = {
    'mask': mask,
    'target': target,
    'source': source,
    'dims': dims
  }
  
  # Get ground truth gradients
  gt_gradients = torch.stack(I.get_mixing_gradients(new_image_data, device=device), dim=2).squeeze(0)
  vgg16_features = VGG16_Model().to(device=device)
  mean_shift = MeanShift().to(device=device)
  
  # define optimizer and loss function
  optimizer = optim.LBFGS([input_img.requires_grad_()], lr=1.5, max_iter=200)
  mse_loss = nn.MSELoss().to(device=device)
  
  # Algorithms configuration
  run = [0]
  num_step = 1000
  w_grad, w_cont, w_tv, w_style = 1e4, 1e1, 1e-6, 0.05
  configurations = {
    'num_step': num_step,
    'alg config': {
      'w_grad': w_grad,
      'w_cont': w_cont,
      'w_tv': w_tv,
      'w_style': w_style
    }
  }
  logger.info(f'Blending algorithms configurations: {pretty_print(configurations)}')

  pbar = tqdm(total=num_step, desc='Blending operation ...', position=0)
  style_layers = vgg16_features.style_layers
  content_layers = vgg16_features.content_layers
  
  while run[0] < num_step:
    def closure():
      # zero grad optimizer
      optimizer.zero_grad()
      blend_img = (input_img * mask + target * (1 - mask))
      
      # gradient loss
      blend_gradients = torch.stack(I.get_blending_gradients(blend_img, device=device), dim=2).squeeze(0)
      loss_grad = w_grad * mse_loss(blend_gradients, gt_gradients) 
      
      # Content source loss
      input_features = vgg16_features(I.normalize_image(blend_img))
      source_features = vgg16_features(I.normalize_image(source))
      loss_content = 0
      for content_layer in content_layers:
          loss_content += mse_loss(input_features[content_layer], source_features[content_layer])
      loss_content /= (len(content_layers)/w_cont)
      
      # Style source loss
      loss_source_style = 0
      for style_layer in style_layers:
          loss_source_style += mse_loss(input_features[style_layer], source_features[style_layer])
      loss_source_style /= (len(style_layers)/w_style)
      
      # TV Reg Loss
      loss_tv = w_tv * (torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + 
                  torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :])))
      
      # colect total loss
      loss_total = loss_grad + loss_content + loss_tv + loss_source_style
      
      if (run[0] + 1)%50 == 0 or (run[0] + 1 == 1):
        with torch.no_grad():
          I._pil_image(blend_img).save(os.path.join(results_path, "image_iter.jpg"))
          
      # Backward Optimization Step
      loss_total.backward()

      # Update pbar
      pbar_stats = {
          "loss_grad": loss_grad.item(),
          "loss_content": loss_content.item(),
          "loss_source_style": loss_source_style.item(),
          "loss_tv": loss_tv.item(),
          "loss_total": loss_total.item()
      }
      pbar.set_postfix(**pbar_stats)
      pbar.update()
      
      # Update run
      run[0] += 1
      return loss_total
    
    # Optimize
    optimizer.step(closure)
    
  plots_results(input_img=input_img, mask=mask, target=target,
                naive_copy=naive_copy, results_path=results_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Python Image Blending'
                                   ' Script for crown and paw porject')
  parser.add_argument('--image-source-path', '-is-path', type=str, required=True,
                      help='path for image source')
  parser.add_argument('--image-target-path', '-it-path', type=str, required=True, 
                      help='path for image target')
  parser.add_argument('--target-dim-path', '-td-path', type=str, required=True,
                      help='path for target dims and positions. Is a Json file')
  parser.add_argument('--model-path', '-mp', type=str,  required=True,
                      help='model path for mask prediction')
  parser.add_argument('--device', '-d', type=str, default='cuda' 
                                    if torch.cuda.is_available() 
                                    else 'cpu', help='Device to execute script')
  parser.add_argument('--results-path', type=str, default='results',
                      help='result path to save the blend image')
  
  logger.info("Blending Script Begins...")
  # Parse args
  args = parser.parse_args()
  logger.info(f'arguments: {pretty_print(args.__dict__)}')
  
  # load tensors images
  logger.info('loading tensor images for source image...')
  source_image_tensor = I.load_image(path=args.image_source_path)
  
  logger.info('loading tensor images for target image...')
  target_image_tensor = I.load_image(path=args.image_target_path)
  
  logger.info('loading dims and positions')
  dims = np.array(json.load(open(args.target_dim_path, "r")), dtype=np.float32)
  
  logger.info('Loading Mask Model')
  Predictor.set_config(model_path=args.model_path)
  
  logger.info('Predicting mask...')
  mask = predict_mask(source=source_image_tensor)
  logger.info('Preparing images arrays for blending algorithms...')
  mask, source, target, dims = I.prepare_images_arrays(mask=I._numpy(mask),
                            source=I._numpy(source_image_tensor),
                            target=I._numpy(target_image_tensor),
                            dims=dims)
  
  mask = mask.unsqueeze(0)
  source = source.unsqueeze(0)
  target = target.unsqueeze(0)
  
  logger.info(f'Blend alg begins in device: {torch.device(args.device)} ...')
  
  blend(source_temp=source, mask_temp=mask, target_temp=target,
        results_path=args.results_path, dims=[int(d) for d in dims], device=torch.device(args.device))