import logging
import torch
import json
import argparse

from blending.utils import image as I
from blending.utils import dataset as D
from blending.utils import general as G
from blending.utils import plots as P
from blending.segmentation import (Predictor, SegmentationModule, VGGFCN)

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()

def pretty_print(obj):
  return json.dumps(obj=obj, indent=3)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Python Export Mask Model'
                                   ' Script for crown and paw porject')
  parser.add_argument('--model-path', '-mp', type=str,  required=True,
                      help='model path for mask prediction')
  parser.add_argument('--model-exported-path', '-mpe', type=str, required=True,
                      help='Path where model will be stored the exported one')
  logger.info("Export mask model Script Begins...")
  # Parse args
  args = parser.parse_args()
  logger.info(f'arguments: {pretty_print(args.__dict__)}')
  
  logger.info('Loading Mask Model')
  Predictor.set_config(model_path=args.model_path)
  
  logger.info('Exporting model to ' + args.model_exported_path)
  Predictor.export_torch_script(args.model_exported_path)
  