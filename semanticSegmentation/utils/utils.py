import yaml
from pathlib import Path

def load_yaml(path):
  with open(Path(path)) as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
  return yaml_dict