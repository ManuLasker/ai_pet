import os

if __name__ == "__main__":
  data_dir = "data"
  new_data_dir = "cvat_data"
  num_file = 1
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      new_name = str(num_file).zfill(10)+"."+file.split(".")[-1]
      os.rename(src=os.path.join(root, file), 
                dst=os.path.join(new_data_dir, new_name))
      num_file += 1