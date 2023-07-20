"""
  This main function is used to augment the dataset.
  I want to obtain 301 samples for each class, preserving the original ones.
  The transformation applied are:
    - Random horizontal flip
    - Random vertical flip
    - Random affine transformation
"""

import os
import numpy as np
from torchvision import transforms
from PIL import Image

# Specify the paths
parent_folder = 'Dataset_cv_train_cropped'
output_folder = 'Dataset_cv_train_cropped_augmented'
desired_samples = 300

transformation = transforms.Compose([
  transforms.RandomHorizontalFlip(p=0.5),
  transforms.RandomVerticalFlip(p=0.5),
  transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=7, fill=255),
])


# Iterate through the parent folder and its subfolders
for root, dirs, files in os.walk(parent_folder):
  if len(files) <= 0:
    continue
  folder_name = os.path.basename(root)
  
  # Create the corresponding folder structure in the output folder
  new_subfolder = os.path.join(output_folder, folder_name)
  os.makedirs(new_subfolder, exist_ok=True)

  # Copy the original images to the output folder
  count = 0
  for file in files:
    # Get the full path of the image
    image_path = os.path.join(root, file)

    # Get the relative path of the image (excluding the parent folder)
    relative_path = os.path.relpath(root, parent_folder)

    # Save the image in the output folder
    new_image_path = os.path.join(new_subfolder, file)
    os.system(f'cp {image_path} {new_image_path}')
    count += 1
  
  # Perform data augmentation
  while count <= desired_samples:
    # Get the full path of the image
    image_path = os.path.join(root, files[np.random.randint(len(files))])

    # Get the relative path of the image (excluding the parent folder)
    relative_path = os.path.relpath(root, parent_folder)

    # Applying torchivison transformations
    image = Image.open(image_path)
    image_transformed = transformation(image)

    # Save the image in the output folder
    new_image_path = os.path.join(new_subfolder, f'{count}.jpg')
    image_transformed.save(new_image_path)
    count += 1
  print(f"Folder {folder_name} done! with {count} images")

  
      
