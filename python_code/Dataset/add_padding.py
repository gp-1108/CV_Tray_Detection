"""
  This file is used to add white padding to the images in the dataset.
  Eventually this feature only introduced noise in the dataset and was not used.
"""
import os
from PIL import Image
import cv2
import numpy as np

def apply_mask_and_fill(image_path_1, image_path_2, new_image_path):
  image_1 = cv2.imread(image_path_1)
  image_2 = cv2.imread(image_path_2)

  # Resize image_1 to match image_2
  image_1 = cv2.resize(image_1, (image_2.shape[1], image_2.shape[0]))

  # Now loop over the first image
  # if the pixel has value (255,255,255) then set the second image pixel to (255,255,255)
  # else keep the value
  for i in range(image_1.shape[0]):
    for j in range(image_1.shape[1]):
      if np.all(image_1[i,j] == 255):
        image_2[i,j] = 255

  cv2.imwrite(new_image_path, image_2)

dataset_path = 'Dataset_cv_train_cropped_augmented'
output_path = 'Dataset_cv_train_cropped_augmented_padded'
test_dataset_path = 'Dataset_cv_test'

# Get a list of all the images in the test dataset
test_images = []
for root, dirs, files in os.walk(test_dataset_path):
  if len(files) <= 0:
    continue
  folder_name = os.path.basename(root)
  for file in files:
    # Get the full path of the image
    image_path = os.path.join(root, file)
    test_images.append(image_path)

  

# Iterate through the parent folder and its subfolders
for root, dirs, files in os.walk(dataset_path):
  if len(files) <= 0:
    continue
  folder_name = os.path.basename(root)
  
  # Create the corresponding folder structure in the output folder
  new_subfolder = os.path.join(output_path, folder_name)
  os.makedirs(new_subfolder, exist_ok=True)

  # Copy the original images to the output folder
  for file in files:
    # Get the full path of the image
    image_path = os.path.join(root, file)

    # Get the relative path of the image (excluding the parent folder)
    relative_path = os.path.relpath(root, dataset_path)

    # Save the image in the output folder
    new_image_path = os.path.join(new_subfolder, file)

    apply_mask_and_fill(test_images[np.random.randint(len(test_images))], image_path, new_image_path)

    # Open the image
    image = Image.open(new_image_path)

    padding = 15
    width, height = image.size
    new_width = width + 2 * padding
    new_height = height + 2 * padding
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    new_image.paste(image, (padding, padding))
    new_image.save(new_image_path)
  print(f"Folder {folder_name} done!")