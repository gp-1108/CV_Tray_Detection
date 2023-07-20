"""
  This script performs interactive cropping of images in a folder and its subfolders.
  It was used to crop the images in the dataset for the Computer Vision course.
"""
import os
import cv2

# Function to perform interactive cropping
def perform_crop(image):
  # Set the screen resolution
  screen_res = (1920, 1200)

  # Check if image dimensions exceed the screen resolution
  if image.shape[0] > screen_res[1] or image.shape[1] > screen_res[0]:
    # Calculate the aspect ratio of the image
    image_ratio = image.shape[1] / image.shape[0]

    # Calculate the new width and height based on the screen resolution
    new_width = screen_res[0]
    new_height = int(new_width / image_ratio)

    # Resize the image to fit within the screen resolution
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
  else:
    # Use the original image if its dimensions are smaller than the screen resolution
    image_resized = image

  # Display the resized image
  cv2.imshow('Resized Image', image_resized)

  # Prompt the user to select the region of interest (ROI) for cropping
  roi = cv2.selectROI('Resized Image', image_resized, fromCenter=False, showCrosshair=True)

  # Calculate the scaling factor for the original image
  scaling_factor = image.shape[0] / image_resized.shape[0]

  # Scale the ROI coordinates back to the original image size
  roi_scaled = (roi[0] * scaling_factor, roi[1] * scaling_factor,
                roi[2] * scaling_factor, roi[3] * scaling_factor)

  # Crop the image based on the scaled ROI
  cropped_image = image[int(roi_scaled[1]):int(roi_scaled[1] + roi_scaled[3]),
                        int(roi_scaled[0]):int(roi_scaled[0] + roi_scaled[2])]

  # Close the image display windows
  cv2.destroyAllWindows()

  return cropped_image


# Specify the paths
main_folder = 'Dataset_cv_train'
new_folder = 'Dataset_cv_train_cropped'

# Iterate through the main folder and its subfolders
for root, dirs, files in os.walk(main_folder):
  for file in files:
    # Check if the file is an image
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
      # Get the full path of the image
      image_path = os.path.join(root, file)

      # Get the relative path of the image (excluding the main folder)
      relative_path = os.path.relpath(root, main_folder)

      # Create the corresponding folder structure in the new folder
      new_subfolder = os.path.join(new_folder, relative_path)
      os.makedirs(new_subfolder, exist_ok=True)

      # Save the cropped image in the new folder
      new_image_path = os.path.join(new_subfolder, file)

      if os.path.exists(new_image_path):
        print(f'Skipping existing image: {new_image_path}')
        continue
      
      # Load the image
      image = cv2.imread(image_path)

      # Perform cropping interactively
      cropped_image = perform_crop(image)


      cv2.imwrite(new_image_path, cropped_image)

      print(f'Saved cropped image: {new_image_path}')
