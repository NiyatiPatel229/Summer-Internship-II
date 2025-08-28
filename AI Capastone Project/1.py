# --- Task 1: Get shape of a single image ---
import numpy as np
print("Image shape:", image_data.shape)  # assuming image_data is a numpy array

# --- Task 2: Display first four images in class_0_non_agri directory ---
import matplotlib.pyplot as plt
import os
from PIL import Image

img_dir = './images_dataSAT/class_0_non_agri/'
img_files = sorted(os.listdir(img_dir))[:4]
for idx, fname in enumerate(img_files):
    img = Image.open(os.path.join(img_dir, fname))
    plt.subplot(1, 4, idx+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()

# --- Task 3: Create sorted agri_images_paths list ---
import glob
dir_agri = './images_dataSAT/class_1_agri/'
agri_images_paths = sorted(glob.glob(os.path.join(dir_agri, '*')))
print("Total images:", len(agri_images_paths))

# --- Task 4: Number of agricultural images ---
num_agri_images = len(os.listdir('./images_dataSAT/class_1_agri/'))
print("Number of agri images:", num_agri_images)

