#!/usr/bin/env python
# coding: utf-8

# # Biodiversity Monitoring: Design Phase (Part 1)

# ## 1. Import Python packages

import os, sys # packages for defining paths for use with MegaDetector model

import glob                       # package for working with pathnames and patterns
from random import choice         # function to select a random entry from a list
import matplotlib.pyplot as plt   # package to create plots

# Set paths for MegaDetector's dependencies
os.environ['PYTHONPATH'] += ":/home/jovyan/work/ai4eutils"
os.environ['PYTHONPATH'] += ":/home/jovyan/work/CameraTraps"
os.environ['PYTHONPATH'] += ":/home/jovyan/work/yolov5"
sys.path.insert(0, "./ai4eutils")
sys.path.insert(0, "./CameraTraps")
sys.path.insert(0, "./yolov5")

# Import the Pytorch detector
from detection.pytorch_detector import PTDetector
# Import the utility functioms for working with images
import visualization.visualization_utils as viz_utils

import utils2 # utility functions defined for this lab

print('All packages imported successfully!')


# ## 2. Find animals in images with MegaDetector

# Load the MegaDetector model
model_file='md_v5a.0.0.pt'
megadetector = PTDetector(model_file)


# ### 2.1 Explore the MegaDetector output

# Print out the contents of the "data" folder
get_ipython().system('ls data')

sample_im_file = './data/bustardkori/KAR_S1_D03_R1_IMAG0111.JPG'
# Load the image using the viz_utils tools
sample_image = viz_utils.load_image(sample_im_file)
# Show the image
sample_image

# Run MegaDetector on a single image
megadetector_result = megadetector.generate_detections_one_image(sample_image, sample_im_file, detection_threshold=0.6)
print(megadetector_result)


# Use the MegaDetector result to draw the bounding box on the image
utils2.draw_bounding_box(sample_image, megadetector_result)

# Create a list of example images
examples = list(glob.iglob(f'./data/**/*.JPG', recursive=True))
print('List of example images created')

# Grab a random image and run MegaDetector on it
random_im_file = choice(examples)
random_image = viz_utils.load_image(random_im_file)
random_megadetector_result = megadetector.generate_detections_one_image(random_image, random_im_file, detection_threshold=0.6)
print(random_megadetector_result)
utils2.draw_bounding_box(random_image, random_megadetector_result)

# Run the MegaDetector on an image with multiple detections
example_im_file = './data/gemsbokoryx/KAR_S1_A01_R1_IMAG00291.JPG'
example_image = viz_utils.load_image(example_im_file)
example_megadetector_result = megadetector.generate_detections_one_image(example_image, example_im_file, detection_threshold=0.6)
print(example_megadetector_result)
utils2.draw_bounding_box(example_image, example_megadetector_result)


# ## 3. Crop and square images

get_ipython().run_line_magic('matplotlib', 'inline')
# Choose a random file path
random_im_file = choice(examples)
# Load the image
random_image = viz_utils.load_image(random_im_file)
# Detect animals using megadetector and crop the detected areas
res = megadetector.generate_detections_one_image(random_image, random_im_file, detection_threshold=0.6)
print(f"Megadetector output:\n{res}\n\nDetected animals:")
animals = utils2.crop_image(random_image, res, './data','/tmp')

# Plot the image with bounding boxes and cropped images
utils2.plot_detections(random_image, animals, res)

# ## 5. Process the Dataset

root_dir = './data'
utils2.preprocess_dataset(root_dir, megadetector, utils2.crop_image, 0.6, 0)
print('Images processed successfully')

# Plot a random set of the cropped images
utils2.plot_cropped_images()





