#!/usr/bin/env python
# coding: utf-8

# # Biodiversity Monitoring: Implement Phase

# packages for interacting with the operating system and logging
import os, sys 
import logging

# Tensorflow neural network framework packages
import tensorflow as tf
from tensorflow.keras.applications import nasnet

from IPython.display import display # package for displaying images in Jupyter
from PIL import Image # package for loading images in Python

# Setup some paths for MegaDetector
os.environ['PYTHONPATH'] += ":/home/jovyan/work/ai4eutils"
os.environ['PYTHONPATH'] += ":/home/jovyan/work/CameraTraps"
os.environ['PYTHONPATH'] += ":/home/jovyan/work/yolov5"
sys.path.insert(0, "./ai4eutils")
sys.path.insert(0, "./CameraTraps")
sys.path.insert(0, "./yolov5")

# Supress the TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ignore tf warning messages
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Load the Pytorch Detector
from detection.pytorch_detector import PTDetector


import utils2 # utility functions defined for this lab

print('All packages imported successfully!')


# ## 2. Load the Metadata

# Define important variables
IMAGE_DIR = 'sample_data'
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 11

# Get the labels and categories
label2cat = utils2.get_labels()
cat2label = {v:k for k,v in label2cat.items()}
print(cat2label)


# ## 3. Load the MegaDetector

# Load the MegaDetector v5a model
model_file='md_v5a.0.0.pt'
megadetector = PTDetector(model_file)


# ## 4. Load the Fine-tuned NASNet model

# Load the base model (NASNETMobile)
base_model = nasnet.NASNetMobile(include_top=False)

base_model.trainable = False

# Add the top layers for classifiying karoo pictures
model = utils2.get_transfer_model(
    model_to_transfer=base_model,
    num_classes=NUM_CLASSES,
    img_height=IMAGE_SIZE[0],
    img_width=IMAGE_SIZE[1],
)
# Load the weights of the fine tunned model
model_weight_path = 'models/model_cnn_finetuned_nasnet_150epocha_augmented.h5'
model.load_weights(model_weight_path)


# ## 5. Detect and Classify Animals 

image = Image.open("sample_data/test/jackalblackbacked/KAR_S1_B02_R1_IMAG0937.JPG")
result = utils2.draw_bounding_box(image, megadetector, model, label2cat)
display(result) 


# Run the next cell to select an image that you want to classify from the Karoo dataset. 

components = utils2.animal_detection_on_server(display, megadetector, model, label2cat)

display(components['fileChooser'])
display(components['output'])


# Run the next cell to upload your own pictures and see the result. Processing a single image can take up to 1 minute depending on its size. 

components2 = utils2.animal_detection_local(display, megadetector, model, label2cat)

display(components2['fileUpload'])
display(components2['output'])





