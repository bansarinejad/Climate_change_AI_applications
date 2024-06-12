#!/usr/bin/env python
# coding: utf-8

# # Biodiversity Monitoring: Design Phase (Part 2)

#  6.2 Visual inspection

# ## 1. Import Python packages

import os, sys        # packages to interact with the Operating System
import pandas as pd   # package for reading in and manipulating data
import numpy as np    # package for numerical operations
import matplotlib.pyplot as plt # package to create plots
from IPython.display import Image as IPythonImage  # package to display images in Jupyter

# various packages for neural network model creation and evaluation
import tensorflow as tf 
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy 
from tensorflow.keras.applications import nasnet

# Set paths for visualization_utils
os.environ['PYTHONPATH'] += ":/home/jovyan/work/ai4eutils"
os.environ['PYTHONPATH'] += ":/home/jovyan/work/CameraTraps"
os.environ['PYTHONPATH'] += ":/home/jovyan/work/yolov5"
sys.path.insert(0, "./ai4eutils")
sys.path.insert(0, "./CameraTraps")
sys.path.insert(0, "./yolov5")

import utils2 # utility functions defined for this lab

# Configure Python to ignore Tensorflow warnings
utils2.ignore_tf_warning_messages()

# Set random seed for reproducibility
RANDOM_SEED = 42
tf.keras.utils.set_random_seed(RANDOM_SEED)

print('All packages imported successfully!')

# # 2. Try out the original NASNet model

# Load the pre-trained NASNet model
original_nasnet_model = nasnet.NASNetMobile(include_top=True)
print("NASNet model loaded successfully")

# Test image prediction with the pre-trained NASNet model on the Snapshot Karoo dataset
IMAGE_DIR = 'data_crops'
label2cat_full = {i:category for i, category in enumerate(sorted(next(os.walk(f'{IMAGE_DIR}/train'))[1]))}
cat2label_full = {v:k for k,v in label2cat_full.items()}
TEST_DIR = IMAGE_DIR+'/test'
IMAGE_SIZE = (224, 224)
test_imgs = utils2.get_test_imgs(TEST_DIR)

utils2.pick_img_and_plot_predictions(test_imgs, original_nasnet_model, nasnet.decode_predictions, cat2label_full, IMAGE_SIZE)

# ## 3.  Explore data augmentation

# Define batch variables
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
OUTPUT_DIR = 'data_final'

# Load in the images as TensorFlow datasets
train_ds_full, _, _ = utils2.load_data(IMAGE_DIR, BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED)

# Load example data
images, labels = next(iter(train_ds_full))
print('examples loaded')

# Select an example image for augmentation (must be a number between 0 and 31)
selected_image = 3

image = images[selected_image].numpy().astype("uint8")
label = label2cat_full[labels[selected_image].numpy()]

utils2.plot_single_image(image, label)

# Augment the image with a flip
utils2.data_aug_flip(image)

# Augment the image with a zoom factor
utils2.data_aug_zoom(image)

# Augment the image with a rotation
utils2.data_aug_rot(image)

# Augment the image with a contrast 
utils2.data_aug_contrast(image)

# Apply a random set of image augmentations
utils2.data_aug_random(image);

# ## 4. Balance the Dataset

### this cell will take about 4 minutes to run

# Resample the data
utils2.resample_data('data_crops', OUTPUT_DIR, train_ds_full, 11, 500)

# Load in the images as TensorFlow datasets
train_ds, val_ds, test_ds = utils2.load_data(OUTPUT_DIR, BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED)

# Get the labels and categories
label2cat = {i:category for i, category in enumerate(sorted(next(os.walk(f'{OUTPUT_DIR}/train'))[1]))}
cat2label = {v:k for k,v in label2cat.items()}

label2cat

# Count up the number of example images for each class of animal before and after resampling
count_original = utils2.count_examples_per_class(train_ds_full, label2cat_full, cat2label_full)
count_resampled = utils2.count_examples_per_class(train_ds, label2cat, cat2label)

# Plot the bar charts 
utils2.plot_histograms_of_data(count_original, count_resampled, cat2label, label2cat)

# ## 5. Create model based on NASNet

# Load the NASNet pre-trained base model
base_model = nasnet.NASNetMobile(include_top=False)
print("NASNet model loaded successfully")

# Print out a summary of the NASNet model architecture
base_model.summary()

# Prepare a model for fine tuning
base_model.trainable = False
NUM_CLASSES = 11
model = utils2.get_transfer_model(
    model_to_transfer=base_model,
    num_classes=NUM_CLASSES,
    img_height=IMAGE_SIZE[0],
    img_width=IMAGE_SIZE[1],
)

# Load the fine tuned model weights
model_weight_path = 'models/model_cnn_finetuned_nasnet_150epocha_augmented.h5'
model.load_weights(model_weight_path)

# Run additional fine tuning epochs
epochs = 1
history_finetune = model.fit(
    train_ds,
    epochs=epochs
)

# Plot the training history for the fine tuning process
utils2.plot_training_history('history_training')

# ## 6. Model evaluation

# Create lists for storing the predictions and labels
y_pred = []
y_true = []

# Loop over the data generator
for data, label in test_ds:
    # Make predictions on data using the model. Store the results.
    y_pred.extend(tf.argmax(model.predict(data), axis=1).numpy())
    # Store corresponding labels
    y_true.extend(label.numpy())
# Print out the overall accuracy of the model predictions
print('Overall model accuracy:', np.sum(np.array(y_true) == y_pred)/len(y_pred))

# Plot the confusion matrix
utils2.plot_cm(y_true, y_pred, label2cat)

# ### 6.2 Visual inspection

# Plot images and the top three model predictions by confidence level
utils2.pick_img_and_plot_predictions(test_imgs, model, label2cat, cat2label, IMAGE_SIZE)



