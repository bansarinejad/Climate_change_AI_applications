#!/usr/bin/env python
# coding: utf-8

# # Biodiversity Monitoring: Explore Phase

# ## 1. Import Python packages

import pandas as pd # package for reading in and manipulating data              
import utils # utility functions for this lab

print('All packages imported successfully!')

# ## 2. Inspect the data

# Print out the contents of the "data" folder
get_ipython().system('ls data')

# Within one folder there are images of the same class. See the example below.

# Print out the contents of one folder inside of the data folder
get_ipython().system('ls data/rhinocerosblack')


# The file name contains some meta information about the image:
#     * KAR_S1: It indicates that the image is from the Karoo dataset season 1
#     * A01 | A02 | B01 | B02 ...: The camera trap location code
#     * R#: Repetition. This is always R1 in this dataset
#     * IMG####:  A consecutive number of the image. Must be unique within each location

IMAGE_DIR = 'data/'

meta_data = utils.get_metadata(IMAGE_DIR)
# Print the number of rows and columns in the metadata dataframe 
print(f"Shape of the dataframe (rows, columns): {meta_data.shape}")

# Show the first few entries in the dataframe
meta_data.head()

# Extract all of the unique class names and locations.
class_names = sorted(meta_data['class'].unique())
locations = sorted(meta_data['location'].unique())
    
print(f"Class names:\n{class_names}")
print(f"\nCamera locations:\n{locations}")

# Count images by location and print out the result
location_all = meta_data['location']
location_count = location_all.value_counts().to_frame().reset_index()
location_count.columns = ["location",  "number_of_images"]
location_count.loc["Total"] = location_count.sum(numeric_only=True, axis=0)
location_count

# Count the number of images per animal and print out the result.
animal_labels_all = meta_data['class']
animal_count = animal_labels_all.value_counts().to_frame().reset_index()
animal_count.columns = ["animal",  "number_of_images"]
animal_count.loc["Total"] = animal_count.sum(numeric_only=True, axis=0)
animal_count


# ## 3. Visualize the distribution of images

# Visualize the table as a pie chart
class_counts = meta_data['class'].value_counts()
utils.plot_donut_chart(class_counts)


# Plot the distribution of animals at each camera location
utils.plot_bar_chart(meta_data)


# Display a set of images from the same camera location
camera_location = 'B02'
utils.plot_random_images(meta_data, camera_location)


# Display a set of images, one from each camera location
utils.plot_images_from_all_locations(meta_data)

# Create a list of three examples that form a sequence
examples = ['data/baboon/KAR_S1_B01_R1_IMAG6738.JPG',
'data/baboon/KAR_S1_B01_R1_IMAG6737.JPG',
'data/baboon/KAR_S1_B01_R1_IMAG6736.JPG',]

# Plot the examples
utils.plot_examples(examples)


# Create a list of specific examples where the animals are hard to recognize
examples = ['data/baboon/KAR_S1_D01_R1_IMAG0158.JPG',
'data/eland/KAR_S1_E01_R1_IMAG0362.JPG',
'data/gemsbokoryx/KAR_S1_E02_R1_IMAG2502.JPG',
'data/birdother/KAR_S1_C03_R1_IMAG0361.JPG',
'data/kudu/KAR_S1_C04_R1_IMAG0833.JPG',
'data/birdother/KAR_S1_C03_R1_IMAG1214.JPG',
'data/gemsbokoryx/KAR_S1_E02_R1_IMAG2586.JPG',
'data/baboon/KAR_S1_B03_R1_IMAG0114.JPG',
'data/eland/KAR_S1_A01_R1_IMAG00129.JPG']

# Plot the examples
utils.plot_examples(examples)




