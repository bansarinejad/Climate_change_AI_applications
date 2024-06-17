#!/usr/bin/env python
# coding: utf-8

# Import Python packages.
import folium # package for animations
import folium.plugins as plugins # extras for animations
import pandas as pd # package for reading in and manipulating data
from sklearn.neighbors import KNeighborsRegressor # package for doing KNN
from datetime import datetime # package for manipulating dates

import utils # utility functions defined for this lab

print("All packages imported successfully!")

# Load the dataset with missing values filled in.
full_dataset = pd.read_csv('data/full_data_with_imputed_values.csv')
full_dataset['DateTime'] = pd.to_datetime(full_dataset['DateTime'], dayfirst=True)

full_dataset.head(5)

# Define a value for k
k = 3
# Define the target pollutant
target = 'PM2.5'
# Define a grid cell size (higher value implies a finer grid)
n_points_grid = 64
neighbors_model = KNeighborsRegressor(n_neighbors=k, weights = 'distance', metric='sqeuclidean')
# Isolate a single time step from the dataset
time_step = datetime.fromisoformat('2021-04-05T08:00:00')
time_step_data = full_dataset[full_dataset['DateTime'] == time_step]
neighbors_model.fit(time_step_data[['Latitude', 'Longitude']], time_step_data[[target]])
# Generate a map of predictions for Bogotá
predictions_xy, dlat, dlon = utils.predict_on_bogota(neighbors_model, n_points_grid)
utils.create_heat_map(predictions_xy, time_step_data, dlat, dlon, target)

# Make an estimate of mean absolute error for k=1
utils.calculate_mae_for_k(full_dataset, k=1, target_pollutant=target)

# Make an estimate of mean absolute error (MAE) for a range of k values.
kmin = 1
kmax = 7

for kneighbors in range(kmin, kmax+1):
    mae = utils.calculate_mae_for_k(full_dataset, k=kneighbors, target_pollutant=target)
    print(f'k = {kneighbors}, MAE = {mae}')

k = 3
start_date = datetime.fromisoformat('2021-08-02T08:00:00')
end_date = datetime.fromisoformat('2021-08-05T08:00:00')

utils.create_heat_map_with_date_range(full_dataset, start_date, end_date, k, target)

# Choose parameters for the animation
k = 3
n_points_grid = 64
# Filter a date range
start_date = datetime.fromisoformat('2021-08-04T08:00:00')
end_date = datetime.fromisoformat('2021-08-05T08:00:00')

# Create the features for the animation (these are the shapes that will appear on the map)
features = utils.create_animation_features(full_dataset, start_date, end_date, k, n_points_grid, target)
print('Features for the animation created successfully! Run the next cell to see the result!')

# Create the map animation using the folium library
map_animation = folium.Map(location=[4.7110, -74.0721], zoom_start = 11) 
# Add the features to the animation
plugins.TimestampedGeoJson(
    {"type": "FeatureCollection", "features": features},
    period="PT1H",
    duration='PT1H',
    add_last_point=True
).add_to(map_animation)

# Run the animation
map_animation
