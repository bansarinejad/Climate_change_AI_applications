#!/usr/bin/env python
# coding: utf-8

# # Wind Power Forecasting: Design Phase (Part 2)

# ## 1. Import Python packages.

import pandas as pd # package for reading in and manipulating the data
import utils # utility functions for this lab

print('All packages imported successfully!')

# ## 2. Load the dataset

# Load the data into a pandas dataframe
curated_data = pd.read_csv("data/wind_data.csv")

# Print out the turbines that are included in the dataset
print(f'Turbines included in the dataset: {curated_data.TurbID.unique()}')

curated_data.head(5)

# ## 3. Slice your data and resample to an hourly frequency

# Select the turbine you will use
turb_id = 6

# Prepare data for feeding into the network
data = utils.prepare_data(curated_data, turb_id)

data.head(5)

# ## 4. Split your data into train, validation, and test sets

# Split the data into training, validation and testing
data_splits = utils.train_val_test_split(data)

print("training, validation and testing splits successfully created and normalized.")


# ## 5. Establish a baseline

# Compute the random baseline
random_window, random_baseline = utils.random_forecast(data_splits)

# Plot the predicted vs real values for the test split
random_baseline_mae = random_window.plot_long(random_baseline, data_splits)


# ### 5.2. Replicate the last 24 hours (tomorrow == today)

# Create a model by predicting the next 24 hours of wind power will be the same as the previous 24 hours
utils.interact_repeat_forecast(data_splits, baseline_mae=random_baseline_mae)


# ### 5.3. Compute a moving average

# Compute the baseline (try changing the n_days parameter)
moving_avg_window, moving_avg_baseline = utils.moving_avg_forecast(data_splits, n_days=1)

# Plot the predicted vs real values for the test split
utils.prediction_plot(moving_avg_window.plot_long, moving_avg_baseline, data_splits, baseline_mae=random_baseline_mae)


# ## 6. Visualize a time series of your target and predictors

# Generate time series plot showing train, validation, and test data
utils.plot_time_series(data_splits)


# ## 7. Train neural network models using historical data

# Plot input and output sequences
utils.window_plot(data_splits)


# ### 7.2. Train a neural network using only the target's history

# Only using the target without any other predictors
features = ["Patv"]

# Compute the forecasts
window, model, _data_splits = utils.train_conv_lstm_model(data, features, days_in_past=1)

# Plot the predicted vs real values for the test split
utils.prediction_plot(window.plot_long, model, _data_splits, baseline_mae=random_baseline_mae)


# ### 7.3. Train a neural network using all the features history

# Now using all features
features = list(data.columns)

# Compute the forecasts
window, model, _data_splits = utils.train_conv_lstm_model(data, features, days_in_past=1)

# Plot the predicted vs real values for the test split
utils.prediction_plot(window.plot_long, model, _data_splits, baseline_mae=random_baseline_mae)


# ## 8. Include wind speed forecasts in your model

# Create a new dataframe with the perfect forecasts
data_with_perfect_wspd_forecasts = utils.add_wind_speed_forecasts(data)

# Include all features along with the external forecasts
features = list(data_with_perfect_wspd_forecasts.columns)

# Compute the power output forecasts
window, model, _data_splits = utils.train_conv_lstm_model(data_with_perfect_wspd_forecasts, features, days_in_past=1)

# Plot the predicted vs actual values for the test split
utils.prediction_plot(window.plot_long, model, _data_splits, baseline_mae=random_baseline_mae)


# ### 8.2 Generate "synthetic" wind speed forecasts

# Load the data
weather_forecasts = utils.load_weather_forecast()

# Plot measured wind speed and forecasts for different periods into the future
utils.plot_forecast(weather_forecasts)

# Plot wind speed error as a function of hours into the future for different locations
utils.plot_mae_forecast(weather_forecasts)

# Create a new dataframe with the forecasts
data_with_wspd_forecasts = utils.add_wind_speed_forecasts(data, add_noise="linearly_increasing")

utils.plot_forecast_with_noise(data_with_wspd_forecasts)


# ### 8.3 Train a neural network model using synthetic wind speed forecasts

# Create a new dataframe with the external forecasts
data_with_wspd_forecasts = utils.add_wind_speed_forecasts(data, add_noise="mimic_real_forecast")

# Include all features along with the external forecasts
features = list(data_with_wspd_forecasts.columns)

# Compute the forecasts
window, model, _data_splits = utils.train_conv_lstm_model(data_with_wspd_forecasts, features, days_in_past=1)

# Plot the predicted vs real values for the test split
utils.prediction_plot(window.plot_long, model, _data_splits, baseline_mae=random_baseline_mae)

