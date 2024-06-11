#!/usr/bin/env python
# coding: utf-8

# # Wind Power Prediction: Explore Phase
# 
# 
# In this first lab you will perform an exploratory analysis of the Spatial Dynamic Wind Power Forecasting or [SDWPF dataset](https://arxiv.org/abs/2208.04360), which contains data from 134 wind turbines in a wind farm in China. The SDWPF data was provided by the Longyuan Power Group, which is the largest wind power producer in China and Asia. This dataset was used in the [Baidu KDD Cup in 2022](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction) where teams competed for $35,000 in prize money. 
# 


import pandas as pd # package for reading in and manipulating the data
import seaborn as sns # package for data visualization
import ipywidgets as widgets # package for creating interactive visuals
import matplotlib.pyplot as plt # package for making plots
from IPython.display import display # package for displaying visuals
import utils # utility functions provided for this lab

print('All packages imported successfully!')


# ## 2. Load the dataset

# Load the data from the csv file
raw_data = pd.read_csv("./data/wtbdata_245days.csv")

# Add units to numerical features
raw_data.columns = ["TurbID", "Day", "Tmstamp", "Wspd (m/s)", "Wdir (°)", "Etmp (°C)", "Itmp (°C)", "Ndir (°)", "Pab1 (°)", "Pab2 (°)", "Pab3 (°)", "Prtv (kW)", "Patv (kW)"]

# Print the first 5 rows of the dataset
raw_data.head(5)


# You can look at the [paper](https://arxiv.org/abs/2208.04360) to learn more about the dataset.
# 
# Every entry (or row) in the dataset contains the following information:
# 
# - `TurbID`: Wind turbine identification number.
# 
# 
# - `Day`: The number of the day represented as a string (first day is May 1st 2020).
# 
# 
# - `Tmstamp`: The hour and minute of the date of the measurement.
# 
# 
# - `Wspd`: The wind speed recorded by the anemometer measured in meters per second.
# 
# 
# - `Wdir`: The angle between the wind direction and the position of turbine nacelle measured in degrees.
# 
# 
# - `Etmp`: Temperature of the surounding environment measured in degrees Celsius.
# 
# 
# - `Itmp`: Temperature inside the turbine nacelle measured in degrees Celsius.
# 
# 
# - `Ndir`: Nacelle direction, i.e., the yaw angle of the nacelle measured in degrees.
# 
# 
# - `Pab1`: Pitch angle of blade 1 measured in degrees.
# 
# 
# - `Pab2`: Pitch angle of blade 2 measured in degrees.
# 
# 
# - `Pab3`: Pitch angle of blade 3 measured in degrees.
# 
# 
# - `Prtv`: Reactive power measured in kW.
# 
# 
# - `Patv`: Active power measured in kW → **note: this is the target variable you will be trying to predict**.
# 

# ## 3. Inspect and address missing values

# Print missing values per feature
print("Number of missing values per column:\n")
print(raw_data.isnull().sum())

# Dataframe with only missing values
mv = raw_data[raw_data.isnull().any(axis=1)]

# Compute missing and total values
num_na_values, total_values = len(mv), len(raw_data)
print(f"\nThe dataset contains {total_values} rows, of which {num_na_values} are missing.\n\nThis is {(num_na_values/total_values)*100:.3f}% of the total data.")

# Display the button to inspect values
button = widgets.Button(description="Inspect 🔍")
output = widgets.Output()
display(button, output)

# Number of samples to randomnly inspect with each click
num_samples = 5 # You can change this value

# Inspect missing values by clicking button
button.on_click(utils.inspect_missing_values(mv, num_samples, output))

# Drop missing values
raw_data = raw_data.dropna()

# Print missing values per feature
print("Number of missing values per feature:\n")
print(raw_data.isnull().sum())


# ## 4. Calculate descriptive statistics

# Make a list of all columns in the dataset
all_features = list(raw_data.columns)
# Make a list of only the columns containing numerical features
numerical_features = [f for f in all_features if f not in ["TurbID", "Day", "Tmstamp"]]

print(f"The numerical features are:\n\n{numerical_features}")

# Create a dataframe with descriptive statistics
descriptive_stats = raw_data[numerical_features].describe()
# Format the dataframe to show three decimal places and display the dataframe
descriptive_stats.apply(lambda s: s.apply('{0:.3f}'.format))


# ## 5. Select a subset of turbines

# Keep the data for a 10-turbine wind farm
top_turbines = utils.top_n_turbines(raw_data, 10)


# ## 6. Visualize the data

# ### 6.1 Histograms

# Define a number of bins to use (feel free to changes this and see what happens!)
bin_size=64

# Generate histograms
utils.histogram_plot(top_turbines, numerical_features, bin_size)


# #### Histogram comparison between two turbines

# Define a number of bins to use (feel free to changes this and see what happens!)
bin_size=32

# Generate histograms
utils.histogram_comparison_plot(top_turbines, numerical_features, bin_size)


# ### 6.2 Box plots and violin plots

# Generate the box/violin plots across all turbines
utils.box_violin_plot(top_turbines, numerical_features)


# ### 6.3 Scatterplots

# Generate the per turbine scatterplot for any two features
utils.scatterplot(top_turbines, numerical_features)


# ### 6.4 Pairplot

# Identify a turbine and fraction of data to plot (you can changes these values!)
turb_id = 6     # the id of the turbine
fraction = 0.01 # the fraction of the rows to include in the plot

utils.plot_pairplot(top_turbines, turb_id, numerical_features, fraction)


# ### 6.5 Correlation matrix

# Generate a corrlation matrix
utils.correlation_matrix(top_turbines[numerical_features])


# ### 6.6  Time Series 

# Create proper datetimes (this takes around 15 secs)
raw_data = utils.format_datetime(top_turbines, initial_date_str="01 05 2020")

raw_data.head(5)


# #### Plot time series

# Generate a time series plot of features for a paricular turbine
utils.plot_time_series(top_turbines, numerical_features)


# ####  Time Series for a pair of turbines

# Generate a time series plot of features for a pair of turbines
utils.time_series_turbine_pair(top_turbines, numerical_features)


