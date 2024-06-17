#!/usr/bin/env python
# coding: utf-8

# # Air Quality: - Explore the Data

# Import packages
import pandas as pd # package for reading in and manipulating data
from datetime import datetime # package for manipulating dates

import utils # utils functions defined for this lab

print('All packages imported successfully!')


# ## 2. Load the data

# Read in the data
raw_data = pd.read_csv('data/RMCAB_air_quality_sensor_data.csv')

# Modify the DateTime column format
raw_data['DateTime'] = pd.to_datetime(utils.fix_dates(raw_data, 'DateTime'), dayfirst=True)
# Rename "OZONO" column from Spanish to English
raw_data = raw_data.rename(columns={'OZONO': 'OZONE'})

# List all of the pollutants that you will be working with
pollutants_list = ['PM2.5', 'PM10',  'NO', 'NO2', 'NOX', 'CO', 'OZONE']

# Print out the number of lines in the dataframe
print(f"The dataset contains {len(raw_data)} entries")

# Print out the first few lines in the dataframe
raw_data.head(5)


# ## 3. Count null values

# Print out a count of null values for each column in the dataset
raw_data.isnull().sum()


# ## 4. Plot histograms of different pollutants

# Define a number of bins to use (feel free to changes this and see what happens!)
number_of_bins=64

# Generate histograms
utils.create_histogram_plot(raw_data, number_of_bins)


# ## 5. Make box plots of pollutants across all sensor stations

# Generate boxplots of pollutant values for each sensor station
utils.create_boxplot(raw_data)


# ## 6. Investigate scatter plots of different pollutants and look for correlation

# Generate scatterplots of different pollutants against one another
# Different colors show various densities of points on the plot
utils.create_scatterplot(raw_data)


# ## 7. Generate simultaneous scatterplots and histograms across all pollutants

# generate a grid of histograms and scatterplots of your data
utils.plot_pairplot(raw_data, pollutants_list)


# ## 8. Construct a correlation matrix to quantitatively look for correlation

# generate a correlation matrix
utils.create_correlation_matrix(raw_data, pollutants_list)


# ## 9. Look at measurements over time for different pollutants at different stations

# define the date range to show in the plot
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 1, 31)
# generate a time series plot of pollutant data for a paricular station
utils.create_time_series_plot(raw_data, start_date, end_date)


# ## 10. Visualize the data in a map representation

# add some extra features like latitude and longitude to the data for mapping
enriched_raw_data = utils.add_extra_features(raw_data)

# choose a variable to calculate long-term averages for
x_variable = 'hour_of_day' # Options ['day_of_week', 'hour_of_day']
# choose a pollutant which you are interested in
y_variable = 'PM2.5' # Options ['PM2.5', 'PM10', 'NO', 'NO2', 'NOX', 'CO', 'OZONE']

# generate a map representation of the data
utils.create_map_with_plots(enriched_raw_data, x_variable, y_variable)

