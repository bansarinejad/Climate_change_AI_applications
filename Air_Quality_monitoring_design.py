#!/usr/bin/env python
# coding: utf-8

# #  Air Quality: Design Phase

# Import packages
import pandas as pd # package for reading in and manipulating data
from datetime import datetime # package for manipulating dates
from sklearn.model_selection import train_test_split # package for splitting data

import utils # utility functions defined for this lab

print('All packages imported successfully!')

# ## 2. Load the data

# Read in the data
raw_data = pd.read_csv('data/RMCAB_air_quality_sensor_data.csv')
# Modify the DateTime column format
raw_data['DateTime'] = pd.to_datetime(utils.fix_dates(raw_data, 'DateTime'), dayfirst=True)
# Rename columns from Spanish to English
raw_data = raw_data.rename(columns={'OZONO': 'OZONE'})
# print out the number of rows in the dataframe
print(f"The dataset contains {len(raw_data)} entries")
# Print out the first few lines in the dataframe
raw_data.head(5)


# ## 3. Add location (latitude and longitude of sensor stations) to the dataframe

# Read in csv file containing location data and format appropriately
stations = pd.read_csv('data/stations_loc.csv')
stations = stations[['Sigla', 'Latitud', 'Longitud']]
# Rename the columns from Spanish to English
stations = stations.rename(columns={'Sigla': 'Station', 'Latitud': 'Latitude', 'Longitud': 'Longitude'})
# Parse the dates
stations['Latitude'] = stations['Latitude'].apply(utils.parse_dms)
stations['Longitude'] = stations['Longitude'].apply(utils.parse_dms)
# Add location data as extra columns to air pollution dataset
raw_data_enriched = pd.merge(raw_data, stations, on='Station', how='inner')
raw_data_enriched.head(5)


# ## 4. Visualize the extent and distribution of missing data

# Print out a count of missing data for each column in the dataset
raw_data_enriched.isnull().sum()


# ### 4.2 Visualize missing data in a time series

# Plot a time series for a particular date range, pollutant, and sensor station
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 1, 31)
utils.create_time_series_plot(raw_data, start_date, end_date)


# ### 4.3 Visualize the distribution of gaps in the data

# Plot the distribution of gap sizes in the data
utils.plot_distribution_of_gaps(raw_data, target='PM2.5')


# ## 5. Visualize simple methods for estimating missing values

# Specify the date to plot
day = datetime(2021, 5, 6)

# Create a plot to display the linear interpolation for a range of hours
utils.visualize_missing_values_estimation(raw_data_enriched, day);


# ## 6. Run the nearest neighbor method to establish a baseline

# Create a nearest neighbor model and run it on your test data
regression_scores = {}
regression_scores['baseline_model'] = utils.calculate_mae_for_nearest_station(raw_data_enriched, target='PM2.5')
print(regression_scores['baseline_model'])


# ## 7. Prepare the data to train a neural network model

# Define the variable you want to predict
target = 'PM2.5'

# Create new columns for day of week and hour of day
raw_data_for_imputing = raw_data_enriched.copy()
raw_data_for_imputing['day_week'] = pd.DatetimeIndex(raw_data_for_imputing['DateTime']).weekday
raw_data_for_imputing['hour'] = pd.DatetimeIndex(raw_data_for_imputing['DateTime']).hour

# Create a numerical representation of station ID and add as extra columns to the dataframe
one_hot = pd.get_dummies(raw_data_for_imputing.Station, prefix='Station')
raw_data_for_imputing = raw_data_for_imputing.join(one_hot)

# Make a copy of the dataframe before dropping rows with missing values
data_no_missing = raw_data_for_imputing.copy()  
# Drop all rows containing missing values
data_no_missing.dropna(inplace=True)
# Print out the number of missing values in the PM2.5 column as a check (should print 0)
null_remaining = data_no_missing[target].isnull().sum()
if null_remaining == 0: 
    print('missing values removed and data prepared successfully!')


# ### 7.2 Split data into training and testing sets


train_data, test_data = train_test_split(data_no_missing, test_size=0.20, random_state=57)

print(f'Train dataset size: {train_data.shape}')
print(f'Test dataset size: {test_data.shape}')



# Define the features you will base your predictions on 
pollutants_list = ['PM10','PM2.5','NO','NO2','NOX','CO','OZONE']
pollutants_except_target = [i for i in pollutants_list if i != target]
feature_names = ['day_week', 'hour'] + list(one_hot.columns) + pollutants_except_target

# Define the neural network model architecture
model = utils.build_keras_model(input_size=len(feature_names))

# Train and test the model
model, scaler, mae = utils.train_and_test_model(feature_names, target, train_data, test_data, model)
regression_scores['neural_network_model'] = mae


# Print out the MAE result
for model_name, model_score in regression_scores.items():
    print(model_name, ' : ', model_score)

# ## 9. Visualize the results from your baseline model compared to the neural network.

start_date = datetime(2021, 2, 1)
end_date = datetime(2021, 2, 3)

utils.create_plot_with_preditions(data_no_missing, model, scaler, feature_names, target,  start_date, end_date)

# ## 10. Estimate missing sensor measurements across all pollutants. 

# Estimate non-PM2.5 missing values with the nearest neighbor method
imputed_with_baseline_model = utils.impute_nontarget_missing_values_interpolate(raw_data_for_imputing, feature_names, target)
# Estimate missing PM2.5 values with 
data_with_imputed = utils.impute_target_missing_values_neural_network(raw_data_for_imputing, model, scaler, imputed_with_baseline_model, target)
# Print out a count of missing data for each column in the dataset with imputed values
data_with_imputed[pollutants_list].isnull().sum()

# Print out a sample of the dataframe
data_with_imputed.sample(25)

# ## 11. Visualize the results of filling in missing PM2.5 values


# Define a start and end date and visualize missing values that have been imputed
start_date = datetime(2021, 3, 1)
end_date = datetime(2021, 4, 30)
utils.create_time_series_plot(data_with_imputed, start_date, end_date)


#write the resulting dataset to a new csv file

#data_with_imputed.to_csv('full_data_with_imputed_values.csv')
