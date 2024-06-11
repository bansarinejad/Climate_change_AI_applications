#!/usr/bin/env python
# coding: utf-8

# # Wind Power Forecasting: Design Phase (Part 1)

# ## 1. Import Python packages

import numpy as np # package for numerical calculations
import pandas as pd # package for reading in and manipulating data
import utils # utility functions for this lab

print('All packages imported successfully!')


# ## 2. Load the dataset

# Load the data from the csv file
raw_data = pd.read_csv("./data/wtbdata_245days.csv")

# Select only the top 10 turbines
top_turbines = utils.top_n_turbines(raw_data, 10)

# Format datetime (this takes around 15 secs)
top_turbines = utils.format_datetime(top_turbines, initial_date_str="01 05 2020")

# Print out the first few lines of data
top_turbines.head()


# ## 3. Catalog abnormal values


# Initially include all rows
top_turbines["Include"] = True

# Define conditions for abnormality
conditions = [
    np.isnan(top_turbines.Patv),
    (top_turbines.Pab1 > 89) | (top_turbines.Pab2 > 89) | (top_turbines.Pab3 > 89),
    (top_turbines.Ndir < -720) | (top_turbines.Ndir > 720),
    (top_turbines.Wdir < -180) | (top_turbines.Wdir > 180),
    (top_turbines.Patv <= 0) & (top_turbines.Wspd > 2.5)
]

# Exclude abnormal features
for condition in conditions:
    top_turbines = utils.tag_abnormal_values(top_turbines, condition)
    
top_turbines.head()


# Cut out all abnormal values
clean_data = top_turbines[top_turbines.Include].drop(["Include"], axis=1)

clean_data.head()


# ## 4. Establish a baseline for wind power estimation

utils.linear_univariate_model(clean_data)


# ## 5. Feature engineering

# Aggregate pab features
clean_data = utils.cut_pab_features(clean_data)

clean_data.head(5)


# ### 5.2 Transform angle features

# Transform all angle-encoded features
for feature in ["Wdir", "Ndir", "Pab"]:
    utils.transform_angles(clean_data, feature)  
    
clean_data.head(5)


# ### 5.3 Fix temperatures and active power


# Fix temperature values
clean_data = utils.fix_temperatures(clean_data)

# Fix negative active powers
clean_data["Patv"] = clean_data["Patv"].apply(lambda x: max(0, x))

clean_data.head(5)


# ### 5.4 Create time features

# Generate time signals
clean_data = utils.generate_time_signals(clean_data)

clean_data.head(5)

# Define predictor features 
predictors = [f for f in clean_data.columns if f not in ["Datetime", "TurbID", "Patv"]]

# Define target feature
target = ["Patv"]

# Re-arrange features before feeding into models
model_data = clean_data[["TurbID"]+predictors+target]

model_data.head(5)


# ## 6. Update linear model baseline with more features

# Create a linear model with more features
utils.linear_multivariate_model(model_data, predictors)
# Running the interaction below might take a minute


# ## 7. Use a neural network to improve wind power estimation

# Train a neural network model
utils.neural_network(model_data, predictors)





