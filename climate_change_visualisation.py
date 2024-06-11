#!/usr/bin/env python
# coding: utf-8

# # Global Temperature Change

# ## 1. Import Python Packages

# Import libraries
get_ipython().run_line_magic('matplotlib', 'notebook')

import utils 
import pandas as pd
import numpy as np
import IPython
from itables import init_notebook_mode
init_notebook_mode(all_interactive = False)
print('All packages imported successfully!')

# ## 2. Load and Inspect the Dataset

temperature_data = pd.read_csv('data/global_temperature.csv') #import the data
temperature_data.columns.values[4:] = temperature_data.columns[4:].map(int) #column names to int for easy manipulation
print('the dataset contains', len(temperature_data), 'rows')
temperature_data.tail()

# ## 3. Visualize Global Average Temperature Rise <a class="anchor" id="global"></a>

# Selection of the columns with the time series
plot_data = temperature_data[np.array(range(1880,2022,1))]
# Get difference with baseline
diff = plot_data.sub(plot_data.iloc[:, 101:132].dropna().mean(axis=1) - 0.69, axis=0)  

# Create the bar chart
utils.bar_global(diff)

# ## 4. Visualize Local Temperature Behavior Around the Globe<a class="anchor" id="local"></a>

utils.local_temp_map(temperature_data)

# ## 5. Visualize Global Temperature Anomalies <a class="anchor" id="century"></a>

utils.slider_global_temp()

# ## 6. Visualize the Impact of Temperature Rise <a class="anchor" id="consequences"></a>

#https://climate.nasa.gov/images-of-change/?id=796#796-collapsing-ice-shelf-reveals-a-possible-new-island-eastern-antarctica
src = 'https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=dab72496-6d1f-11ed-b5bd-6595d9b17862'
IPython.display.IFrame(src, width = 700, height = 720)


# ### Melting Glaciers in Tibet <a class="anchor" id="tibet"></a>

#https://climate.nasa.gov/images-of-change/?id=778#778-melting-glaciers-enlarge-lakes-on-tibetan-plateau
src = 'https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=269f7416-6d21-11ed-b5bd-6595d9b17862'
IPython.display.IFrame(src, width = 750, height = 520)


# ### Glacier Retreat in Alaska <a class="anchor" id="alaska"></a>

#https://climate.nasa.gov/images-of-change?id=777#777-grand-plateau-glacier-retreats
src = 'https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=0c37b40e-7027-11ed-b5bd-6595d9b17862'
IPython.display.IFrame(src, width = 700, height = 550)




