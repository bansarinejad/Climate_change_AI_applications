import random
import pickle
import os
import shap
import torch
import mplcursors
import base64
import folium
import IPython
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
from matplotlib import cm, colors
from IPython.display import clear_output, display
from datetime import datetime, timedelta
from ipywidgets import interact, interact_manual, fixed
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.inspection import permutation_importance
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Dict, Optional

def bar_global(data: pd.core.frame.DataFrame) -> None:
    '''
    Plotting function that gets the temperature timeseries dataframe as input
    and creates and interactive bar plot with the average differences per year from 1880 to 2021.

    data: Wide timeseries dataframe with temperature for multiple weather stations in Celsius. Stations are in the rows,
    years are in the columns.
    '''

    values = np.array(data.mean())  # get global average difference per year
    clrs = ["blue" if (x < 0) else "red" for x in values]  # colors for positive and negative differences
    fig, ax = plt.subplots(figsize=(9, 4))  # set plot size
    line = ax.bar(np.array(range(1880, 2022, 1)), values, color=clrs)  # bar plot
    ax.set_xlim(1879, 2022)  # set axis limits
    plt.title("Station network temperature difference \nwith respect to 1850-1900 average", fontsize=12)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Temperature ($ ^{\circ}$C)", fontsize=12)

    # cursor interaction for creating labels
    cursor = mplcursors.cursor()

    @cursor.connect("add")
    def on_add(sel):
        x, y, width, height = sel.artist[sel.index].get_bbox().bounds
        sel.annotation.get_bbox_patch().set(fc="white", alpha=1)
        sel.annotation.set(
            text=f"Year {int(x) + 1} \n {values[int(x) - 1880 + 1]:.2f}" + "C" + "$^{\circ}$",
            position=(0, 20),
            anncoords="offset points",
        )
        sel.annotation.xy = (x + width / 2, height)
        sel.axvline(x=x, color="k")

    # show plot
    plt.show()


def local_temp_map(df: pd.core.frame.DataFrame) -> folium.Map:
    '''
    Plotting function that gets the temperature timeseries dataframe with additional information and returns an interactive map
    with bar plots for each station.

    df: Wide timeseries dataframe with temperature for multiple weather stations in Celsius. Stations are in the rows,
    years are in the columns. Contains geographical coordinates and station names.
    '''

    folium_map = folium.Map(
        location=[35, 0],
        zoom_start=1.5,
        tiles="cartodb positron",
        max_bounds=False,
    )
    loc = df[["LATITUDE", "LONGITUDE"]]
    width, height = 500, 230

    for index, location_info in loc.iterrows():
        png = f"plots/{df['STATION'].loc[index]}.png"
        encoded = base64.b64encode(open(png, "rb").read())
        html = '<img src="data:image/png;base64,{}" style="height:100%";>'.format
        iframe = folium.IFrame(
            html(encoded.decode("UTF-8")), width=width, height=height
        )
        popup = folium.Popup(iframe, max_width=2650)
        folium.Marker(
            [location_info['LATITUDE'], location_info['LONGITUDE']],
            popup=popup,
            icon=folium.Icon(color='black', icon_color='white'),
        ).add_to(folium_map)

    return folium_map


def slider_global_temp() -> None:
    '''
    Creates an interactive slider that shows the global temperature around the globe from 1884 to 2020
    '''

    # set up plot
    out = widgets.Output()

    def update(Year=1884):
        with out:
            clear_output(wait=True)
            display(IPython.display.Image(f'data/NASA/{Year}.png'))

    slider = widgets.IntSlider(
        min=1884, max=2020, layout=widgets.Layout(width='95%')
    )
    widgets.interactive(update, Year=slider)

    layout = widgets.Layout(
        flex_flow='column', align_items='center', width='700px'
    )

    wid = widgets.HBox(children=(slider, out), layout=layout)
    display(wid)

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 25
FONT_SIZE_AXES = 20


def plot_turbines(raw_data: pd.core.frame.DataFrame):
    """Plot turbines' relative positions.

    Args:
        raw_data (pd.core.frame.DataFrame): The dataset used.
    """
    turb_locations = pd.read_csv("./data/turb_location.csv")
    turbs = turb_locations[turb_locations.TurbID.isin(raw_data.TurbID.unique())]
    turbs = turbs.reset_index()
    n = list(raw_data.TurbID.unique())

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title("Spatial location of wind turbines")
    ax.scatter(turbs.x, turbs.y, marker="1", s=500, c="green")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for i, txt in enumerate(n):
        ax.annotate(txt, (turbs.x[i], turbs.y[i]))


def top_n_turbines(
    raw_data: pd.core.frame.DataFrame, n: int
) -> pd.core.frame.DataFrame:
    """Keeps only the top n turbines that produced more energy on average.

    Args:
        raw_data (pd.core.frame.DataFrame): The full dataset.
        n (int): Desired number of turbines to keep.

    Returns:
        pd.core.frame.DataFrame: The dataset with only the data from the top n turbines.
    """
    sorted_patv_by_turbine = (
        raw_data.groupby("TurbID").mean()["Patv (kW)"].sort_values(ascending=False)
    )

    top_turbines = list(sorted_patv_by_turbine.index)[:n]

    print(
        f"Original data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines.\n"
    )

    raw_data = raw_data[raw_data["TurbID"].isin(top_turbines)]

    print(
        f"Sliced data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines."
    )

    return raw_data


def format_datetime(
    data: pd.core.frame.DataFrame, initial_date_str: str
) -> pd.core.frame.DataFrame:
    """Formats Day and Tmstamp features into a Datetime feature.

    Args:
        data (pd.core.frame.DataFrame): The original dataset.
        initial_date_str (str): The initial date.

    Returns:
        pd.core.frame.DataFrame: The dataframe with formatted datetime.
    """
    if "Datetime" in data.columns:
        return data

    initial_date = datetime.strptime(initial_date_str, "%d %m %Y").date()

    data["Date"] = data.apply(
        lambda x: str(initial_date + timedelta(days=(x.Day - 1))), axis=1
    )

    data["Datetime"] = data.apply(
        lambda x: datetime.strptime(f"{x.Date} {x.Tmstamp}", "%Y-%m-%d %H:%M"), axis=1
    )

    data.drop(["Day", "Tmstamp", "Date"], axis=1, inplace=True)

    data = data[["Datetime"] + [col for col in list(data.columns) if col != "Datetime"]]

    return data


def inspect_missing_values(
    mv_df: pd.core.frame.DataFrame, num_samples: int, output: widgets.Output
):
    """Interactive dataframe inspector to visualize missing values.

    Args:
        mv_df (pd.core.frame.DataFrame): Dataframe with missing values.
        num_samples (int): Number of samples to inspect at any given time.
        output (widgets.Output): Output of the widget (this is for visualization purposes)
    """

    def on_button_clicked(b):
        with output:
            output.clear_output()
            random_index = random.sample([*range(len(mv_df))], num_samples)
            display(mv_df.iloc[random_index].head(num_samples))

    return on_button_clicked


def histogram_plot(df: pd.core.frame.DataFrame, features: List[str], bins: int = 16):
    """Create interactive histogram plots.

    Args:
        df (pd.core.frame.DataFrame): The dataset used.
        features (List[str]): List of features to include in the plot.
        bins (int, optional): Number of bins in the histograms. Defaults to 16.
    """

    def _plot(turbine, feature):
        data = df[df.TurbID == turbine]
        plt.figure(figsize=(8, 5))
        x = data[feature].values
        plt.xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        sns.histplot(x, bins=bins)
        plt.ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        plt.title(f"Feature: {feature} - Turbine: {turbine}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(
        options=df.TurbID.unique(), value=df.TurbID.unique()[-1], description="Turbine"
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    interact(_plot, turbine=turbine_selection, feature=feature_selection)


def histogram_comparison_plot(
    df: pd.core.frame.DataFrame, features: List[str], bins: int = 16
):
    """Create interactive histogram plots.

    Args:
        df (pd.core.frame.DataFrame): The dataset used.
        features (List[str]): List of features to include in the plot.
        bins (int, optional): Number of bins in the histograms. Defaults to 16.
    """

    def _plot(turbine1, turbine2, feature):
        data_1 = df[df.TurbID == turbine1]
        data_2 = df[df.TurbID == turbine2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

        x_1 = data_1[feature].values
        x_2 = data_2[feature].values

        ax1.set_xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)

        ax1.set_ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        ax2.set_ylabel(f"Count", fontsize=FONT_SIZE_AXES)

        sns.histplot(x_1, bins=bins, ax=ax1)
        sns.histplot(x_2, bins=bins, ax=ax2, color="green")

        ax1.set_title(f"Turbine: {turbine1}", fontsize=FONT_SIZE_TITLE)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        ax2.set_title(f"Turbine: {turbine2}", fontsize=FONT_SIZE_TITLE)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        fig.tight_layout()
        fig.show()

    turbine_selection1 = widgets.Dropdown(
        options=df.TurbID.unique(),
        value=df.TurbID.unique()[-2],
        description="Turbine ID",
        style={"description_width": "initial"},
    )

    turbine_selection2 = widgets.Dropdown(
        options=df.TurbID.unique(),
        value=df.TurbID.unique()[-1],
        description="Another Turbine ID",
        style={"description_width": "initial"},
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    interact(
        _plot,
        turbine1=turbine_selection1,
        turbine2=turbine_selection2,
        feature=feature_selection,
    )


def box_violin_plot(df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive violin/box plots for the dataset.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        features (List[str]): List of features to include in the plot.
    """
    labels = df["TurbID"].unique()

    def _plot(feature="Wspd", plot_type="box"):
        plt.figure(figsize=(18, 8))
        scale = "linear"
        plt.yscale(scale)
        if plot_type == "violin":
            sns.violinplot(
                data=df, y=feature, x="TurbID", order=labels, color="seagreen"
            )
        elif plot_type == "box":
            sns.boxplot(data=df, y=feature, x="TurbID", order=labels, color="seagreen")
        plt.title(f"Feature: {feature}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.xlabel(f"TurbID", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        plt.show()

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    plot_type_selection = widgets.Dropdown(
        options=["violin", "box"], description="Plot Type"
    )

    interact(_plot, feature=feature_selection, plot_type=plot_type_selection)


def scatterplot(df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive scatterplots of the data.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        features (List[str]): List of features to include in the plot.
    """
    df_clean = df.dropna(inplace=False)

    def _plot(turbine, var_x, var_y):
        plt.figure(figsize=(12, 6))
        df_clean_2 = df_clean[df_clean.TurbID == turbine]
        x = df_clean_2[var_x].values
        y = df_clean_2[var_y].values

        plt.plot(
            x, y,
            marker='o', markersize=3, markerfacecolor='blue', 
            markeredgewidth=0,
            linestyle='', 
            alpha=0.5
        )
        
        
        plt.xlabel(var_x, fontsize=FONT_SIZE_AXES)
        plt.ylabel(var_y, fontsize=FONT_SIZE_AXES)

        plt.title(f"Scatterplot of {var_x} against {var_y}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(
        options=df.TurbID.unique(), value=df.TurbID.unique()[-1], description="Turbine"
    )

    x_var_selection = widgets.Dropdown(options=features, description="X-Axis")

    y_var_selection = widgets.Dropdown(
        options=features, description="Y-Axis", value="Patv (kW)"
    )

    interact(
        _plot,
        turbine=turbine_selection,
        var_x=x_var_selection,
        var_y=y_var_selection,
    )


def correlation_matrix(data: pd.core.frame.DataFrame):
    """Plots correlation matrix for a given dataset.

    Args:
        data (pd.core.frame.DataFrame): The dataset used.
    """
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, cbar=False, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Features")
    plt.show()


def plot_time_series(df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive plots for the time series in the dataset.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        features (List[str]): Features to include in the plot.
    """

    def plot_time_series(turbine, feature, date_range, fix_temps):
        data = df[df.TurbID == turbine]
        if fix_temps:
            min_etemp = data["Etmp (°C)"].quantile(0.01)
            data["Etmp (°C)"] = data["Etmp (°C)"].apply(
                lambda x: np.nan if x < min_etemp else x
            )
            data["Etmp (°C)"] = data["Etmp (°C)"].interpolate()
            min_itemp = data["Itmp (°C)"].quantile(0.01)
            data["Itmp (°C)"] = data["Itmp (°C)"].apply(
                lambda x: np.nan if x < min_itemp else x
            )
            data["Itmp (°C)"] = data["Itmp (°C)"].interpolate()

        data = data[data.Datetime > date_range[0]]
        data = data[data.Datetime < date_range[1]]
        plt.figure(figsize=(15, 5))
        plt.plot(data["Datetime"], data[feature], "-")
        plt.title(f"Time series of {feature}", fontsize=FONT_SIZE_TITLE)
        plt.ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(
        options=df.TurbID.unique(),
        value=df.TurbID.unique()[-1],
        description="Turbine ID",
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    dates = pd.date_range(datetime(2020, 5, 1), datetime(2020, 12, 31), freq="D")

    options = [(date.strftime("%b %d"), date) for date in dates]
    index = (0, len(options) - 1)

    date_slider_selection = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Date (2020)",
        orientation="horizontal",
        layout={"width": "550px"},
    )

    fix_temps_button = widgets.Checkbox(
        value=False, description="Fix Temperatures", disabled=False
    )

    interact(
        plot_time_series,
        turbine=turbine_selection,
        feature=feature_selection,
        date_range=date_slider_selection,
        fix_temps=fix_temps_button,
    )


def time_series_turbine_pair(original_df: pd.core.frame.DataFrame, features: List[str]):
    """Creates interactive plots for the time series for a pair of turbines in the dataset.

    Args:
        original_df (pd.core.frame.DataFrame): The data used.
        features (List[str]): Features to include in the plot.
    """

    def plot_time_series(turbine_1, turbine_2, feature, date_range, fix_temps):
        df = original_df
        if fix_temps:
            df_2 = original_df.copy(deep=True)

            min_etemp = df_2["Etmp (°C)"].quantile(0.01)
            df_2["Etmp (°C)"] = df_2["Etmp (°C)"].apply(
                lambda x: np.nan if x < min_etemp else x
            )
            df_2["Etmp (°C)"] = df_2["Etmp (°C)"].interpolate()
            min_itemp = df_2["Itmp (°C)"].quantile(0.01)
            df_2["Itmp (°C)"] = df_2["Itmp (°C)"].apply(
                lambda x: np.nan if x < min_itemp else x
            )
            df_2["Itmp (°C)"] = df_2["Itmp (°C)"].interpolate()
            df = df_2

        data_1 = df[df.TurbID == turbine_1]
        data_1 = data_1[data_1.Datetime > date_range[0]]
        data_1 = data_1[data_1.Datetime < date_range[1]]

        data_2 = df[df.TurbID == turbine_2]
        data_2 = data_2[data_2.Datetime > date_range[0]]
        data_2 = data_2[data_2.Datetime < date_range[1]]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
        ax1.plot(data_1["Datetime"], data_1[feature], "-")
        ax1.set_title(f"Time series of {feature} for turbine {turbine_1}", fontsize=FONT_SIZE_TITLE)
        ax2.plot(data_2["Datetime"], data_2[feature], "-", c="green")
        ax2.set_title(f"Time series of {feature} for turbine {turbine_2}", fontsize=FONT_SIZE_TITLE)
        ax1.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax1.set_xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel(f"Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.tight_layout()
        plt.show()

    turbine_selection_1 = widgets.Dropdown(
        options=original_df.TurbID.unique(),
        value=original_df.TurbID.unique()[-2],
        description="Turbine ID",
    )

    turbine_selection_2 = widgets.Dropdown(
        options=original_df.TurbID.unique(),
        value=original_df.TurbID.unique()[-1],
        description="Another Turbine ID",
        style={"description_width": "initial"},
    )

    feature_selection = widgets.Dropdown(
        options=features,
        description="Feature",
    )

    fix_temps_button = widgets.Checkbox(
        value=False, description="Fix Temperatures", disabled=False
    )

    dates = pd.date_range(datetime(2020, 5, 1), datetime(2020, 12, 31), freq="D")

    options = [(date.strftime("%b %d"), date) for date in dates]
    index = (0, len(options) - 1)

    date_slider_selection = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description="Date (2020)",
        orientation="horizontal",
        layout={"width": "550px"},
    )

    interact(
        plot_time_series,
        turbine_1=turbine_selection_1,
        turbine_2=turbine_selection_2,
        feature=feature_selection,
        date_range=date_slider_selection,
        fix_temps=fix_temps_button,
    )

    
def plot_pairplot(
    original_df: pd.core.frame.DataFrame,
    turb_id: int,
    features: List[str],
    fraction: float=0.01
):
    """Creates a pairplot of the features.

    Args:
        df (pd.core.frame.DataFrame): The data used.
        turb_id (int): Selected turbine ID
        features (List[str]): List of features to include in the plot.
        fraction (float): amount of data to plot, to reduce time.
    """
    data_single_turbine = original_df[original_df.TurbID==turb_id][features]
    data_single_turbine = data_single_turbine.sample(frac=fraction)
    with sns.plotting_context(rc={"axes.labelsize":20}):
        sns.pairplot(data_single_turbine)
    plt.show()

def fix_temperatures(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Replaces very low temperature values with linear interpolation.

    Args:
        data (pd.core.frame.DataFrame): The dataset.

    Returns:
        pd.core.frame.DataFrame: Dataset with fixed temperatures.
    """
    min_etemp = data["Etmp"].quantile(0.01)
    data["Etmp"] = data["Etmp"].apply(lambda x: np.nan if x < min_etemp else x)
    data["Etmp"] = data["Etmp"].interpolate()
    min_itemp = data["Itmp"].quantile(0.01)
    data["Itmp"] = data["Itmp"].apply(lambda x: np.nan if x < min_itemp else x)
    data["Itmp"] = data["Itmp"].interpolate()

    return data


def tag_abnormal_values(
    df: pd.core.frame.DataFrame, condition: pd.core.series.Series
) -> pd.core.frame.DataFrame:
    """Determines if a given record is an abnormal value.

    Args:
        df (pd.core.frame.DataFrame): The dataset used.
        condition (pd.core.series.Series): Series that includes if a record meets one of the conditions for being an abnormal value.

    Returns:
        pd.core.frame.DataFrame: Dataset with tagger abnormal values.
    """
    indexes = df[condition].index
    df.loc[indexes, "Include"] = False
    return df


def cut_pab_features(raw_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Deletes redundant Pab features from dataset.

    Args:
        raw_data (pd.core.frame.DataFrame): The dataset used.

    Returns:
        pd.core.frame.DataFrame: The dataset without the redundant Pab features.
    """

    raw_data = raw_data.drop(["Pab2", "Pab3"], axis=1)
    raw_data = raw_data.rename(columns={"Pab1": "Pab"})

    return raw_data


def generate_time_signals(raw_data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Creates time signal features (time-of-day) for the data.

    Args:
        raw_data (pd.core.frame.DataFrame): The dataset uded.

    Returns:
        pd.core.frame.DataFrame: The dataset with the new features.
    """
    if "Day sin" in raw_data.columns:
        return raw_data

    date_time = pd.to_datetime(raw_data.Datetime, format="%Y-%m-%d %H:%M")
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60

    raw_data["Time-of-day sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    raw_data["Time-of-day cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    return raw_data

def format_datetime(
    data: pd.core.frame.DataFrame, initial_date_str: str
) -> pd.core.frame.DataFrame:
    """Formats Day and Tmstamp features into a Datetime feature.

    Args:
        data (pd.core.frame.DataFrame): The original dataset.
        initial_date_str (str): The initial date.

    Returns:
        pd.core.frame.DataFrame: The dataframe with formatted datetime.
    """
    if "Datetime" in data.columns:

        return data

    initial_date = datetime.strptime(initial_date_str, "%d %m %Y").date()

    data["Date"] = data.apply(
        lambda x: str(initial_date + timedelta(days=(x.Day - 1))), axis=1
    )

    data["Datetime"] = data.apply(
        lambda x: datetime.strptime(f"{x.Date} {x.Tmstamp}", "%Y-%m-%d %H:%M"), axis=1
    )

    data.drop(["Day", "Tmstamp", "Date"], axis=1, inplace=True)

    data = data[["Datetime"] + [col for col in list(data.columns) if col != "Datetime"]]

    return data


def transform_angles(
    data: pd.core.frame.DataFrame, feature: str, drop_original: bool = True
):
    """Transform angles into their Sin/Cos encoding.

    Args:
        data (pd.core.frame.DataFrame): The dataset used.
        feature (str): Name of the angle feature.
        drop_original (bool, optional): Wheter to drop the original column from the dataset. Defaults to True.
    """
    # np.cos and np.sin expect angles in radians
    rads = data[feature] * np.pi / 180

    # Compute Cos and Sin
    data[f"{feature}Cos"] = np.cos(rads)
    data[f"{feature}Sin"] = np.sin(rads)

    if drop_original:
        data.drop(feature, axis=1, inplace=True)


def plot_wind_speed_vs_power(
    ax: plt.Axes,
    x1: Iterable,
    y1: Iterable,
    x2: Iterable,
    y2: Iterable
):
    """Plots wind speed on x-axis and wind power on y axis.

    Args:
        ax (mpl.axes._subplots.AxesSubplot): Axis on which to plot.
        x1, y1: The x, y original data to be plotted. Both can be None if not available.
        x2, y2: The x, y data model to be plotted. Both can be None if not available.
    """
    # Plot the original data
    ax.scatter(
        x1, y1, color="blue", edgecolors="white", s=15, label="actual"
    )
    # Plot the model
    ax.scatter(
        x2, y2,
        color="orange", edgecolors="black", s=15, marker="D", label="model"
    )
    ax.set_xlabel("Wind Speed (m/s)", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Active Power (kW)", fontsize=FONT_SIZE_AXES)
    ax.set_title("Wind Speed vs. Power Output", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.legend(fontsize=FONT_SIZE_TICKS)


def plot_predicted_vs_real(
    ax: plt.Axes,
    x1: Iterable,
    y1: Iterable,
    x2: Iterable,
    y2: Iterable
):
    """Plots predicted vs. actual data.

    Args:
        ax (mpl.axes._subplots.AxesSubplot): Axis on which to plot.
        x1, y1: The x, y original data to be plotted. Both can be None if not available.
        x2, y2: The x, y data to plot a line. Both can be None if not available.
    """
    # Plot predicted vs real y
    ax.scatter(
        x1, y1, color="orange", edgecolors="black", label="Predicted vs. actual values", marker="D"
    )
    # Plot straight line
    ax.plot(
        x2, y2, color="blue", linestyle="--", linewidth=4, label="actual = predicted",
    )
    ax.set_xlabel("Actual Power Values (kW)", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Predicted Power Values (kW)", fontsize=FONT_SIZE_AXES)
    ax.set_title("Predicted vs. Actual Power Values (kW)", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    ax.legend(fontsize=FONT_SIZE_TICKS)

    
def fit_and_plot_linear_model(data_og: pd.core.frame.DataFrame, turbine: int, features: List[str]):
    # Get the data for the selected turbine
    data = data_og[data_og.TurbID == turbine]

    # Create the linear regression model
    features = list(features)
    y = data["Patv"]
    X = data[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LinearRegression().fit(X_train, y_train)

    # Prepare the data for plotting
    X_plot = data["Wspd"]
    Y_real = data["Patv"]
    y_test_preds = reg.predict(X_test)

    X_eq_Y = np.linspace(0, max([max(y_test), max(y_test_preds)]), 100)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Plotting on the left side plot
    if "Wspd" in features:
        plot_wind_speed_vs_power(ax1, X_plot, Y_real, X_test["Wspd"], y_test_preds)
    else:
        plot_wind_speed_vs_power(ax1, X_plot, Y_real, None, None)
        print("The model could not be plotted here as Wspd is not among the features")
    # Plotting on the right side plot
    plot_predicted_vs_real(ax2, y_test, y_test_preds, X_eq_Y, X_eq_Y)
    
    plt.tight_layout()
    plt.show()

    # Create a plot of feature imporance if there is more than one feature
    if len(features) > 1:
        # Create data for feature importance
        bunch = permutation_importance(
            reg, X_test, y_test, n_repeats=10, random_state=42
        )
        imp_means = bunch.importances_mean
        ordered_imp_means_args = np.argsort(imp_means)[::-1]

        results = {}
        for i in ordered_imp_means_args:
            name = list(X_test.columns)[i]
            imp_score = imp_means[i]
            results.update({name: [imp_score]})

        results_df = pd.DataFrame.from_dict(results)

        # Create a plot for feature importance
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
        ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)
        ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICKS)

        sns.barplot(data=results_df, orient="h", ax=ax, color="deepskyblue", width=0.3)

        plt.show()

    # Print out the mean absolute error
    mae = metrics.mean_absolute_error(y_test, y_test_preds)
    print(f"Turbine {turbine}, Mean Absolute Error (kW): {mae:.2f}\n")

    
def linear_univariate_model(data_og: pd.core.frame.DataFrame):
    """Creates an interactive plot of the univariate linear model for predicting energy output using wind speed as unique predictor.

    Args:
        data_og (pd.core.frame.DataFrame): The dataset used.
    """

    turbine_selection = widgets.Dropdown(
        options=data_og.TurbID.unique(), description="Turbine"
    )

    interact(fit_and_plot_linear_model, data_og=fixed(data_og), turbine=turbine_selection, features=fixed(["Wspd"]))  

    
def linear_multivariate_model(data_og: pd.core.frame.DataFrame, features: List[str]):
    """Creates an interactive plot to showcase multivariate linear regression.

    Args:
        data_og (pd.core.frame.DataFrame): The data used.
        features (List[str]): List of features to include in the prediction.
    """

    turbine_selection = widgets.Dropdown(
        options=data_og.TurbID.unique(), description="Turbine"
    )

    feature_selection = widgets.SelectMultiple(
        options=features,
        value=list(features),
        description="Features",
        disabled=False,
    )

    interact_manual(fit_and_plot_linear_model, data_og=fixed(data_og), turbine=turbine_selection, features=feature_selection)    


def split_and_normalize(data: pd.core.frame.DataFrame, features: List[str]):
    """Generates the train, test splits and normalizes the data.

    Args:
        data (pd.core.frame.DataFrame): The dataset used.
        features (List[str]): Features to include in the prediction process.

    Returns:
        tuple: The normalized train/test splits along with the train mean and standard deviation.
    """

    X = data[features]
    y = data["Patv"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    to_normalize = ["Wspd", "Etmp", "Itmp", "Prtv"]

    f_to_normalize = [feature for feature in features if feature in to_normalize]

    f_not_to_normalize = [
        feature for feature in features if feature not in to_normalize
    ]

    X_train_mean = X_train[f_to_normalize].mean()
    X_train_std = X_train[f_to_normalize].std()

    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    X_train[f_to_normalize] = (X_train[f_to_normalize] - X_train_mean) / X_train_std
    X_test[f_to_normalize] = (X_test[f_to_normalize] - X_train_mean) / X_train_std

    y_train = (y_train - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    X_train = torch.from_numpy(X_train).type(torch.float)
    X_test = torch.from_numpy(X_test).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float).unsqueeze(dim=1)
    y_test = torch.from_numpy(y_test).type(torch.float).unsqueeze(dim=1)

    return (X_train, X_test, y_train, y_test), (
        X_train_mean,
        X_train_std,
        y_train_mean,
        y_train_std,
    )


def batch_data(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
):
    """Creates batches from the original data.

    Args:
        X_train (torch.Tensor): Train predictors.
        X_test (torch.Tensor): Test predictors.
        y_train (torch.Tensor): Train target.
        y_test (torch.Tensor): Test target.
        batch_size (int): Desired batch size.

    Returns:
        tuple: Train and test DataLoaders.
    """
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


class RegressorNet(nn.Module):
    """A vanilla feed forward Neural Network with 3 hidden layers."""

    def __init__(self, input_size):
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x


def compile_model(features: List[str]):
    """Compiles the Pytorch network with an appropriate Loss and Optimizer.

    Args:
        features (List[str]): List of predictors to use.

    Returns:
        tuple: The model, loss function and optimizer used.
    """
    model = RegressorNet(input_size=len(features))
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    return model, loss_fn, optimizer


def train_model(
    model: RegressorNet,
    loss_fn: torch.nn.modules.loss.L1Loss,
    optimizer: torch.optim.Adam,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int,
):
    """Trains the neural network.

    Args:
        model (RegressorNet): An instance of the neural network.
        loss_fn (torch.nn.modules.loss.L1Loss): L1 loss (aka as Mean Absolute Error)
        optimizer (torch.optim.Adam): Adam Optimizer
        train_loader (torch.utils.data.DataLoader): The train data
        test_loader (torch.utils.data.DataLoader): The test data
        epochs (int): Desired number of epochs to train

    Returns:
        RegressorNet: The trained model.
    """

    for epoch in range(epochs):

        model.train()

        for batch, (X, y) in enumerate(train_loader):
            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)

            # 3. Zero grad optimizer
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Step the optimizer
            optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():

            for batch, (X, y) in enumerate(test_loader):
                # 1. Forward pass
                test_pred = model(X)

                # 2. Calculate the loss
                test_loss = loss_fn(test_pred, y)

        if epoch % 1 == 0:
            print(
                f"Epoch: {epoch} | Train loss: {loss:.5f} | Test loss: {test_loss:.5f}"
            )

    return model


def plot_feature_importance(
    model: RegressorNet,
    features: List[str],
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
):
    """Creates a feature importance plot by using SHAP values.

    Args:
        model (RegressorNet): The trained model.
        features (List[str]): List of predictors used.
        train_loader (torch.utils.data.DataLoader): Training data.
        test_loader (torch.utils.data.DataLoader): Testing data.
    """

    x_train_batch, _ = next(iter(train_loader))
    x_test_batch, _ = next(iter(test_loader))

    model.eval()
    e = shap.DeepExplainer(model, x_train_batch)
    shap_values = e.shap_values(x_test_batch)
    
    means = np.mean(np.abs(shap_values), axis=0)
    results = sorted(zip(features, means), key = lambda x: x[1], reverse=True)
    results_df = pd.DataFrame.from_dict({k: [v] for (k, v) in results})

    # Create a plot for feature importance
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.set_xlabel("Importance Score", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel("Feature", fontsize=FONT_SIZE_AXES)
    ax.set_title("Feature Importance", fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    sns.barplot(data=results_df, orient="h", ax=ax, color="deepskyblue", width=0.3)
    
    return shap_values


def neural_network(data_og: pd.core.frame.DataFrame, features: List[str]):
    """Creates an interactive plot of the prediction process when using a neural network.

    Args:
        data_og (pd.core.frame.DataFrame): The data used.
        features (List[str]): The features to include in the prediction process.
    """

    def fit_nn(turbine, features):
        data = data_og[data_og.TurbID == turbine]
        features = list(features)
        print(f"Features used: {features}\n")
        print(f"Training your Neural Network...\n")

        (X_train, X_test, y_train, y_test), (
            X_train_mean,
            X_train_std,
            y_train_mean,
            y_train_std,
        ) = split_and_normalize(data, features)
        train_loader, test_loader = batch_data(
            X_train, X_test, y_train, y_test, batch_size=32
        )
        model, loss_fn, optimizer = compile_model(features)
        model = train_model(
            model, loss_fn, optimizer, train_loader, test_loader, epochs=5
        )
        print(f"\nResults:")

        y_test_denormalized = (y_test * y_train_std) + y_train_mean
        test_preds = model(X_test).detach().numpy()
        test_preds_denormalized = (test_preds * y_train_std) + y_train_mean
        X_plot = data["Wspd"]
        Y_real = data["Patv"]
        X_eq_Y = np.linspace(0, max(y_test_denormalized), 100)
        
        print(
            f"Mean Absolute Error: {metrics.mean_absolute_error(y_test_denormalized, test_preds_denormalized):.2f}\n"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if "Wspd" in features:
            test_preds = model(X_test).detach().numpy()
            test_preds_denormalized = (test_preds * y_train_std) + y_train_mean

            X_test_2 = X_test.detach().numpy()
            X_test_denormalized = (X_test_2[:, 0] * X_train_std[0]) + X_train_mean[0]
            
            plot_wind_speed_vs_power(ax1, X_plot, Y_real, X_test_denormalized, test_preds_denormalized)
        else:
            plot_wind_speed_vs_power(ax1, X_plot, Y_real, None, None)
            print("The model could not be plotted here as Wspd is not among the features")

        plot_predicted_vs_real(ax2, y_test_denormalized, test_preds_denormalized, X_eq_Y, X_eq_Y)

        plt.show()          

        train_loader, test_loader = batch_data(
            X_train, X_test, y_train, y_test, batch_size=128
        )

        plot_feature_importance(model, features, train_loader, test_loader)

    turbine_selection = widgets.Dropdown(
        options=data_og.TurbID.unique(), description="Turbine"
    )
    feature_selection = widgets.SelectMultiple(
        options=features,
        value=list(features),
        description="Features",
        disabled=False,
    )
    interact_manual(fit_nn, turbine=turbine_selection, features=feature_selection)

def prepare_data(df: pd.core.frame.DataFrame, turb_id: int) -> pd.core.frame.DataFrame:
    """Pre-process data before feeding to neural networks for training.
    This includes:
    - Resampling to an hourly basis
    - Using data from a single turbine
    - Format datetime
    - Mask abnormal values
    - Re-order columns

    Args:
        df (pandas.core.frame.DataFrame): The curated data from the previous lab.
        turb_id (int): ID of the turbine to use.

    Returns:
        pandas.core.frame.DataFrame: Processed dataframe.
    """
    df = df[5::6]
    df = df[df.TurbID == turb_id]
    df = df.drop(["TurbID"], axis=1)
    df.index = pd.to_datetime(df.pop("Datetime"), format="%Y-%m-%d %H:%M")
    df = df.mask(df.Include == False, -1)
    df = df.drop(["Include"], axis=1)

    df = df[
        [
            "Wspd",
            "Etmp",
            "Itmp",
            "Prtv",
            "WdirCos",
            "WdirSin",
            "NdirCos",
            "NdirSin",
            "PabCos",
            "PabSin",
            "Patv",
        ]
    ]

    return df


def normalize_data(
    train_data: pd.core.frame.DataFrame,
    val_data: pd.core.frame.DataFrame,
    test_data: pd.core.frame.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.core.series.Series, pd.core.series.Series
]:
    """Normalizes train, val and test splits.

    Args:
        train_data (pd.core.frame.DataFrame): Train split.
        val_data (pd.core.frame.DataFrame): Validation split.
        test_data (pd.core.frame.DataFrame): Test split.

    Returns:
        tuple: Normalized splits with training mean and standard deviation.
    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, val_data, test_data, train_mean, train_std


@dataclass
class DataSplits:
    """Class to encapsulate normalized/unnormalized train, val, test, splits."""

    train_data: pd.core.frame.DataFrame
    val_data: pd.core.frame.DataFrame
    test_data: pd.core.frame.DataFrame
    train_mean: pd.core.series.Series
    train_std: pd.core.series.Series
    train_df_unnormalized: pd.core.frame.DataFrame
    val_df_unnormalized: pd.core.frame.DataFrame
    test_df_unnormalized: pd.core.frame.DataFrame


def train_val_test_split(df: pd.core.frame.DataFrame) -> DataSplits:
    """Splits a dataframe into train, val and test.

    Args:
        df (pd.core.frame.DataFrame): The data to split.

    Returns:
        data_splits (DataSplits): An instance that encapsulates normalized/unnormalized splits.
    """
    n = len(df)
    train_df = df[0 : int(n * 0.7)]
    val_df = df[int(n * 0.7) : int(n * 0.9)]
    test_df = df[int(n * 0.9) :]

    train_df_un = train_df.copy(deep=True)
    val_df_un = val_df.copy(deep=True)
    test_df_un = test_df.copy(deep=True)

    train_df_un = train_df_un.mask(train_df_un.Patv == -1, np.nan)
    val_df_un = val_df_un.mask(val_df_un.Patv == -1, np.nan)
    test_df_un = test_df_un.mask(test_df_un.Patv == -1, np.nan)

    train_df, val_df, test_df, train_mn, train_st = normalize_data(
        train_df, val_df, test_df
    )

    ds = DataSplits(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        train_mean=train_mn,
        train_std=train_st,
        train_df_unnormalized=train_df_un,
        val_df_unnormalized=val_df_un,
        test_df_unnormalized=test_df_un,
    )

    return ds


def plot_time_series(data_splits: DataSplits) -> None:
    """Plots time series of active power vs the other features.

    Args:
        data_splits (DataSplits): Turbine data.
    """
    train_df, val_df, test_df = (
        data_splits.train_df_unnormalized,
        data_splits.val_df_unnormalized,
        data_splits.test_df_unnormalized,
    )

    def plot_time_series(feature):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
        ax1.plot(train_df["Patv"], color="blue", label="training")
        ax1.plot(val_df["Patv"], color="green", label="validation")
        ax1.plot(test_df["Patv"], color="red", label="test")
        ax1.set_title("Time series of Patv (target)", fontsize=FONT_SIZE_TITLE)
        ax1.set_ylabel("Active Power (kW)", fontsize=FONT_SIZE_AXES)
        ax1.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax1.legend(fontsize=15)
        ax1.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        ax2.plot(train_df[feature], color="blue", label="training")
        ax2.plot(val_df[feature], color="green", label="validation")
        ax2.plot(test_df[feature], color="red", label="test")
        ax2.set_title(f"Time series of {feature} (predictor)", fontsize=FONT_SIZE_TITLE)
        ax2.set_ylabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        ax2.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax2.legend(fontsize=15)
        ax2.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)

        plt.tight_layout()
        plt.show()

    feature_selection = widgets.Dropdown(
        options=[f for f in list(train_df.columns) if f != "Patv"],
        description="Feature",
    )

    interact(plot_time_series, feature=feature_selection)


def compute_metrics(
    true_series: np.ndarray, forecast: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes MSE and MAE between two time series.

    Args:
        true_series (np.ndarray): True values.
        forecast (np.ndarray): Forecasts.

    Returns:
        tuple: MSE and MAE metrics.
    """

    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae


class WindowGenerator:
    """Class that handles all of the windowing and plotting logic for time series."""

    def __init__(
        self,
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=["Patv"],
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot(self, model=None, plot_col="Patv", max_subplots=1):
        inputs, labels = self.example
        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.title("Inputs (past) vs Labels (future predictions)", fontsize=FONT_SIZE_TITLE)
            plt.ylabel(f"{plot_col} (normalized)", fontsize=FONT_SIZE_AXES)
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                color="green",
                linestyle="--",
                label="Inputs",
                marker="o",
                markersize=10,
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.plot(
                self.label_indices,
                labels[n, :, label_col_index],
                color="orange",
                linestyle="--",
                label="Labels",
                markersize=10,
                marker="o"
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="*",
                    edgecolors="k",
                    label="Predictions",
                    c="pink",
                    s=64,
                )
            plt.legend(fontsize=FONT_SIZE_TICKS)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.xlabel("Timestep", fontsize=FONT_SIZE_AXES)

    def plot_long(
        self,
        model,
        data_splits,
        plot_col="Patv",
        time_steps_future=1,
        baseline_mae=None,
    ):
        train_mean, train_std = data_splits.train_mean, data_splits.train_std
        self.test_size = len(self.test_df)
        self.test_data = self.make_test_dataset(self.test_df, self.test_size)

        inputs, labels = next(iter(self.test_data))

        plt.figure(figsize=(20, 6))
        plot_col_index = self.column_indices[plot_col]

        plt.ylabel(f"{plot_col} (kW)", fontsize=FONT_SIZE_AXES)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        labels = (labels * train_std.Patv) + train_mean.Patv

        upper = 24 - (time_steps_future - 1)
        lower = self.label_indices[-1] - upper
        self.label_indices_long = self.test_df.index[lower:-upper]

        plt.plot(
            self.label_indices_long[:],
            labels[:, time_steps_future - 1, label_col_index][:],
            label="Labels",
            c="green",
        )

        if model is not None:
            predictions = model(inputs)
            predictions = (predictions * train_std.Patv) + train_mean.Patv
            predictions_for_timestep = predictions[
                :, time_steps_future - 1, label_col_index
            ][:]
            predictions_for_timestep = tf.nn.relu(predictions_for_timestep).numpy()
            plt.plot(
                self.label_indices_long[:],
                predictions_for_timestep,
                label="Predictions",
                c="orange",
                linewidth=3,
            )
            plt.legend(fontsize=FONT_SIZE_TICKS)
            _, mae = compute_metrics(
                labels[:, time_steps_future - 1, label_col_index][:],
                predictions_for_timestep,
            )

            if baseline_mae is None:
                baseline_mae = mae

            print(
                f"\nMean Absolute Error (kW): {mae:.2f} for forecast.\n\nImprovement over random baseline: {100*((baseline_mae -mae)/baseline_mae):.2f}%"
            )
        plt.title("Predictions vs Real Values for Test Split", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Date", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        return mae

    def make_test_dataset(self, data, bs):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=bs,
        )

        ds = ds.map(self.split_window)

        return ds

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, "_example", None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


def generate_window(
    train_df: pd.core.frame.DataFrame,
    val_df: pd.core.frame.DataFrame,
    test_df: pd.core.frame.DataFrame,
    days_in_past: int,
    width: int = 24
) -> WindowGenerator:
    """Creates a windowed dataset given the train, val, test splits and the number of days into the past.

    Args:
        train_df (pd.core.frame.DataFrame): Train split.
        val_df (pd.core.frame.DataFrame): Val Split.
        test_df (pd.core.frame.DataFrame): Test split.
        days_in_past (int): How many days into the past will be used to predict the next 24 hours.

    Returns:
        WindowGenerator: The windowed dataset.
    """
    OUT_STEPS = 24
    multi_window = WindowGenerator(
        input_width=width * days_in_past,
        label_width=OUT_STEPS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        shift=OUT_STEPS,
    )
    return multi_window


def create_model(num_features: int, days_in_past: int) -> tf.keras.Model:
    """Creates a Conv-LSTM model for time series prediction.

    Args:
        num_features (int): Number of features used for prediction.
        days_in_past (int): Number of days into the past to predict next 24 hours.

    Returns:
        tf.keras.Model: The uncompiled model.
    """
    CONV_WIDTH = 3
    OUT_STEPS = 24
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Masking(
                mask_value=-1.0, input_shape=(days_in_past * 24, num_features)
            ),
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(CONV_WIDTH)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=True)
            ),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(32, return_sequences=False)
            ),
            tf.keras.layers.Dense(
                OUT_STEPS * 1, kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Reshape([OUT_STEPS, 1]),
        ]
    )

    return model


def compile_and_fit(
    model: tf.keras.Model, window: WindowGenerator, patience: int = 2
) -> tf.keras.callbacks.History:
    """Compiles and trains a model given a patience threshold.

    Args:
        model (tf.keras.Model): The model to train.
        window (WindowGenerator): The windowed data.
        patience (int, optional): Patience threshold to stop training. Defaults to 2.

    Returns:
        tf.keras.callbacks.History: The training history.
    """
    EPOCHS = 20
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
    )

    tf.random.set_seed(432)
    np.random.seed(432)
    random.seed(432)

    history = model.fit(
        window.train, epochs=EPOCHS, validation_data=window.val, callbacks=[early_stopping]
    )
    
    if len(history.epoch) < EPOCHS:
        print("\nTraining stopped early to prevent overfitting, as the validation loss is increasing for two consecutive steps.")
    
    return history


def train_conv_lstm_model(
    data: pd.core.frame.DataFrame, features: List[str], days_in_past: int
) -> Tuple[WindowGenerator, tf.keras.Model, DataSplits]:
    """Trains the Conv-LSTM model for time series prediction.

    Args:
        data (pd.core.frame.DataFrame): The dataframe to be used.
        data (list[str]): The features to use for forecasting.
        days_in_past (int): How many days in the past to use to forecast the next 24 hours.

    Returns:
        tuple: The windowed dataset, the model that handles the forecasting logic and the data used.
    """
    data_splits = train_val_test_split(data[features])

    train_data, val_data, test_data, train_mean, train_std = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
        data_splits.train_mean,
        data_splits.train_std,
    )

    window = generate_window(train_data, val_data, test_data, days_in_past)
    num_features = window.train_df.shape[1]

    model = create_model(num_features, days_in_past)
    history = compile_and_fit(model, window)
    
    return window, model, data_splits


def prediction_plot(
    func: Callable, model: tf.keras.Model, data_splits: DataSplits, baseline_mae: float
) -> None:
    """Plot an interactive visualization of predictions vs true values.

    Args:
        func (Callable): Function to close over. Should be the plot_long method from the WindowGenerator instance.
        model (tf.keras.Model): The trained model.
        data_splits (DataSplits): The data used.
        baseline_mae (float): MAE of baseline to compare against.
    """

    def _plot(time_steps_future):
        mae = func(
            model,
            data_splits,
            time_steps_future=time_steps_future,
            baseline_mae=baseline_mae,
        )

    time_steps_future_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_future=time_steps_future_selection)
    
    
def random_forecast(
    data_splits: DataSplits, n_days: int = 1
) -> Tuple[WindowGenerator, tf.keras.Model]:
    """Generates a random forecast for a time window.

    Args:
        data_splits (DataSplits): The data to be used.
        n_days (int, optional): Period from which to draw the random values. Defaults to 1.

    Returns:
        tuple: The windowed dataset and the model that handles the forecasting logic.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )

    random_window = generate_window(train_data, val_data, test_data, n_days)

    class randomBaseline(tf.keras.Model):
        def call(self, inputs):
            tf.random.set_seed(424)
            np.random.seed(424)
            random.seed(424)
            stacked = tf.random.shuffle(inputs)

            return stacked[:, :, -1:]

    random_baseline = randomBaseline()
    random_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return random_window, random_baseline


def repeat_forecast(
    data_splits: DataSplits, shift: int=24
) -> Tuple[WindowGenerator, tf.keras.Model]:
    """Performs a repeated forecast logic.

    Args:
        data_splits (DataSplits): The data to be used.
        n_days (int): Period to repeat.

    Returns:
        tuple: The windowed dataset and the model that handles the forecasting logic.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )
    repeat_window = generate_window(train_data, val_data, test_data, 1, shift)

    class RepeatBaseline(tf.keras.Model):
        def call(self, inputs):
            return inputs[:, :, -1:]

    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return repeat_window, repeat_baseline


def interact_repeat_forecast(
    data_splits: DataSplits, baseline_mae: float
) -> None:
    """Plot an interactive visualization of predictions vs true values.

    Args:
        func (Callable): Function to close over. Should be the plot_long method from the WindowGenerator instance.
        model (tf.keras.Model): The trained model.
        data_splits (DataSplits): The data used.
        baseline_mae (float): MAE of baseline to compare against.
    """

    def _plot(shift):
        repeat_window, repeat_baseline = repeat_forecast(data_splits, shift=shift)
        _ = repeat_window.plot_long(repeat_baseline, data_splits, baseline_mae=baseline_mae)

    shift_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, shift=shift_selection)

def moving_avg_forecast(data_splits: DataSplits, n_days: int) -> Tuple[WindowGenerator, tf.keras.Model]:
    """Performs a moving average forecast logic.

    Args:
        data_splits (DataSplits): The data to be used.
        n_days (int): Period to repeat.

    Returns:
        tuple: The windowed dataset and the model that handles the forecasting logic.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )
    moving_avg_window = generate_window(train_data, val_data, test_data, n_days)

    class avgBaseline(tf.keras.Model):
        def call(self, inputs):
            m = tf.math.reduce_mean(inputs, axis=1)
            stacked = tf.stack([m for _ in range(inputs.shape[1])], axis=1)

            return stacked[:, :, -1:]

    moving_avg_baseline = avgBaseline()
    moving_avg_baseline.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    return moving_avg_window, moving_avg_baseline


def add_wind_speed_forecasts(
    df: pd.core.frame.DataFrame, add_noise=False
) -> pd.core.frame.DataFrame:
    """Creates syntethic wind speed forecasts. The more into the future, the more noise these have.

    Args:
        df (pd.core.frame.DataFrame): Dataframe with data from turbine.
        periods (list, optional): Periods for which to create the forecast. Defaults to [*range(1, 30, 1)].

    Returns:
        pd.core.frame.DataFrame: The new dataframe with the synth forecasts.
    """

    df_2 = df.copy(deep=True)
    # Periods for which to create the forecast.
    periods=[*range(1, 30, 1)]
    
    for period in periods:
        
        if add_noise == "linearly_increasing":
            np.random.seed(8752)
            noise_level = 0.2 * period
            noise = np.random.randn(len(df)) * noise_level
        
        elif add_noise == "mimic_real_forecast":
            np.random.seed(8752)
            noise_level = 2 + 0.05 * period
            noise = np.random.randn(len(df)) * noise_level
        else:
            noise = 0
        
        padding_slice = df_2["Wspd"][-period:].to_numpy()
        values = np.concatenate((df_2["Wspd"][period:].values, padding_slice)) + noise
        
        df_2[f"fc-{period}h"] = values

    return df_2


def plot_forecast_with_noise(
    data_with_wspd_forecasts: pd.core.frame.DataFrame,
) -> None:
    """Creates an interactive plot that shows how the synthetic forecasts change when the future horizon is changed.

    Args:
        data_with_wspd_forecasts (pd.core.frame.DataFrame): Dataframe that includes synth forecasts.
    """

    def _plot(noise_level):
        fig, ax = plt.subplots(figsize=(20, 6))

        df = data_with_wspd_forecasts
        synth_data = df[f"fc-{noise_level}h"][
            5241 - noise_level : -noise_level
        ].values
        synth_data = tf.nn.relu(synth_data).numpy()
        real_data = df["Wspd"][5241:].values
        real_data = tf.nn.relu(real_data).numpy()

        mae = metrics.mean_absolute_error(real_data, synth_data)

        print(f"\nMean Absolute Error (m/s): {mae:.2f} for forecast\n")
        ax.plot(df.index[5241:], real_data, label="true values")
        ax.plot(
            df.index[5241:],
            synth_data,
            label="syntethic predictions",
        )

        ax.set_title("Generated wind speed forecasts", fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=FONT_SIZE_AXES)
        ax.set_xlabel("Date", fontsize=FONT_SIZE_AXES)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        ax.legend()

    noise_level_selection = widgets.IntSlider(
        value=1,
        min=1,
        max=25,
        step=1,
        description="Noise level in m/s (low to high)",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=False,
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, noise_level=noise_level_selection)


def window_plot(data_splits: DataSplits) -> None:
    """Creates an interactive plots to show how the data is windowed depending on the number of days into the past that are used to forecast the next 24 hours.

    Args:
        data_splits (DataSplits): Data used.
    """
    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data,
    )

    def _plot(time_steps_past):
        window = generate_window(train_data, val_data, test_data, time_steps_past)
        window.plot()

    time_steps_past_selection = widgets.IntSlider(
        value=1,
        min=1,
        max=14,
        step=1,
        description="Days before",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_past=time_steps_past_selection)

    
def load_weather_forecast() -> Dict[str, Dict[List[datetime], List[float]]]:
    """Loads the wind data and forecast for three locations and returns it in a form of dictionary.
    """
    with open("data/weather_forecast.pkl", "rb") as f:
        weather_forecasts = pickle.load(f)
    return weather_forecasts


def plot_forecast(weather_forecasts: Dict[str, Dict[List[datetime], List[float]]]) -> None:
    """Creates an interactive plot of true values vs forecasts for the wind data.

    Args:
        weather_forecasts (dict): History of weather and weather forecasts.
    """
    def _plot(city, time_steps_future):
        format_timestamp = "%Y-%m-%d %H:%M:%S"

        weather_forecast = weather_forecasts[city]
        
        dates_real, winds_real = weather_forecast[0]
        dates_real = [datetime.strptime(i, format_timestamp) for i in dates_real]
        dates_forecast, winds_forecast = weather_forecast[time_steps_future]
        dates_forecast = [datetime.strptime(i, format_timestamp) for i in dates_forecast]

        # Set the min and max date for plotting, so it always plots the same
        min_date = datetime.strptime("2022-11-16 18:00:00", format_timestamp)
        max_date = datetime.strptime("2023-01-11 15:00:00", format_timestamp)
        
        # Find the overlap of the data and limit it to the plotting range
        dates_real, dates_forecast, winds_real, winds_forecast = prepare_wind_data(
            dates_real, dates_forecast, winds_real, winds_forecast, min_date, max_date
        )
        
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(dates_real, winds_real, label="Actual windspeed")
        ax.plot(dates_forecast, winds_forecast, label=f"Forecasted windspeed {time_steps_future} Hours in the Future")
        ax.set_title(f"History of Actual vs Forecasted Windspeed in {city}", fontsize=25)
        ax.set_ylabel("Wind Speed (m/s)", fontsize=20)
        ax.set_xlabel("Date", fontsize=20)
        ax.tick_params(axis="both", labelsize=15)
        ax.legend(fontsize=15)
        
        mae = metrics.mean_absolute_error(winds_real, winds_forecast)
        print(f"\nMean Absolute Error (m/s): {mae:.2f} for forecast\n")
       
    city_selection = widgets.Dropdown(
        options=weather_forecasts.keys(),
        description='City',
    )
    time_steps_future_selection = widgets.IntSlider(
        value=1,
        min=3,
        max=120,
        step=3,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, city=city_selection, time_steps_future=time_steps_future_selection)

    
def prepare_wind_data(
    dates0: List[datetime],
    dates1: List[datetime],
    winds0: List[float],
    winds1: List[float],
    min_bound: Optional[str]=None,
    max_bound: Optional[str]=None
) -> Tuple[List[datetime], List[datetime], List[float], List[float]]:
    """Takes in two datasets of wind data.
    Finds the data points that appear in both datasets (at the same time) and are between the specified time bounds.

    Args:
        dates0 (list): list of dates for the first dataset
        dates1 (list): list of dates for the second dataset
        winds0 (list): list of wind speed for the first dataset (corresponding to dates0)
        winds1 (list): list of wind speed for the second dataset (corresponding to dates1)
        min_bound (datetime): minimum bound for plotting
        max_bound (datetime): maximum bound for plotting
    """
    winds0_overlap = []
    winds1_overlap = []
    dates0_overlap = []
    dates1_overlap = []
    
    # Only take the dates that are in both datasets and within the limits if specified
    for date, wind in zip(dates0, winds0):
        if (date in dates1 and 
            (min_bound is None or date > min_bound) and
            (max_bound is None or date < max_bound)
           ):
            winds0_overlap.append(wind)
            dates0_overlap.append(date)
    for date, wind in zip(dates1, winds1):
        if (date in dates0 and 
            (min_bound is None or date > min_bound) and
            (max_bound is None or date < max_bound)
           ):
            winds1_overlap.append(wind)
            dates1_overlap.append(date)
    
    return dates0_overlap, dates1_overlap, winds0_overlap, winds1_overlap


def plot_mae_forecast(weather_forecasts: Dict[str, Dict[List[datetime], List[float]]]) -> None:
    """Creates an interactive plot MAE of wind forecasts.

    Args:
        weather_forecasts (dict): Weather and weather forecast data.
    """
    def _plot(city):
        weather_forecast = weather_forecasts[city]
        
        times = sorted(weather_forecast.keys())[1::]
        maes = []
        
        dates_real, winds_real = weather_forecast[0]
        for time in times:
            dates_forecast, winds_forecast = weather_forecast[time]
            dates_real, dates_forecast, winds_real, winds_forecast = prepare_wind_data(
                dates_real, dates_forecast, winds_real, winds_forecast
            )
            mae = metrics.mean_absolute_error(winds_real, winds_forecast)
            maes.append(mae)
            
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(times, maes, marker="*")
        ax.set_title("Mean Absolute Error of Actual vs Predicted Wind Speed", fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Mean Absolute Error (m/s)", fontsize=FONT_SIZE_AXES)
        ax.set_xlabel("Hours into the future", fontsize=FONT_SIZE_AXES)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
               
    city_selection = widgets.Dropdown(
        options=weather_forecasts.keys(),
        description='City',
    )
    
    interact(_plot, city=city_selection)
    
def get_metadata(data_folder: str) -> pd.core.frame.DataFrame:
    '''Inspect the data_folder and create a data frame using the metainformation of the images.
    The folder name are the classes and place code must be extracted out of the file name.
    
    Args:
        data_folder (str): The location of the data.

    Returns:
        pd.core.frame.DataFrame: The dataframe with metadata.
    '''
    # Find all file names within the data folder.
    all_paths = [y for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.JPG'))]

    meta_data_list = []
    for file_path in all_paths:
        # Split the path into subfolders and file name.
        data_folder_name, class_folder_name, file_name = file_path.split("/")
        # Camera locaion is given within the file name.
        camera_location = file_name.split('_')[2]
        meta_data_list.append([camera_location, class_folder_name, file_path])

    # Create a dataframe with metadata.
    meta_data = pd.DataFrame(data=meta_data_list, columns=['location', 'class', 'path'])
    
    return meta_data


def plot_donut_chart(class_counts: pd.core.frame.DataFrame):
    '''Plot a donut chart of the class distribution in the dataset.
    
    Args:
        class_counts (pd.core.frame.DataFrame): The dataframe with info about classes.
    '''
        
    fig = px.pie(
        pd.DataFrame({'class': class_counts.index, 'values': class_counts.values}),
        values='values',
        names='class',
        title='Distribution of Animals', 
        hole = 0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.show()


def plot_bar_chart(meta_data: pd.core.frame.DataFrame):
    '''Plot a bar chart of the class distribution in the dataset.
    
    Args:
        meta_data (pd.core.frame.DataFrame): The dataframe with data about all data.
    '''
    cmap = colors.ListedColormap(cm.tab20c.colors + cm.tab20b.colors, name='tab40')
    
    cross = pd.crosstab(index=meta_data['location'], 
                        columns=meta_data["class"],
                        normalize='index')
    cross.plot(
        kind="bar", 
        stacked=True,
        figsize=(11, 6),
        fontsize=12,
        cmap=cmap
    )
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title("Relative class distribution by location", fontsize=20)
    plt.xlabel("Location", fontsize=16)
    plt.ylabel("Relative class distribution", fontsize=16)

    plt.show()
    


def plot_random_images(meta_data: pd.core.frame.DataFrame, location: str):
    '''Plots a 3x3 grid of random images.
    
    Args:
        meta_data (pd.core.frame.DataFrame): The dataframe with data about all data.
        location (str): The location of the camera trap from which the images are taken.
    '''
    plt.figure(figsize=(15, 15))
    for sample_number in range(9):
        sample_row = meta_data[meta_data['location']==location].sample()
        path = sample_row.path.values[0]
        ax = plt.subplot(3, 3, sample_number + 1)
        plt.imshow(mpimg.imread(path))
        plt.title(sample_row['class'].values[0], fontsize=14)
        plt.axis("off")

        
def plot_images_from_all_locations(meta_data: pd.core.frame.DataFrame):
    '''Plots a 4x4 grid of images. Each image is a random image from a different location.
    
    Args:
        meta_data (pd.core.frame.DataFrame): The dataframe with data about all data.
    '''
    locations = sorted(meta_data['location'].unique())
    plt.figure(figsize=(15, 15))
    for loc_index, location in enumerate(locations):
        example = meta_data[meta_data['location']==location].sample()
        path = example.path.values[0]
        image = mpimg.imread(path)
        ax = plt.subplot(4, 4, loc_index + 1)
        plt.imshow(image)
        plt.title(f'loc_{location}_{example["class"].values[0]}', fontsize=14)
        plt.axis("off")
        

def plot_examples(sequence: List[str]):
    '''Plots a grid of specified images.
    
    Args:
        sequence (List[str]): A list of paths to images to be plotted.
    '''
    plt.figure(figsize=(15, 15))
    columns = 3
    rows = len(sequence) // columns + (len(sequence) % columns > 0)
    for index, example in enumerate(sequence):
        location=example.split("/")[2].split("_")[2]
        animal = example.split("/")[1]
        ax = plt.subplot(rows, columns, index + 1)
        plt.imshow(mpimg.imread(example))
        plt.title(f'loc_{location}_{animal}', fontsize=14)
        plt.axis("off")
