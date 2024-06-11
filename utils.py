import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mplcursors
import base64
import folium
import IPython
import ipywidgets as widgets
from IPython.display import clear_output, display
from datetime import datetime, timedelta
from ipywidgets import interact, interact_manual, fixed
from typing import List, Iterable
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.inspection import permutation_importance

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
