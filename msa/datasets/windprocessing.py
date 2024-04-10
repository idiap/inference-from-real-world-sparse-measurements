# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Wind processing utilities."""
from math import pi

import numpy as np
import pandas as pd
import torch
import yaml

input_label = [
    "LAT",
    "LONG",
    "TRACK_ALTITUDE",
    "UNIX_TIME",
]

WIND_SPEED = "WIND_SPEED"
WIND_ANGLE = "WIND_ORIGIN_ANGLE_DEG"
FLIGHT_ID = "FLIGHT_ID"
UNIX_TIME = "UNIX_TIME"
DAY = "DAY"
VX = "VX"
VY = "VY"
speed_label = [VX, VY]
label = input_label + speed_label


def select_day(df, day):
    """Select a day from a dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        day (int): Day to select.

    Returns:
        pd.DataFrame: Dataframe containing only the selected day.
    """
    df = df[df["DAY"] == day].copy()
    df.loc[:, "DAY"] = 0
    return df


def process_df(df):
    """Process a dataframe containing wind data.

    Processing:
        - Filter invalid flights.
        - Add weektime.
        - Convert polar coordinates to carthesian coordinates.
        - Sort by time.

    Args:
        df (pd.DataFrame): Dataframe containing the data.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    df = filter_invalid_flights(df)
    df[UNIX_TIME] = add_weektime(df[UNIX_TIME], df[DAY])
    df[VX], df[VY] = polar2carthesian(df[WIND_SPEED], df[WIND_ANGLE])
    df = df.sort_values(by="UNIX_TIME")
    # Mandatory for trajectory dataset
    df = df.reset_index(drop=True)

    fid2indices = create_fid2indices(df)

    return torch.tensor(df[label].values).float(), fid2indices


def create_fid2indices(df):
    """Create a dictionary mapping flight ids to indices of all the data point that it contains.

    Args:
        df (pd.DataFrame): Dataframe containing the data.

    Returns:
        dict: Dictionary mapping flight ids to indices.
    """
    return {
        str(fid): torch.tensor((df[df["FLIGHT_ID"] == fid]).index)
        for fid in df["FLIGHT_ID"].unique()
    }


def filter_invalid_flights(df):
    """Filter invalid flights.

    Invalid flights are flights with wind speed greater than 250 m/s.
    or flights with missing data.

    Args:
        df (pd.DataFrame): Dataframe containing the data.

    Returns:
        pd.DataFrame: Dataframe containing only valid flights.
    """
    SPEED_OUTLIER_LIMIT = 250
    invalid = df[df[WIND_SPEED] > SPEED_OUTLIER_LIMIT][FLIGHT_ID].unique()
    df = df.dropna(axis=0)
    df = df[~(df[FLIGHT_ID].isin(invalid))]
    return df


def add_weektime(time, day):
    """Correct bug in the dataset that resets the time to 0 at midnight.

    Args:
        time (np.array): UNIX time.
        day (int): Day.

    Returns:
        np.array: Time series with weektime.
    """
    duration_day = 60 * 60 * 24
    time = time + day * duration_day
    return time


def polar2carthesian(r, angle):
    """Convert polar coordinates to carthesian coordinates.

    Args:
        r (float): Radius.
        angle (float): Angle in degrees starting north clock wise.

    Returns:
        float: x coordinate.
        float: y coordinate.
    """
    x = r * np.cos((90 - angle) / 360 * 2 * pi)
    y = r * np.sin((90 - angle) / 360 * 2 * pi)

    return x, y


def carthesian2polar(x, y):
    """Convert carthesian coordinates to polar coordinates.

    Args:
        x (float): x coordinate.
        y (float): y coordinate.

    Returns:
        float: Radius.
        float: Angle in degrees starting north clock wise.
    """
    if isinstance(x, np.ndarray):
        r = np.sqrt(x**2 + y**2)
        angle = 90 - np.arctan2(y, x) * 360 / 2 / pi
        return r, angle

    r = torch.sqrt(x**2 + y**2)
    angle = 90 - torch.atan2(y, x) * 360 / 2 / pi
    return r, angle


def save_count(data_list, filename):
    """Save the number of data points per flight.

    Args:
        data_list (list): List of data points.
        filename (str): Name of the file to save the data.
    """
    name_and_count = [(str(i), len(d)) for i, d in enumerate(data_list)]
    pd.DataFrame(name_and_count, columns=["name", "count"]).to_csv(filename)


def save_means_stds(means, std, filename):
    """Save the means and standard deviations of the data.

    Args:
        means (torch.Tensor): Means of the data.
        std (torch.Tensor): Standard deviations of the data.
        filename (str): Name of the file to save the data.
    """
    columns = [la + "_mean" for la in label] + [la + "_std" for la in label]
    means_and_std = torch.cat([means, std]).squeeze().tolist()
    d = dict(zip(columns, means_and_std))
    with open(filename, "w+") as file:
        yaml.dump(d, file)


def load_means_stds(filename):
    """Load the means and standard deviations of the data.

    Args:
        filename (str): Name of the file to load the data.

    Returns:
        torch.Tensor: Means of the data.
        torch.Tensor: Standard deviations of the data.
    """
    with filename.open("r") as file:
        d = yaml.safe_load(file)

    means = torch.tensor([d[L + "_mean"] for L in label])
    stds = torch.tensor([d[L + "_std"] for L in label])

    return means, stds
