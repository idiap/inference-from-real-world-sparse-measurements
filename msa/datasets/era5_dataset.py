# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""ERA 5 dataset."""
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from dotenv import load_dotenv
from torch.utils import data

from msa.datasets.utils import StackDataset

from .utils import shapes

load_dotenv()


def grib2grid(path):
    """Read a .grib file and returns grid as a list of numpy arrays.

    files are available here : https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
    Choose the following parameters:
    - 10m u-component of wind
    - 10m v-component of wind
    - 100m u-component of wind
    - 100m v-component of wind
    - Surface pressure
    - 2m temperature
    - total cloud cover

    time of day (all)
    days (all)
    year: 2000 (train)
    year: 2010, month January, September (val)

    """
    print(f"Reading {path}")
    ds = xr.load_dataset(path, engine="cfgrib")

    return [ds[k].values for k in ds.data_vars.keys()]


def grid2dataset(grid):
    """Convert a xarray to a torch tensor.

    Logic:
        - Start with a numpy array (nt_tot, nlat, nlon, 7)
        - convert to tensors and squeeze
        - take pairs of images with 50 dt in between
        - sample 64, from context and 256 from target
    """
    print(len(grid), grid[0].shape)
    nt_tot, nlat, nlon = grid[0].shape
    nt = nt_tot - 50
    x = np.random.randint(0, nlat, (nt, 1024))
    y = np.random.randint(0, nlon, (nt, 1024))
    a = np.tile(np.arange(0, nt)[:, None], (1, 1024))

    xs = torch.from_numpy(x).div(nlat).mul(2).sub(1)
    ys = torch.from_numpy(y).div(nlon).mul(2).sub(1)
    xc = torch.cat((xs[..., None], ys[..., None]), dim=2).float()
    yc = torch.cat([torch.tensor(g[a, x, y]).unsqueeze(-1) for g in grid], dim=-1)

    x = np.random.randint(0, nlat, (nt, 1024))
    y = np.random.randint(0, nlon, (nt, 1024))
    a = np.tile(np.arange(50, nt_tot)[:, None], (1, 1024))

    xs = torch.from_numpy(x).div(nlat).mul(2).sub(1)
    ys = torch.from_numpy(y).div(nlon).mul(2).sub(1)
    xt = torch.cat((xs[..., None], ys[..., None]), dim=2).float()

    yt = torch.cat([torch.tensor(g[a, x, y]).unsqueeze(-1) for g in grid[2:4]], dim=-1)

    return (xc, yc, xt), yt


def process(path):
    """Process a single file."""
    print(f"Processing {path}")
    grid = grib2grid(path)
    dataset = grid2dataset(grid)
    print(f"Saving {path.stem} Results shapes: {shapes(dataset)}")
    torch.save(dataset, path.parent / f"{path.stem}.pt")


def search_and_process(idx=None):
    """Process the era5 dataset.

    Logic:
        - list all the .grib files
        - foreach file:
        - if the corresponding .pt file exists, skip
        - else:
        - read the file
        - convert to grid
        - convert to dataset
        - save the dataset
    """
    path = Path(os.getenv("ERA5_DATASET"))
    path.mkdir(parents=True, exist_ok=True)

    files = sorted(path.glob("**/*.grib"))
    print(files)

    if idx is not None:
        process(files[idx])
        return

    for f in files:
        if (path / f"{f.stem}.pt").exists():
            print(f"Skipping {f}")
            continue

        process(f)
        break


def combine_files(files):
    """Combine data from a file list into a single dataset.

    Args:
        files (list): list of files to combine

    Returns:
        dataset (StackDataset): combined dataset
    """
    (xc, yc, xt), yt = torch.load(files[0])

    for i, f in enumerate(files, 1):
        (xc_, yc_, xt_), yt_ = torch.load(f)
        xc = torch.cat((xc, xc_), dim=0)
        yc = torch.cat((yc, yc_), dim=0)
        xt = torch.cat((xt, xt_), dim=0)
        yt = torch.cat((yt, yt_), dim=0)

        print(
            f"{i}/{len(files)} : xc: {xc.shape}, yc: {yc.shape}, xt: {xt.shape},yt {yt.shape}"
        )

    inputs = StackDataset(xc, yc, xt)
    dataset = StackDataset(inputs, yt)

    return dataset


def combine_dataset():
    """Combine the train and val dataset into a single dataset."""
    path = Path(os.getenv("ERA5_DATASET"))

    files_trains = list(path.glob("train/*.pt"))
    files_val = list(path.glob("val/*.pt"))

    for files, name in zip([files_trains, files_val], ["train", "val"]):
        dataset = combine_files(files)
        torch.save(dataset, path / f"{name}.pt")


class ScaleDataset(data.Dataset):
    """Scale a dataset by mean and std."""

    def __init__(self, dataset, mean_yc, std_yc, mean_yt, std_yt):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
            meanx (float): Mean of the x coordinates.
            stdx (float): Standard deviation of the x coordinates.
            meany (float): Mean of the y coordinates.
            stdy (float): Standard deviation of the y coordinates.
        """
        self.dataset = dataset
        self.mean_yc = mean_yc
        self.std_yc = std_yc
        self.mean_yt = mean_yt
        self.std_yt = std_yt

    def __getitem__(self, i):
        """Get an item.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Normalized Inputs and targets of the item.
        """
        (xc, yc, xt), yt = self.dataset[i]
        yc = (yc - self.mean_yc) / self.std_yc
        yt = (yt - self.mean_yt) / self.std_yt
        return (xc, yc, xt), yt

    def __len__(self):
        """Get the number of items.

        Returns:
            int: Number of items.
        """
        return len(self.dataset)


def get_mean_std(dataset):
    """Get the mean and std of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to split.

    Returns:
        tuple: Mean and std of the dataset.
    """
    yc = torch.cat([i[0][1] for i in dataset])
    yt = torch.cat([i[1] for i in dataset])
    return yc.mean(0), yc.std(0), yt.mean(0), yt.std(0)


class Scale(nn.Module):
    """Scale the output of a model to match the scale of the target."""

    def __init__(self, mu, std):
        """Initialize the scale module.

        Args:
            mu (torch.Tensor): Mean of the target.
            std (torch.Tensor): Standard deviation of the target.
        """
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("std", std)

    def forward(self, y):
        """Scale the output.

        Args:
            y (torch.Tensor): Output of the model.
        """
        return y * self.std + self.mu


class ScaleRMSE(nn.Module):
    """Compute the RMSE of a model scaled to match the scale of the target."""

    def __init__(self, scale):
        """Initialize the module.

        Args:
            scale (Scale): Scale module.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x, y):
        """Compute the scaled RMSE.

        Args:
            x (torch.Tensor): Output of the model.
            y (torch.Tensor): Target.
        """
        return F.mse_loss(self.scale(x), self.scale(y)).sqrt()


def datasets():
    """Return the train and val dataset."""
    path = Path(os.getenv("ERA5_DATASET"))
    train = torch.load(path / "train.pt")
    val = torch.load(path / "val.pt")

    mean_yc, std_yc, mean_yt, std_yt = get_mean_std(train)
    train = ScaleDataset(train, mean_yc, std_yc, mean_yt, std_yt)
    val = ScaleDataset(val, mean_yc, std_yc, mean_yt, std_yt)
    metric = ScaleRMSE(Scale(mean_yt, std_yt))

    return train, val, metric


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--process", action="store_true")
    parser.add_argument("--idx", type=int, default=None)
    args = parser.parse_args()

    if args.process:
        search_and_process(args.idx)
    else:
        combine_dataset()
