# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Load and preprocess the darcy flow dataset."""
import os

import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset

from .utils import Scale, ScaleRMSE, StackDataset


class ScaleDataset(Dataset):
    """Scale a dataset by mean and std."""

    def __init__(self, dataset, mean_yc, std_yc, mean_yt, std_yt):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
            mean_yc (torch.Tensor): Mean of the context targets.
            std_yc (torch.Tensor): Std of the context targets.
            mean_yt (torch.Tensor): Mean of the target targets.
            std_yt (torch.Tensor): Std of the target targets.
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


def datasets():
    """Load the darcy Flow dataset.

    Returns:
        tuple: Train dataset, validation dataset, metric.
    """
    load_dotenv()
    train_dataset = torch.load(os.getenv("darcy_TRAIN_PATH"))
    xc, yc, xt, yt = (t.flatten(0, 1) for t in train_dataset)
    train_dataset = StackDataset(StackDataset(xc, yc, xt), yt)
    mean_yc, std_yc = yc.mean(), yc.std()
    mean_yt, std_yt = yt.mean(), yt.std()
    val_dataset = torch.load(os.getenv("darcy_VAL_PATH"))
    xc, yc, xt, yt = (t.flatten(0, 1) for t in val_dataset)
    val_dataset = StackDataset(StackDataset(xc, yc, xt), yt)

    train_dataset = ScaleDataset(train_dataset, mean_yc, std_yc, mean_yt, std_yt)
    val_dataset = ScaleDataset(val_dataset, mean_yc, std_yc, mean_yt, std_yt)
    metric = ScaleRMSE(Scale(mean_yt, std_yt))

    return train_dataset, val_dataset, metric
