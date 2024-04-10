# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Utilities to load the datasets."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


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


class ScaleMSE(nn.Module):
    """Compute the MSE of a model scaled to match the scale of the target."""

    def __init__(self, scale):
        """Initialize the module.

        Args:
            scale (Scale): Scale module.
        """
        super().__init__()
        self.scale = scale

    def forward(self, x, y):
        """Compute the scaled MSE.

        Args:
            x (torch.Tensor): Output of the model.
            y (torch.Tensor): Target.
        """
        return F.mse_loss(self.scale(x), self.scale(y))


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


def shapes(obj):
    """Recursively get the shapes of a nested object."""
    if isinstance(obj, torch.Tensor):
        return obj.shape

    elif isinstance(obj, tuple):
        return tuple(shapes(o) for o in obj)


class StackDataset(Dataset):
    """Stacks multiple datasets together."""

    def __init__(self, *datasets):
        """Initialize the dataset.

        Args:
            datasets list(torch.utils.data.Dataset): The datasets to stack.
        """
        self.datasets = datasets
        assert all(len(d) == len(datasets[0]) for d in datasets)

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.datasets[0])

    def __getitem__(self, index):
        """Get an item from the dataset.

        Args:
            index (int): The index of the item to get.

        Returns:
            tuple: The item at the given index.
        """
        return tuple(d[index] for d in self.datasets)
