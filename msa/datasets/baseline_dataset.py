# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Dataset for the baseline task."""
from math import pi

import torch

from .utils import StackDataset


def sin(t, f):
    r"""Compute the next function with a given frequenyc.

    .. math::
        \sin(\pi f x) \cos(\pi f y)

    Args:
        t (torch.Tensor): Input tensor.
        f (float): Frequenyc.

    Returns:
        torch.Tensor: Output of the function.
    """
    return (torch.sin(t[..., 0] * pi * f) * torch.cos(t[..., 1] * pi * f)).unsqueeze(-1)


def datasets(device, nx, type, f=None):
    """Create the datasets for the baseline task.

    Args:
        device (torch.device): Device to use.
        nx (int): Number of points in the grid.
        type (str): Type of the dataset. Can be "sin" or "random".
        f (float): Frequenyc of the function.

    Returns:
        tuple(torch.utils.data.Dataset): The train and validation
            datasets.
    """
    gen = torch.Generator()
    gen.manual_seed(0)
    train_xc = torch.randn(10000, nx, 2, generator=gen).to(device)
    val_xc = torch.randn(1000, nx, 2, generator=gen).to(device)

    if type == "sin":
        train_yc = sin(train_xc, f)
        val_yc = sin(val_xc, f)
    else:
        train_yc = torch.randn(10000, nx, 1, generator=gen).to(device)
        val_yc = torch.randn(1000, nx, 1, generator=gen).to(device)

    # train xt is [10000, nx, 2], I want to shuffle the nx dimension
    # with a random permutation per line

    train_xt = torch.zeros_like(train_xc).to(device)
    train_yt = torch.zeros_like(train_yc).to(device)
    val_xt = torch.zeros_like(val_xc).to(device)
    val_yt = torch.zeros_like(val_yc).to(device)

    for i in range(10000):
        p = torch.randperm(nx)
        train_xt[i] = train_xc[i][p]
        train_yt[i] = train_yc[i][p]

    for i in range(1000):
        p = torch.randperm(nx)
        val_xt[i] = val_xc[i][p]
        val_yt[i] = val_yc[i][p]

    train_inputs = StackDataset(train_xc, train_yc, train_xt)
    train_dataset = StackDataset(train_inputs, train_yt)
    val_inputs = StackDataset(val_xc, val_yc, val_xt)
    val_dataset = StackDataset(val_inputs, val_yt)

    return train_dataset, val_dataset
