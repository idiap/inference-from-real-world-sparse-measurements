# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Metrics for the wind prediction task."""
from functools import partial

import torch.nn.functional as F

from ..datasets.windprocessing import carthesian2polar


def rmse(x, y):
    """Compute the RMSE.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: RMSE.
    """
    return F.mse_loss(x, y).sqrt()


def angle_mae(x, y):
    """Compute the MAE of the angle.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: MAE of the angle.
    """
    _, ax = carthesian2polar(x[..., 0], x[..., 1])
    _, ay = carthesian2polar(y[..., 0], y[..., 1])
    return F.l1_loss(ax, ay)


def norm_mae(x, y):
    """Compute the MAE of the norm.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: MAE of the norm.
    """
    nx, _ = carthesian2polar(x[..., 0], x[..., 1])
    ny, _ = carthesian2polar(y[..., 0], y[..., 1])
    return F.l1_loss(nx, ny)


def rel_mse(x, y):
    """Compute the relative MSE.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: Relative MSE.
    """
    return rmse(x, y) / y.abs().max()


def rel_bias(x, y, dim=0):
    """Compute the relative bias.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.
        dim (int): Dimension along which to compute the relative bias.

    Returns:
        torch.Tensor: Relative bias.
    """
    return (x[..., dim] - y[..., dim]).mean() / y[..., dim].mean(-1)


rel_bias_x = partial(rel_bias, dim=0)
rel_bias_x.__name__ = "rel_bias_x"
rel_bias_y = partial(rel_bias, dim=1)
rel_bias_y.__name__ = "rel_bias_y"


def rsd(x, y):
    """Compute the relative standard deviation.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: Relative standard deviation.
    """
    return x.std() / y.std()


def r2(x, y):
    """Compute the R2 score.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: R2 score.
    """
    return 1 - F.mse_loss(x, y) / y.var()


def nse(x, y):
    """Compute the Nash-Sutcliffe efficienyc.

    Args:
        x (torch.Tensor): Output of the model.
        y (torch.Tensor): Target.

    Returns:
        torch.Tensor: Nash-Sutcliffe efficienyc.
    """
    return 1 - (F.mse_loss(x, y, reduction="sum") / (y - y.mean(0)).pow(2).sum())


metrics = [
    rmse,
    angle_mae,
    norm_mae,
    rel_mse,
    rel_bias_x,
    rel_bias_y,
    rsd,
    r2,
    nse,
]
