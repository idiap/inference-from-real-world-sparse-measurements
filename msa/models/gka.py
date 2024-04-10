# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Gaussian Kernel Averaging."""
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class GKA(nn.Module):
    """Gaussian Kernel Averaging with a learned sigmas."""

    def __init__(self, dim_x, dim_yc, dim_yt, dim_h, nlayers=4):
        """Initialize the model.

        Args:
            dim_x (int): input dimension
            dim_yc (int): context output dimension
            dim_yt (int): target output dimension
            dim_h (int): hidden dimension
            nlayers (int): number of hidden layers
        """
        super().__init__()
        self.log_sigmas = MLP(dim_x, dim_x, dim_h, nlayers)

    def sigmas(self, xt):
        """Compute sigmas.

        Args:
            xt (torch.Tensor): target input

        Returns:
            torch.Tensor: sigmas
        """
        return self.log_sigmas(xt).exp()

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context input
            yc (torch.Tensor): context output
            xt (torch.Tensor): target input

        Returns:
            torch.Tensor: target output
        """
        # (B, C, 1, X) - (B, 1, T, X) = (B, C, T, X)
        A = (xc.unsqueeze(2) - xt.unsqueeze(1)) ** 2
        # sum 3 (B, C, T, X) * (B, 1, T, X) = (B, C, T)
        P = -(A * self.sigmas(xt).unsqueeze(1)).sum(3)
        # (B, C, T)
        S = F.softmax(P, dim=1)
        # (B, C, 1, Y) * (B, C, T, 1)
        return (yc.unsqueeze(2) * S.unsqueeze(3)).sum(1)
