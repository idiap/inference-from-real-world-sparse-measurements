# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""GEoFNO models.

This file is modified from : https://github.com/neuraloperator/Geo-FNO/blob/main/elasticity/elas_geofno_v2.py (MIT).
One should have a different learning rate for IPHY and FNO2d.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geofno.elasticity.elas_geofno_v2 import IPHI, SpectralConv2d


class GEOFNO(nn.Module):
    """The overall network. It contains 4 layers of the Fourier layer."""

    def __init__(
        self,
        modes,
        width,
        dim_x,
        dim_yc,
        dim_yt,
        s1=40,
        s2=40,
        use_iphi=True,
    ):
        """Build The overall network. It contains 4 layers of the Fourier layer.

        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        super().__init__()
        self.width = width
        self.iphi = IPHI(width, dim_x) if use_iphi else None

        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(
            dim_x + dim_yc, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(width, width, modes, modes, s1, s2)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)
        self.conv3 = SpectralConv2d(width, width, modes, modes)
        self.conv4 = SpectralConv2d(width, width, modes, modes, s1, s2)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.b0 = nn.Conv2d(2, width, 1)
        self.b1 = nn.Conv2d(2, width, 1)
        self.b2 = nn.Conv2d(2, width, 1)
        self.b3 = nn.Conv2d(2, width, 1)
        self.b4 = nn.Conv1d(dim_x, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, dim_yt)

    def forward(self, xc, yc, xt):
        """Execute the forward pass of the network.

        Parameters:
        - xc (Tensor): The input position/value in FNO terms, shape (batch, Nx, d).
        - xc (Tensor): The input mesh (sampling mesh), shape (batch, Nx, 2).
        - yc (Tensor): The input features, shape (batch, Nx, d). `code` is a 3D tensor (not included in iphi).
        - xt (Tensor): The output mesh (query mesh), shape (batch, Nx, 2), corrected.
        """
        u = torch.cat([xc, yc], dim=-1)  # 3 ref: (a(x, y), x, y)
        grid = self.get_grid([yc.shape[0], self.s1, self.s2], u.device).permute(
            0, 3, 1, 2
        )

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=xc, iphi=self.iphi)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=xt, iphi=self.iphi)
        u3 = self.b4(xt.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        """Get the grid for the input.

        Parameters:
        - shape (Tuple): The shape of the grid.
        - device (str): The device to use.

        Returns:
        - Tensor: The grid.
        """
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def optim_groups(self, iphi_lr):
        """Don't optimize the iphi with the same learning rate.

        Args:
            lr (float): learning rate
        """
        if self.iphi is None:
            return self.parameters()

        total_keys = dict(self.named_parameters()).keys()
        iphi_keys = {k for k in total_keys if k.startswith("iphi")}
        other_keys = {k for k in total_keys if k not in iphi_keys}

        inter = other_keys & iphi_keys
        union = other_keys | iphi_keys
        assert len(inter) == 0
        assert len(union) == len(total_keys)

        return [
            {"params": self.iphi.parameters(), "lr": iphi_lr},
            {"params": [p for n, p in self.named_parameters() if n in other_keys]},
        ]


class GEOFNO_3layers(nn.Module):
    """The overall network. It contains 3 layers of the Fourier layer."""

    def __init__(
        self, modes, width, dim_x, dim_yc, dim_yt, s=20, hidden_layer=64, use_iphi=True
    ):
        """Build the overall network. It contains 4 layers of the Fourier layer.

        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        super().__init__()
        self.width = width
        self.iphi = IPHI(width, dim_x) if use_iphi else None

        self.s = s

        self.fc0 = nn.Linear(
            dim_x + dim_yc, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(width, width, modes, modes, s, s)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes, s, s)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.b0 = nn.Conv2d(2, width, 1)
        self.b1 = nn.Conv2d(2, width, 1)
        self.b2 = nn.Conv1d(dim_x, width, 1)

        self.fc1 = nn.Linear(width, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, dim_yt)

    def forward(self, xc, yc, xt):
        """Execute the forward pass of the network.

        Parameters:
        - xc (Tensor): The input position/value in FNO terms, shape (batch, Nx, d).
        - xc (Tensor): The input mesh (sampling mesh), shape (batch, Nx, 2).
        - yc (Tensor): The input features, shape (batch, Nx, d). `code` is a 3D tensor (not included in iphi).
        - xt (Tensor): The output mesh (query mesh), shape (batch, Nx, 2), corrected.
        """
        u = torch.cat([xc, yc], dim=-1)  # 3 ref: (a(x, y), x, y)
        grid = self.get_grid([yc.shape[0], self.s, self.s], u.device)
        grid = grid.permute(0, 3, 1, 2)

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=xc, iphi=self.iphi)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv2(uc, x_out=xt, iphi=self.iphi)
        u3 = self.b2(xt.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        """Get the grid for the input.

        Parameters:
        - shape (Tuple): The shape of the grid.
        - device (str): The device to use.

        Returns:
        - Tensor: The grid.
        """
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def optim_groups(self, iphi_lr):
        """Don't optimize the iphi with the same learning rate.

        Args:
            lr (float): learning rate
        """
        if self.iphi is None:
            return self.parameters()

        total_keys = dict(self.named_parameters()).keys()
        iphi_keys = {k for k in total_keys if k.startswith("iphi")}
        other_keys = {k for k in total_keys if k not in iphi_keys}

        inter = other_keys & iphi_keys
        union = other_keys | iphi_keys
        assert len(inter) == 0
        assert len(union) == len(total_keys)

        return [
            {"params": self.iphi.parameters(), "lr": iphi_lr},
            {"params": [p for n, p in self.named_parameters() if n in other_keys]},
        ]
