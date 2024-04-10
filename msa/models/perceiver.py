# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Perceiver model."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import Net


class PerceiverLayer(nn.Module):
    """Perceiver layer."""

    def __init__(self, n_latents, in_dim, dim_h, nhead, dropout=0.0):
        """Initialize the layer.

        Args:
            n_latents (int): number of latents
            in_dim (int): input dimension
            dim_h (int): hidden dimension
            nhead (int): number of heads
            dropout (float, optional): dropout. Defaults to 0.0.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            in_dim, nhead, dropout=dropout, batch_first=True
        )

        self.latents = nn.Parameter(torch.randn(n_latents, in_dim))

        self.linear1 = nn.Linear(in_dim, dim_h)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.norm1(self._cross_block(x))
        x = self.norm2(self.linear1(x))

        return x

    def _cross_block(self, x):
        """Cross attention block.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        B, _, _ = x.shape
        y = self.latents[None].expand(B, -1, -1)
        x = x + self.self_attn(y, x, x, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        """Feed forward block.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class Perceiver(nn.Module):
    """Perceiver model."""

    def __init__(
        self,
        n_latents,
        n_x,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        nhead,
        nlayers,
        share_blocks=True,
    ):
        """Initialize the model.

        Args:
            n_latents (int): number of latents
            n_x (int): number of positions
            dim_x (int): position dimension
            dim_yc (int): context dimension
            dim_yt (int): target dimension
            dim_h (int): hidden dimension
            nhead (int): number of heads
            nlayers (int): number of layers
            share_blocks (bool, optional): share blocks. Defaults to True.
        """
        super().__init__()
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h + dim_x, dim_h, dim_h, dim_yt])

        self.register_buffer("map_pos", torch.eye(n_latents, n_x))

        if share_blocks:
            self.blocks = nn.ModuleList(
                [PerceiverLayer(n_latents, dim_h + dim_x, dim_h, nhead)] * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    PerceiverLayer(n_latents, dim_h + dim_x, dim_h, nhead)
                    for _ in range(nlayers)
                ]
            )

        self.log_strength = nn.Parameter(torch.zeros(1))

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context positions
            yc (torch.Tensor): context features
            xt (torch.Tensor): target positions

        Returns:
            torch.Tensor: output tensor
        """
        p = self.map_pos @ xc
        scores = self.scaled_dist(xc, p)
        emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)

        for block in self.blocks:
            latents = block(torch.cat((p, latents), dim=-1))
        scores = self.scaled_dist(xt, p)
        z = scores.bmm(latents)
        return self.decoder(torch.cat((z, xt), dim=-1))

    def scaled_dist(self, x, y):
        """Compute scaled distance.

        Args:
            x (torch.Tensor): input tensor
            y (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        pseudo_dist = y.norm(dim=-1).unsqueeze(1) ** 2 - 2 * torch.bmm(
            x, y.transpose(1, 2)
        )
        return F.softmax(-self.log_strength.exp() * pseudo_dist, dim=-1)
