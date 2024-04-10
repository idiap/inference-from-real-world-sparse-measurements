# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Ablation: GEN without graph."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import Net


def neighbors_edges(pos, n=3):
    """Return egdes between neighbors.

    Args:
        pos (torch.Tensor): positions
        n (int): number of neighbors

    Returns:
        torch.Tensor: edges
    """
    dists = torch.norm(pos[:, None, :, :] - pos[:, :, None, :], dim=3)
    receivers = dists.argsort()[..., 1 : n + 1].flatten(1)
    senders = torch.arange(pos.shape[1], device=pos.device).repeat_interleave(n)
    senders = senders[None, ...].repeat(pos.shape[0], 1)

    return torch.cat((senders, receivers), dim=1), torch.cat(
        (receivers, senders), dim=1
    )


class Block(nn.Module):
    """Graph Neural Network Block."""

    def __init__(self, in_dim, dim_h):
        """Initialize the model.

        Args:
            in_dim (int): input dimension
            dim_h (int): hidden dimension
        """
        super().__init__()
        self.message = nn.Linear(2 * in_dim, in_dim)
        self.node = nn.Linear(2 * in_dim, dim_h)
        self.bn1 = nn.LayerNorm(in_dim)
        self.bn2 = nn.LayerNorm(dim_h)

    def forward(self, nodes, senders, receivers):
        """Message passing.

        Args:
            nodes (torch.Tensor): nodes
            senders (torch.Tensor): senders
            receivers (torch.Tensor): receivers

        Returns:
            torch.Tensor: nodes
        """
        senders = senders[:, :, None].repeat(1, 1, nodes.shape[-1])
        receivers = receivers[:, :, None].repeat(1, 1, nodes.shape[-1])

        r = torch.gather(nodes, 1, receivers)
        s = torch.gather(nodes, 1, senders)

        messages = self.message(torch.cat((s, r), dim=-1))
        messages = self.bn1(messages)
        inbox = torch.zeros(nodes.shape, device=nodes.device)
        inbox = inbox.scatter_add(1, receivers, messages)
        return self.bn2(self.node(torch.cat((nodes, inbox), dim=-1)))


class GEN_nograph(nn.Module):
    """GEN without graph structure.

    No bottleck, but way slower than GEN.
    """

    def __init__(
        self,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        message_passing_steps,
        share_blocks=True,
        encoder=None,
    ):
        """Initialize the model.

        Args:
            dim_x (int): input dimension
            dim_yc (int): condition dimension
            dim_yt (int): target dimension
            dim_h (int): hidden dimension
            message_passing_steps (int): number of message passing steps
            share_blocks (bool, optional): share blocks. Defaults to True.
            encoder (nn.Module, optional): encoder. Defaults to None.
        """
        super().__init__()

        if not encoder:
            self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        else:
            self.encoder = encoder

        self.decoder = Net(dims=[dim_h + dim_x, dim_h, dim_h, dim_yt])

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [Block(dim_h + dim_x, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [Block(dim_h + dim_x, dim_h) for _ in range(message_passing_steps)]
            )

        self.log_strength = nn.Parameter(torch.zeros(1))

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): condition
            yc (torch.Tensor): target
            xt (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))
        senders, receivers = neighbors_edges(xc)
        for block in self.gn_blocks:
            latents = block(torch.cat((xc, latents), dim=-1), senders, receivers)

        scores = self.scaled_dist(xt, xc)
        z = scores.bmm(latents)
        return self.decoder(torch.cat((z, xt), dim=-1))

    def scaled_dist(self, x, y):
        """Compute scaled distance.

        Args:
            x (torch.Tensor): input
            y (torch.Tensor): target

        Returns:
            torch.Tensor: distance
        """
        pseudo_dist = y.norm(dim=-1).unsqueeze(1) ** 2 - 2 * torch.bmm(
            x, y.transpose(1, 2)
        )
        return F.softmax(-self.log_strength.exp() * pseudo_dist, dim=-1)
