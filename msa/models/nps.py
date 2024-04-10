# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Neural Processes."""
import torch
import torch.nn as nn

from .mlp import Net

# from .positional_encoding import RFF, PosEncCat


class CNP(nn.Module):
    """Default Neural Process Implementation."""

    def __init__(
        self,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        pos_enc_freq=100.0,
        use_rff=False,
        use_same_pos_enc=False,
        use_mlps=False,
    ):
        """Initialize the model.

        Args:
            dim_x (int): input dimension
            dim_yc (int): context output dimension
            dim_yt (int): target output dimension
            dim_h (int): hidden dimension
        """
        super().__init__()
        self.use_same_pos_enc = use_same_pos_enc
        if self.use_same_pos_enc:
            # if use_mlps:
            self.pos_encoder = Net([dim_x, dim_h, dim_h, dim_h])
            self.val_encoder = Net([dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net([2 * dim_h, dim_h, dim_yt])

            # else:
            #     self.pos_encoder = nn.Linear(dim_x * dim_h * 2, dim_h)
            #     self.val_encoder = nn.Linear(dim_yc * dim_h * 2, dim_h)
            #     self.decoder = nn.Linear(2 * dim_h, dim_yt)

            # if use_rff:
            #     self.pe_x = RFF(dim_x, dim_x * dim_h * 2)
            #     self.pe_v = RFF(dim_yc, dim_yc * dim_h * 2)
            # else:
            #     self.pe_x = PosEncCat(dim_h, pos_enc_freq)
            #     self.pe_v = PosEncCat(dim_h, pos_enc_freq)
        else:
            self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[2 * dim_h, dim_h, dim_h, dim_yt])
            self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context input
            yc (torch.Tensor): context output
            xt (torch.Tensor): target input

        Returns:
            torch.Tensor: target output
        """
        if self.use_same_pos_enc:
            ce_x = self.pos_encoder(xc)
            ce_y = self.val_encoder(yc)
            inputs = ce_x + ce_y
            q = self.pos_encoder(xt)
        else:
            inputs = self.encoder(torch.cat((xc, yc), dim=-1))
            q = self.q_encoder(xt)

        latents = inputs.mean(1, keepdim=True)
        z = latents.repeat(1, xt.shape[1], 1)
        return self.decoder(torch.cat((z, q), dim=-1))


class CNPDist(nn.Module):
    """Neural Process with distance-based attention."""

    def __init__(self, dim_x, dim_yc, dim_yt, dim_h):
        """Initialize the model.

        Args:
            dim_x (int): input dimension
            dim_yc (int): context output dimension
            dim_yt (int): target output dimension
            dim_h (int): hidden dimension
        """
        super().__init__()
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[2 * dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context input
            yc (torch.Tensor): context output
            xt (torch.Tensor): target input

        Returns:
            torch.Tensor: target output
        """
        inputs = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = inputs.mean(1, keepdim=True)
        z = latents.repeat(1, xt.shape[1], 1)
        q = self.q_encoder(xt)
        return self.decoder(torch.cat((z, q), dim=-1))


class CNPCross(nn.Module):
    """Neural Process with cross-attention decoding."""

    def __init__(self, dim_x, dim_yc, dim_yt, dim_h, nhead):
        """Initialize the model.

        Args:
            dim_x (int): input dimension
            dim_yc (int): context output dimension
            dim_yt (int): target output dimension
            dim_h (int): hidden dimension
            nhead (int): number of attention heads
        """
        super().__init__()
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])
        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context input
            yc (torch.Tensor): context output
            xt (torch.Tensor): target input
        """
        inputs = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = inputs.mean(1, keepdim=True)
        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]
        return self.decoder(z)
