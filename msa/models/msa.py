# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""MSA model."""
import torch
import torch.nn as nn

from .mlp import Net

# from .positional_encoding import RFF, PosEncCat
from .transformer import TFSEncoder


class MSA(nn.Module):
    """Default Transformer for the experiments."""

    def __init__(
        self,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        nhead,
        nlayers,
        use_same_pos_enc=False,
        dropout=0.0,
    ):
        """Initialize the model.

        Args:
            dim_x (int): Dimension of the input.
            dim_yc (int): Dimension of the context.
            dim_yt (int): Dimension of the target.
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            nlayers (int): Number of layers.
            use_same_pos_enc (bool): Whether to use the same positional encoding for the context and the target.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.use_same_pos_enc = use_same_pos_enc
        if use_same_pos_enc:
            self.pos_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])
            self.val_encoder = Net(dims=[dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[dim_h, dim_h, dim_yt])

        else:
            self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
            self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        encoder_layer = nn.TransformerEncoderLayer(
            dim_h, nhead, 2 * dim_h, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context value.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target value.
        """
        if self.use_same_pos_enc:
            ce_x = self.pos_encoder(xc)
            ce_y = self.val_encoder(yc)
            ce = ce_x + ce_y
            te = self.pos_encoder(xt)

        else:
            ce = self.encoder(torch.cat((xc, yc), dim=-1))
            te = self.q_encoder(xt)
        enc = torch.cat((ce, te), dim=1)
        z = self.transformer_encoder(enc)
        L = xt.shape[1]
        z = z[:, -L:, :]
        return self.decoder(z)


class MSAEncoderOnly(nn.Module):
    """MSA for baselines."""

    def __init__(
        self,
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
            dim_x (int): Dimension of the input.
            dim_yc (int): Dimension of the context.
            dim_yt (int): Dimension of the target.
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            nlayers (int): Number of layers.
            share_blocks (bool): Whether to share the blocks.
        """
        super().__init__()
        self.context_encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.target_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])

        if share_blocks:
            self.blocks = nn.ModuleList([TFSEncoder(dim_h, dim_h, nhead)] * nlayers)
        else:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h, dim_h, nhead) for _ in range(nlayers)]
            )

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents_context = self.context_encoder(torch.cat((xc, yc), dim=-1))
        latents_target = self.target_encoder(xt)
        latents = torch.cat((latents_context, latents_target), dim=1)
        L = xt.shape[1]

        for block in self.blocks:
            latents = block(latents)

        return self.decoder(latents[:, -L:, :])
