# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""LSTM baseline for reviews."""
import torch
import torch.nn as nn

from .mlp import Net


class LSTM(nn.Module):
    """LSTM baseline for reviews.

    Same idea than the MSA baseline: combine both source and target into one sequence.
    Is supposed to be worse than the MSA baseline as it does not have the right invariance properties.
    """

    def __init__(
        self,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        nlayers,
        use_same_pos_enc=False,
    ):
        """Initialize the model.

        Args:
            dim_x (int): Dimension of the input.
            dim_yc (int): Dimension of the context.
            dim_yt (int): Dimension of the target.
            dim_h (int): Dimension of the hidden layers.
            nlayers (int): Number of layers in the LSTM.
            use_same_pos_enc (bool): Whether to use the same positional encoding for the context and the target.
        """
        super().__init__()

        self.use_same_pos_enc = use_same_pos_enc

        if use_same_pos_enc:
            self.pos_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])
            self.val_encoder = Net(dims=[dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[dim_h, dim_h, dim_yt])

        else:
            self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
            self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])

        self.lstm = nn.LSTM(
            dim_h, dim_h, nlayers, batch_first=True, bidirectional=False
        )

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
        z, _ = self.lstm(enc)
        L = xt.shape[1]
        z = z[:, -L:, :]
        return self.decoder(z)
