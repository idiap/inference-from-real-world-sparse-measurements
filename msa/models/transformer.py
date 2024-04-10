# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""All variants of the Transformer for the experiments and many regressions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import Net

# from .positional_encoding import RFF, PosEncCat


class TFS(nn.Module):
    """Default Transformer for the experiments."""

    def __init__(
        self,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        nhead,
        nlayers_encoder,
        nlayers_decoder,
        pos_enc_freq=100.0,
        use_rff=False,
        use_same_pos_enc=False,
        use_mlps=False,
        dropout=0.0,
    ):
        """Initialize the model.

        Args:
            dim_x (int): Dimension of the input.
            dim_yc (int): Dimension of the context.
            dim_yt (int): Dimension of the target.
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            nlayers_encoder (int): Number of layers in the encoder.
            nlayers_decoder (int): Number of layers in the decoder.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.use_same_pos_enc = use_same_pos_enc
        if use_same_pos_enc:
            # if use_mlps:
            self.pos_encoder = Net([dim_x, dim_h, dim_h, dim_h])
            self.val_encoder = Net([dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net([dim_h, dim_h, dim_yt])

            # else:
            #     self.pos_encoder = nn.Linear(dim_x * dim_h * 2, dim_h)
            #     self.val_encoder = nn.Linear(dim_yc * dim_h * 2, dim_h)
            #     self.decoder = nn.Linear(dim_h, dim_yt)

            # if use_rff:
            #     self.pe_x = RFF(dim_x, dim_x * dim_h * 2)
            #     self.pe_v = RFF(dim_yc, dim_yc * dim_h * 2)
            # else:
            #     self.pe_x = PosEncCat(dim_h, pos_enc_freq)
            #     self.pe_v = PosEncCat(dim_h, pos_enc_freq)
        else:
            self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
            self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        encoder_layer = nn.TransformerEncoderLayer(
            dim_h, nhead, 2 * dim_h, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers_encoder)

        decoder_layer = nn.TransformerDecoderLayer(
            dim_h, nhead, 2 * dim_h, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers_decoder)

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
            latents = ce_x + ce_y
            q = self.pos_encoder(xt)
        else:
            latents = self.encoder(torch.cat((xc, yc), dim=-1))
            q = self.q_encoder(xt)
        latents = self.transformer_encoder(latents)
        z = self.transformer_decoder(q, latents)
        return self.decoder(z)


class TFSEncoderPosLayer(nn.Module):
    """Transformer layer with residual connection and linear layer."""

    def __init__(self, in_dim, dim_h, nhead, dropout=0.0):
        """Initialize the layer.

        Args:
            in_dim (int): Dimension of the input.
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            in_dim, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(in_dim, dim_h)

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(self.linear1(x))

        return x

    def _sa_block(self, x):
        """Self-attention block.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)


class TFSEncoderFullLayer(nn.Module):
    """Standard Transformer layer."""

    def __init__(self, dim_h, nhead, dropout=0.0):
        """Initialize the layer.

        Args:
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            dim_h, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(dim_h, 2 * dim_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2 * dim_h, dim_h)

        self.norm1 = nn.LayerNorm(dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x):
        """Self-attention block.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class TFSEncoderFullPreNormLayer(nn.Module):
    """Standard Transformer layer with pre-normalization."""

    def __init__(self, dim_h, nhead, dropout=0.0):
        """Initialize the layer.

        Args:
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            dim_h, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(dim_h, 2 * dim_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(2 * dim_h, dim_h)

        self.norm1 = nn.LayerNorm(dim_h)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        """Self-attention block.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        """Feed-forward block.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class TFSpos(nn.Module):
    """Transformer with concatenated positional information."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.blocks = nn.ModuleList(
                [TFSEncoderPosLayer(dim_h + dim_x, dim_h, nhead)] * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TFSEncoderPosLayer(dim_h + dim_x, dim_h, nhead)
                    for _ in range(nlayers)
                ]
            )

        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(torch.cat((xc, latents), dim=-1))

        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]

        return self.decoder(z)


class TFSfull(nn.Module):
    """Transformer with full layer."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.blocks = nn.ModuleList([TFSEncoderFullLayer(dim_h, nhead)] * nlayers)
        else:
            self.blocks = nn.ModuleList(
                [TFSEncoderFullLayer(dim_h, nhead) for _ in range(nlayers)]
            )

        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(latents)

        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]

        return self.decoder(z)


class TFSfullpn(nn.Module):
    """Transformer with full layer and pre-norm."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.blocks = nn.ModuleList(
                [TFSEncoderFullPreNormLayer(dim_h, nhead)] * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [TFSEncoderFullPreNormLayer(dim_h, nhead) for _ in range(nlayers)]
            )

        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(latents)

        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]

        return self.decoder(z)


class TFSEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, in_dim, dim_h, nhead, dropout=0.0):
        """Initialize the model.

        Args:
            in_dim (int): Dimension of the input.
            dim_h (int): Dimension of the hidden layers.
            nhead (int): Number of heads in the multi-head attention.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            in_dim, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(in_dim, dim_h)

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(dim_h)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(self.linear1(x))

        return x

    def _sa_block(self, x):
        """Self-attention block.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class EncoderOnly(nn.Module):
    """Only use the encoder part of the transformer then use distance-based decoding."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h + dim_x, dim_h, dim_h, dim_yt])

        if share_blocks:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h + dim_x, dim_h, nhead)] * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h + dim_x, dim_h, nhead) for _ in range(nlayers)]
            )

        self.log_strength = nn.Parameter(torch.zeros(1))

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(torch.cat((xc, latents), dim=-1))

        scores = self.scaled_dist(xt, xc)
        z = scores.bmm(latents)
        return self.decoder(torch.cat((z, xt), dim=-1))

    def scaled_dist(self, x, y):
        """Compute the scaled distance between x and y.

        Args:
            x (torch.Tensor): Input.
            y (torch.Tensor): Input.

        Returns:
            torch.Tensor: Distance.
        """
        pseudo_dist = y.norm(dim=-1).unsqueeze(1) ** 2 - 2 * torch.bmm(
            x, y.transpose(1, 2)
        )
        return F.softmax(-self.log_strength.exp() * pseudo_dist, dim=-1)


class TFSDist(nn.Module):
    """Transformer with distance-based decoding. encode the position as well."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[2 * dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h + dim_x, dim_h, nhead)] * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h + dim_x, dim_h, nhead) for _ in range(nlayers)]
            )

        self.log_strength = nn.Parameter(torch.zeros(1))

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(torch.cat((xc, latents), dim=-1))

        q = self.q_encoder(xt)
        scores = self.scaled_dist(xt, xc)
        z = scores.bmm(latents)
        z = torch.cat((z, q), dim=-1)

        return self.decoder(z)

    def scaled_dist(self, x, y):
        """Compute the scaled distance between x and y.

        Args:
            x (torch.Tensor): Input.
            y (torch.Tensor): Input.

        Returns:
            torch.Tensor: Distance.
        """
        pseudo_dist = y.norm(dim=-1).unsqueeze(1) ** 2 - 2 * torch.bmm(
            x, y.transpose(1, 2)
        )
        return F.softmax(-self.log_strength.exp() * pseudo_dist, dim=-1)


class TFSCross(nn.Module):
    """Transformer with cross-attention decoding."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h + dim_x, dim_h, nhead)] * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [TFSEncoder(dim_h + dim_x, dim_h, nhead) for _ in range(nlayers)]
            )

        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(torch.cat((xc, latents), dim=-1))

        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]

        return self.decoder(z)


class TFSOne(nn.Module):
    """Transformer with one-step decoding."""

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
        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        dim_h, nhead, 2 * dim_h, 0.0, batch_first=True
                    )
                ]
                * nlayers
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        dim_h, nhead, 2 * dim_h, 0.0, batch_first=True
                    )
                    for _ in range(nlayers)
                ]
            )

        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): Context input.
            yc (torch.Tensor): Context output.
            xt (torch.Tensor): Target input.

        Returns:
            torch.Tensor: Target output.
        """
        latents = self.encoder(torch.cat((xc, yc), dim=-1))

        for block in self.blocks:
            latents = block(latents)

        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]

        return self.decoder(z)
