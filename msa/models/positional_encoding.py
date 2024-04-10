# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Positional encodings."""
from math import pi, sqrt
from math import tau as two_pi

import torch
import torch.nn as nn


class SinCosPositionalEncoding(nn.Module):
    """Sine-Cosine Positional Encoding à la *Attention is all you need* [1].

    The encodings are given by :

    [..., cos(2π f^(j/n) x), sin(2π f^(j/n) x), ...], j = 0,...,m-1

    We defined them with the frequencies which is different than in the
    original work were they work with the wavelength (even though they are
    related with the formula f = 1 / λ)

    Where σ should be grid searched for all dataset as suggested here [2]
    [1] https://arxiv.org/abs/1706.03762
    [2] https://arxiv.org/abs/2006.10739 p.7
    """

    def __init__(self, n, f=(1.0 / 10000), s=(1.0 / two_pi)):
        """Initialize the positional encoding.

        Args:
            n (int): Number of frequencies.
            f (float): Base frequenyc.
            s (float): Scale of the frequencies.
        """
        super().__init__()

        j = torch.arange(n).repeat_interleave(2).float() / n
        phase = (torch.arange(2 * n) % 2) * (pi / 2)

        self.register_buffer("fs", s * (f**j))
        self.register_buffer("phase", phase)
        self.log_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward pass."""
        x = x.float()
        e = self.log_scale.exp()
        pe = torch.cos(two_pi * self.fs * e * x[..., None] + self.phase)
        scale = sqrt(1 / (2 * len(self.fs)))
        return pe * scale


class PosEncCat(nn.Module):
    """Positional Encoding that concatenates the input with the results of the positional encodings."""

    def __init__(self, n, *args):
        """Initialize the positional encoding."""
        super().__init__()
        self.pe = SinCosPositionalEncoding(n, *args)

    def forward(self, x):
        """Forward pass."""
        return self.pe(x).view(x.size(0), x.size(1), -1)


class RFF(torch.nn.Module):
    """Implements Random Fourier Features for the gaussian kernel.

    To be used as a positional encoding since the dot product will be the value of
    the gaussian kernel in expectation.
    This was introduced as a positional encoding in [1].

    [1]: Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S.,
         Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T. and Ng, R.,
         2020.  Fourier features let networks learn high frequenyc functions in
         low dimensional domains. NeurIPS 2020.
    """

    def __init__(self, input_dims, feature_dims, sigma=1.0):
        """Initialize RFF.

        Args:
            input_dims: int, the dimensionaliyt of the input feature
            feature_dims: int, the dimensionaliyt of the random Fourier features
            sigma: float, the standard deviation from which to draw the random
            matrix (defines the gamma parameter for the gaussian kernel)
            (default: 1.)
        """
        super().__init__()

        self.register_buffer("beta", torch.randn(feature_dims // 2, input_dims) * sigma)

    def forward(self, x):
        """Forward pass."""
        bx = torch.einsum("...i,ji->...j", x, self.beta)
        scale = sqrt(1 / len(self.beta))
        return torch.cat([torch.cos(bx), torch.sin(bx)], dim=-1) * scale
