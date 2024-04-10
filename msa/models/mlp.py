# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Multi-Layer Perceptron."""
from itertools import tee
from typing import Iterable, Iterator

import torch.nn as nn


def pair(it: Iterable) -> Iterator:
    """Pairwise iterator.

    Args:
        it (Iterable): iterable (a,b,c,d)

    Returns:
        Iterator: pairwise iterator ((a,b), (b,c), (c,d)))
    """
    a, b = tee(it)
    next(b, None)
    yield from zip(a, b)


def join(L: list, sep: object) -> Iterator:
    """Join iterator.

    Args:
        L (list): list of objects
        sep (object): separator

    Returns:
        Iterator: joined iterator (a, sep, b, sep, c, sep, d)
    """
    for i in L[:-1]:
        yield i
        yield sep
    yield L[-1]


def Net(dims: Iterable[int]) -> nn.Sequential:
    """Multi-Layer Perceptron.

    Args:
        dims (Iterable[int]): list of dimensions

    Returns:
        nn.Sequential: MLP
    """
    layers = [nn.Linear(d1, d2) for d1, d2 in pair(dims)]
    return nn.Sequential(*join(layers, nn.ReLU()))


def MLP(in_dim, out_dim, units, hidden_layers=4):
    """Multi-Layer Perceptron.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        units (int): number of units
        hidden_layers (int, optional): number of hidden layers. Defaults to 4.

    Returns:
        nn.Sequential: MLP
    """
    return Net([in_dim, *[units] * hidden_layers, out_dim])
