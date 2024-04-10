# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Import valid convolutions from pyg."""
import inspect

import torch_geometric.nn as gn
from torch_geometric.nn import (
    ARMAConv,
    CGConv,
    ChebConv,
    DNAConv,
    FastRGCNConv,
    FiLMConv,
    GATConv,
    GatedGraphConv,
    GATv2Conv,
    GCNConv,
    GENConv,
    GeneralConv,
    GINConv,
    GINEConv,
    GMMConv,
    GraphConv,
    HEATConv,
    LEConv,
    LGConv,
    MFConv,
    NNConv,
    PDNConv,
    PNAConv,
    RGATConv,
    RGCNConv,
    SAGEConv,
    SGConv,
    SplineConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
)

convs = [
    ARMAConv,
    CGConv,
    ChebConv,
    DNAConv,
    FastRGCNConv,
    FiLMConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GENConv,
    GINConv,
    GINEConv,
    GMMConv,
    GatedGraphConv,
    GeneralConv,
    GraphConv,
    HEATConv,
    LEConv,
    LGConv,
    MFConv,
    NNConv,
    PDNConv,
    PNAConv,
    RGATConv,
    RGCNConv,
    SAGEConv,
    SGConv,
    SplineConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
]
name2conv = {conv.__name__: conv for conv in convs}


if __name__ == "__main__":
    classes = inspect.getmembers(gn, inspect.isclass)

    filtered_classes = []

    for name, cls in classes:
        if "Conv" in name and hasattr(cls, "forward"):
            forward_sig = inspect.signature(cls.forward)
            forward_params = list(forward_sig.parameters.values())
            if (
                len(forward_params) >= 3
                and forward_params[0].name == "self"
                and forward_params[1].name == "x"
                and forward_params[2].name == "edge_index"
            ):
                if (
                    len(forward_params) >= 4
                    and forward_params[3].default != inspect.Parameter.empty
                ):
                    print(name, forward_params)
                    print()
                    filtered_classes.append(name)

    print(filtered_classes)
