# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Graph Element Network and all the ablations."""
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils import data
from torch_geometric.data import Batch, Data

from .mlp import Net

# from .positional_encoding import RFF, PosEncCat
from .pyg_convs import name2conv


class GENPYG(nn.Module):
    """Graph Element Network using message passing scheme from pyg."""

    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        message_passing_steps,
        share_blocks=True,
        mod="GCNConv",
    ):
        """Initialize the model.

        Args:
            graph_structure (nn.Module): graph structure
            dim_x (int): dimension of the input
            dim_yc (int): dimension of the class labels
            dim_yt (int): dimension of the target labels
            dim_h (int): dimension of the hidden layers
            message_passing_steps (int): number of message passing steps
            share_blocks (bool): share the blocks
        """
        super().__init__()
        self.g = graph_structure

        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h + dim_x, dim_h, dim_h, dim_yt])

        m = name2conv[mod]

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [m(dim_h + dim_x, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [m(dim_h + dim_x, dim_h) for _ in range(message_passing_steps)]
            )

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context pos
            yc (torch.Tensor): context values
            xt (torch.Tensor): target pos
        """
        p = self.g.pos.unsqueeze(0).repeat(len(xc), 1, 1)
        scores = self.g(xc)
        emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)

        data = [Data(x=lat, edge_index=self.g.edge_index, pos=p) for lat in latents]
        for d in data:
            d.validate(raise_on_error=True)
        b = Batch.from_data_list(data)

        for block in self.gn_blocks:
            b.x = block(torch.cat((b.x, b.pos.squeeze()), dim=-1), b.edge_index)

        latents = b.x.reshape(len(xc), -1, latents.shape[-1])

        scores = self.g(xt)
        z = scores.bmm(latents)
        z = torch.cat((z, xt), dim=-1)
        return self.decoder(z)

    def optim_groups(self, lr):
        """Don't optimize the graph position with the same learning rate.

        Args:
            lr (float): learning rate
        """
        total_keys = dict(self.named_parameters()).keys()
        other_keys = {pn for pn in total_keys if pn != "g.pos"}

        inter = other_keys & {"g.pos"}
        union = other_keys | {"g.pos"}
        assert len(inter) == 0
        assert len(total_keys - union) == 0

        return [
            {"params": [self.g.pos], "lr": lr},
            {"params": [p for pn, p in self.named_parameters() if pn != "g.pos"]},
        ]


class GraphNetBlock(nn.Module):
    """Take care of message passing."""

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
        """
        messages = self.message(
            torch.cat((nodes[:, receivers], nodes[:, senders]), dim=-1)
        )
        messages = self.bn1(messages)
        inbox = torch.zeros(nodes.shape, device=nodes.device)
        inbox = inbox.index_add(1, receivers, messages)
        return self.bn2(self.node(torch.cat((nodes, inbox), dim=-1)))


class GEN(nn.Module):
    """Graph Element Network."""

    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        message_passing_steps,
        share_blocks=True,
        pos_enc_freq=100.0,
        use_rff=False,
        use_same_pos_enc=False,
        use_mlps=False,
    ):
        """Initialize the model.

        Args:
            graph_structure (nn.Module): graph structure
            dim_x (int): dimension of the input
            dim_yc (int): dimension of the class labels
            dim_yt (int): dimension of the target labels
            dim_h (int): dimension of the hidden layers
            message_passing_steps (int): number of message passing steps
            share_blocks (bool): share the blocks
        """
        super().__init__()
        self.g = graph_structure

        self.use_same_pos_enc = use_same_pos_enc
        if use_same_pos_enc:
            self.pos_encoder = Net([dim_x, dim_h, dim_h, dim_h])
            self.val_encoder = Net([dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net([2 * dim_h, dim_h, dim_yt])

        else:
            self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
            self.decoder = Net(dims=[dim_h + dim_x, dim_h, dim_h, dim_yt])

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h + dim_x, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(dim_h + dim_x, dim_h)
                    for _ in range(message_passing_steps)
                ]
            )

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context pos
            yc (torch.Tensor): context values
            xt (torch.Tensor): target pos
        """
        p = self.g.pos.unsqueeze(0).repeat(len(xc), 1, 1)
        scores = self.g(xc)
        if self.use_same_pos_enc:
            ce_x = self.pos_encoder(xc)
            ce_y = self.val_encoder(yc)
            emb = ce_x + ce_y
        else:
            emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            latents = block(
                torch.cat((p, latents), dim=-1), self.g.senders, self.g.receivers
            )

        scores = self.g(xt)
        z = scores.bmm(latents)
        if self.use_same_pos_enc:
            q = self.pos_encoder(xt)
            z = torch.cat((z, q), dim=-1)
        else:
            z = torch.cat((z, xt), dim=-1)
        return self.decoder(z)

    def optim_groups(self, lr):
        """Don't optimize the graph position with the same learning rate.

        Args:
            lr (float): learning rate
        """
        total_keys = dict(self.named_parameters()).keys()
        other_keys = {pn for pn in total_keys if pn != "g.pos"}

        inter = other_keys & {"g.pos"}
        union = other_keys | {"g.pos"}
        assert len(inter) == 0
        assert len(total_keys - union) == 0

        return [
            {"params": [self.g.pos], "lr": lr},
            {"params": [p for pn, p in self.named_parameters() if pn != "g.pos"]},
        ]


class GENcatpos(nn.Module):
    """Graph Element Network with concatenated position."""

    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        message_passing_steps,
        share_blocks=True,
    ):
        """Initialize the model.

        Args:
            graph_structure (nn.Module): graph structure
            dim_x (int): dimension of the input
            dim_yc (int): dimension of the class labels
            dim_yt (int): dimension of the target labels
            dim_h (int): dimension of the hidden layers
            message_passing_steps (int): number of message passing steps
            share_blocks (bool): share the blocks
        """
        super().__init__()
        self.g = graph_structure

        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[2 * dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h + dim_x, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(dim_h + dim_x, dim_h)
                    for _ in range(message_passing_steps)
                ]
            )

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context pos
            yc (torch.Tensor): context values
            xt (torch.Tensor): target pos

        Returns:
            torch.Tensor: output
        """
        p = self.g.pos.unsqueeze(0).repeat(len(xc), 1, 1)
        scores = self.g(xc)
        emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            latents = block(
                torch.cat((p, latents), dim=-1), self.g.senders, self.g.receivers
            )

        q = self.q_encoder(xt)
        scores = self.g(xt)
        z = scores.bmm(latents)
        z = torch.cat((z, q), dim=-1)
        return self.decoder(z)

    def optim_groups(self, lr):
        """Don't optimize the graph position with the same learning rate.

        Args:
            lr (float): learning rate
        """
        total_keys = dict(self.named_parameters()).keys()
        other_keys = {pn for pn in total_keys if pn != "g.pos"}

        inter = other_keys & {"g.pos"}
        union = other_keys | {"g.pos"}
        assert len(inter) == 0
        assert len(total_keys - union) == 0

        return [
            {"params": [self.g.pos], "lr": lr},
            {"params": [p for pn, p in self.named_parameters() if pn != "g.pos"]},
        ]


class GENnopos(nn.Module):
    """Graph Element Network without position."""

    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        message_passing_steps,
        share_blocks=True,
    ):
        """Initialize the model.

        Args:
            graph_structure (nn.Module): graph structure
            dim_x (int): dimension of the input
            dim_yc (int): dimension of the class labels
            dim_yt (int): dimension of the target labels
            dim_h (int): dimension of the hidden layers
            message_passing_steps (int): number of message passing steps
            share_blocks (bool): share the blocks
        """
        super().__init__()
        self.g = graph_structure

        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[2 * dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h, dim_h) for _ in range(message_passing_steps)]
            )

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context pos
            yc (torch.Tensor): context values
            xt (torch.Tensor): target pos

        Returns:
            torch.Tensor: output
        """
        scores = self.g(xc)
        emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            latents = block(latents, self.g.senders, self.g.receivers)

        q = self.q_encoder(xt)
        scores = self.g(xt)
        z = scores.bmm(latents)
        z = torch.cat((z, q), dim=-1)
        return self.decoder(z)

    def optim_groups(self, lr):
        """Don't optimize the graph position with the same learning rate.

        Args:
            lr (float): learning rate
        """
        total_keys = dict(self.named_parameters()).keys()
        other_keys = {pn for pn in total_keys if pn != "g.pos"}

        inter = other_keys & {"g.pos"}
        union = other_keys | {"g.pos"}
        assert len(inter) == 0
        assert len(total_keys - union) == 0

        return [
            {"params": [self.g.pos], "lr": lr},
            {"params": [p for pn, p in self.named_parameters() if pn != "g.pos"]},
        ]


class GENDist(nn.Module):
    """Graph Element Network with distance."""

    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        message_passing_steps,
        share_blocks=True,
    ):
        """Initialize the model.

        Args:
            graph_structure (nn.Module): graph structure
            dim_x (int): dimension of the input
            dim_yc (int): dimension of the class labels
            dim_yt (int): dimension of the target labels
            dim_h (int): dimension of the hidden layers
            message_passing_steps (int): number of message passing steps
            share_blocks (bool): share the blocks
        """
        super().__init__()
        self.g = graph_structure

        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[2 * dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h + dim_x, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(dim_h + dim_x, dim_h)
                    for _ in range(message_passing_steps)
                ]
            )

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context pos
            yc (torch.Tensor): context values
            xt (torch.Tensor): target pos

        Returns:
            torch.Tensor: output
        """
        p = self.g.pos.unsqueeze(0).repeat(len(xc), 1, 1)
        scores = self.g(xc)
        emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            latents = block(
                torch.cat((p, latents), dim=-1), self.g.senders, self.g.receivers
            )

        q = self.q_encoder(xt)
        scores = self.g(xt)
        z = scores.bmm(latents)
        z = torch.cat((z, q), dim=-1)
        return self.decoder(z)

    def optim_groups(self, lr):
        """Don't optimize the graph position with the same learning rate.

        Args:
            lr (float): learning rate
        """
        total_keys = dict(self.named_parameters()).keys()
        other_keys = {pn for pn in total_keys if pn != "g.pos"}

        inter = other_keys & {"g.pos"}
        union = other_keys | {"g.pos"}
        assert len(inter) == 0
        assert len(total_keys - union) == 0

        return [
            {"params": [self.g.pos], "lr": lr},
            {"params": [p for pn, p in self.named_parameters() if pn != "g.pos"]},
        ]


class GENCross(nn.Module):
    """Graph Element Network with cross attention."""

    def __init__(
        self,
        graph_structure,
        dim_x,
        dim_yc,
        dim_yt,
        dim_h,
        nhead,
        message_passing_steps,
        share_blocks=True,
    ):
        """Initialize the model.

        Args:
            graph_structure (nn.Module): graph structure
            dim_x (int): dimension of the input
            dim_yc (int): dimension of the class labels
            dim_yt (int): dimension of the target labels
            dim_h (int): dimension of the hidden layers
            nhead (int): number of heads
            message_passing_steps (int): number of message passing steps
            share_blocks (bool): share the blocks
        """
        super().__init__()
        self.g = graph_structure

        self.encoder = Net(dims=[dim_x + dim_yc, dim_h, dim_h, dim_h])
        self.decoder = Net(dims=[dim_h, dim_h, dim_h, dim_yt])
        self.q_encoder = Net(dims=[dim_x, dim_h, dim_h, dim_h])

        if share_blocks:
            self.gn_blocks = nn.ModuleList(
                [GraphNetBlock(dim_h + dim_x, dim_h)] * message_passing_steps
            )
        else:
            self.gn_blocks = nn.ModuleList(
                [
                    GraphNetBlock(dim_h + dim_x, dim_h)
                    for _ in range(message_passing_steps)
                ]
            )

        self.cross = nn.MultiheadAttention(dim_h, nhead, batch_first=True)

    def forward(self, xc, yc, xt):
        """Forward pass.

        Args:
            xc (torch.Tensor): context pos
            yc (torch.Tensor): context values
            xt (torch.Tensor): target pos

        Returns:
            torch.Tensor: output
        """
        p = self.g.pos.unsqueeze(0).repeat(len(xc), 1, 1)
        scores = self.g(xc)
        emb = self.encoder(torch.cat((xc, yc), dim=-1))
        latents = scores.transpose(1, 2).bmm(emb)
        for block in self.gn_blocks:
            latents = block(
                torch.cat((p, latents), dim=-1), self.g.senders, self.g.receivers
            )

        q = self.q_encoder(xt)
        z = q + self.cross(q, latents, latents, need_weights=False)[0]
        return self.decoder(z)

    def optim_groups(self, lr):
        """Don't optimize the graph position with the same learning rate.

        Args:
            lr (float): learning rate
        """
        total_keys = dict(self.named_parameters()).keys()
        other_keys = {pn for pn in total_keys if pn != "g.pos"}

        inter = other_keys & {"g.pos"}
        union = other_keys | {"g.pos"}
        assert len(inter) == 0
        assert len(total_keys - union) == 0

        return [
            {"params": [self.g.pos], "lr": lr},
            {"params": [p for pn, p in self.named_parameters() if pn != "g.pos"]},
        ]


class GraphStructure(nn.Module):
    """Graph structure."""

    def __init__(self, pos, senders, receivers, fixed):
        """Initialize the graph structure.

        Args:
            pos (torch.Tensor): positions of the nodes
            senders (torch.Tensor): senders of the edges
            receivers (torch.Tensor): receivers of the edges
            fixed (bool): if True, the positions are fixed
        """
        super().__init__()

        if fixed:
            self.register_buffer("pos", pos)
        else:
            self.pos = nn.Parameter(pos)
        self.register_buffer("senders", senders)
        self.register_buffer("receivers", receivers)

        self.log_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Return the distance from x to the nodes.

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: distance
        """
        pseudo_dist = torch.norm(self.pos, dim=-1) ** 2 - 2 * x @ self.pos.t()
        return F.softmax(-self.log_strength.exp() * pseudo_dist, dim=-1)

    @property
    def edge_index(self):
        """Return the edge index.

        Returns:
            torch.Tensor: edge index
        """
        return torch.stack((self.senders, self.receivers), dim=0)


def grid(n):
    """Create a grid graph.

    Args:
        n (int): number of nodes per dimension

    Returns:
        torch.Tensor: positions
        torch.Tensor: senders
        torch.Tensor: receivers
    """
    x = torch.linspace(0, 1, n)
    I, J = torch.meshgrid(x, x, indexing="ij")
    pos = torch.cat((I.reshape(-1, 1), J.reshape(-1, 1)), dim=1)
    idx = torch.arange(n**2).view(n, n)

    senders = torch.cat(
        (
            idx[1:].flatten(),  # ↑
            idx[:, :-1].flatten(),  # →
            idx[:-1].flatten(),  # ↓
            idx[:, 1:].flatten(),  # ←
        )
    )
    receivers = torch.cat(
        (
            idx[:-1].flatten(),  # ↑
            idx[:, 1:].flatten(),  # →
            idx[1:].flatten(),  # ↓
            idx[:, :-1].flatten(),  # ←
        )
    )
    return pos, senders, receivers


def kmeans_from_dataset(dataset, k=1000, path=None):
    """Compute kmeans from a dataset.

    Args:
        dataset (torch.utils.data.Dataset): dataset
        k (int): number of clusters
        path (pathlib.Path): path to the cache

    Returns:
        torch.Tensor: positions
    """
    if path is None:
        path = Path(__file__).parent / "kmeans.pt"

    kdict = {}
    if path.exists():
        kdict = torch.load(path)
        if k in kdict:
            return kdict[k]

    choices = torch.randperm(len(dataset))[:300]
    sub_dataset = data.Subset(dataset, choices)
    inputs = torch.cat(
        [xc for (xc, _, _), _ in sub_dataset],
        dim=0,
    )

    print(inputs.shape)
    kmeans = KMeans(n_clusters=k, verbose=3).fit(inputs)
    pos = torch.tensor(kmeans.cluster_centers_).float()
    kdict[k] = pos
    torch.save(kdict, path)
    return pos


def neighbors_edges(pos, n=3):
    """Create a graph where each node is connected to its n nearest neighbors.

    Args:
        pos (torch.Tensor): positions
        n (int): number of neighbors

    Returns:
        torch.Tensor: senders
        torch.Tensor: receivers
    """
    dists = torch.norm(pos[None, :, :] - pos[:, None, :], dim=2)
    receivers = dists.argsort()[:, 1 : n + 1].flatten()
    senders = torch.arange(len(pos), device=pos.device).repeat_interleave(n)

    a = torch.cat((senders, receivers))
    b = torch.cat((receivers, senders))

    return a, b


def ba_edges(pos, m=3):
    """Create a graph using the Barabasi-Albert model.

    Args:
        pos (torch.Tensor): positions
        m (int): number of edges to attach from a new node to existing nodes

    Returns:
        torch.Tensor: senders
        torch.Tensor: receivers
    """
    g = nx.barabasi_albert_graph(len(pos), m)
    edge_index = torch.tensor(np.array(g.edges())).long()
    return edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()


def random_graph(p, k=1000):
    """Create a random graph.

    Args:
        p (float): probabiliyt of an edge
        k (int): number of nodes

    Returns:
        torch.Tensor: positions
        torch.Tensor: senders
        torch.Tensor: receivers
    """
    pos = torch.rand(k, 3) * 2 - 1
    edge_index = torch.full((k, k), p).bernoulli().nonzero()
    return pos, edge_index[:, 0].contiguous(), edge_index[:, 1].contiguous()
