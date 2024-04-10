# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Code for the gradient experiment of the inference from real-world data.

This script is used to generate the figures of the gradient experiment of the
extrinsic transformer paper. It is used to compare the gradients of the
transformer and the GEN on the baseline datasets.

Train first:
    python train.py dataset=sin model=ablations/encoder_only/5000 freq=10

How to run:
    python scripts/gradient.py --path <path_to_trained_baselines>

    example:
    python revert.py --path outputs/encoder_only-5000-0
"""
import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

from msa.datasets.baseline_dataset import datasets
from msa.models.gen import GEN, GraphStructure, grid
from msa.models.msa import MSA, MSAEncoderOnly
from msa.models.transformer import TFS, EncoderOnly

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

load_dotenv()
log = logging.getLogger(__name__)


def patched_tfse_forward(self, xc, yc, xt):
    """Replace the forward pass of the transformer to see the latents.

    Args:
        xc (torch.Tensor): Context points
        yc (torch.Tensor): Context values
        xt (torch.Tensor): Target points

    Returns:
        torch.Tensor: Target values
    """
    latents = self.encoder(torch.cat((xc, yc), dim=-1))

    for block in self.blocks:
        latents = block(torch.cat((xc, latents), dim=-1))

    self.latents = latents
    self.latents.retain_grad()

    self.scores = self.scaled_dist(xt, xc)
    self.scores.retain_grad()
    z = self.scores.bmm(latents)
    return self.decoder(torch.cat((z, xt), dim=-1))


def patched_tfs_forward(self, xc, yc, xt):
    """Replace the forward pass of the transformer to see the latents.

    Args:
        xc (torch.Tensor): Context points
        yc (torch.Tensor): Context values
        xt (torch.Tensor): Target points

    Returns:
        torch.Tensor: Target values
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
    self.latents = latents
    self.latents.retain_grad()
    z = self.transformer_decoder(q, latents)
    return self.decoder(z)


def patched_msae_forward(self, xc, yc, xt):
    """Replace the forward pass of the MSA to see the latents.

    Args:
        xc (torch.Tensor): Context points
        yc (torch.Tensor): Context values
        xt (torch.Tensor): Target points

    Returns:
        torch.Tensor: Target values
    """
    latents_context = self.context_encoder(torch.cat((xc, yc), dim=-1))
    latents_target = self.target_encoder(xt)
    latents = torch.cat((latents_context, latents_target), dim=1)
    L = xt.shape[1]

    for block in self.blocks:
        latents = block(latents)
    self.latents = latents
    self.latents.retain_grad()

    return self.decoder(latents[:, -L:, :])


def patched_msa_forward(self, xc, yc, xt):
    """Replace the forward pass of the MSA to see the latents.

    Args:
        xc (torch.Tensor): Context points
        yc (torch.Tensor): Context values
        xt (torch.Tensor): Target points

    Returns:
        torch.Tensor: Target values
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
    self.latents = z
    self.latents.retain_grad()
    L = xt.shape[1]
    z = z[:, -L:, :]
    return self.decoder(z)


def patched_gen_forward(self, xc, yc, xt):
    """Replace the forward pass of the GEN to see the latents.

    Args:
        xc (torch.Tensor): Context points
        yc (torch.Tensor): Context values
        xt (torch.Tensor): Target points

    Returns:
        torch.Tensor: Target values
    """
    p = self.g.pos.unsqueeze(0).repeat(len(xc), 1, 1)
    scores = self.g(xc)
    emb = self.encoder(torch.cat((xc, yc), dim=-1))
    latents = scores.transpose(1, 2).bmm(emb)
    for block in self.gn_blocks:
        latents = block(
            torch.cat((p, latents), dim=-1), self.g.senders, self.g.receivers
        )
    self.latents = latents
    self.latents.retain_grad()

    self.scores = self.g(xt)
    self.scores.retain_grad()
    z = self.scores.bmm(self.latents)
    return self.decoder(torch.cat((z, xt), dim=-1))


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="outputs")
parser.add_argument("--use_random_xt", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    pts = list(Path(args.path).glob("**/*.pt"))
    config_files = [p.parent / ".hydra/config.yaml" for p in pts]

    cfgs = []
    for c in config_files:
        with c.open() as f:
            cfgs.append(yaml.safe_load(f))

    models = []
    for cfg in cfgs:
        match cfg:
            case {"model": {"name": "gen"}}:
                gs = GraphStructure(*grid(cfg["model"]["grid_size"]), fixed=False)
                model = GEN(gs, **cfg["model"]["params"])
                model.forward = patched_gen_forward.__get__(model)
                models.append(model)
            case {"model": {"name": "encoder_only"}}:
                model = EncoderOnly(**cfg["model"]["params"])
                model.forward = patched_tfse_forward.__get__(model)
                models.append(model)
            case {"model": {"name": "tfs"}}:
                model = TFS(**cfg["model"]["params"])
                model.forward = patched_tfs_forward.__get__(model)
                models.append(model)
            case {"model": {"name": "msa_encoder_only"}}:
                model = MSAEncoderOnly(**cfg["model"]["params"])
                model.forward = patched_msae_forward.__get__(model)
                models.append(model)
            case {"model": {"name": "msa"}}:
                model = MSA(**cfg["model"]["params"])
                model.forward = patched_msa_forward.__get__(model)
                models.append(model)
            case _:
                models.append(None)

    pts = [p for p, m in zip(pts, models) if m is not None]
    cfgs = [c for c, m in zip(cfgs, models) if m is not None]
    models = [m for m in models if m is not None]
    print("Loading models")
    for m, p in zip(models, pts):
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()

    names = []
    for cfg in cfgs:
        match cfg:
            case {"model": {"name": "gen", "grid_size": g}}:
                names.append(f"GEN {g}x{g}")
            case {"model": {"name": "encoder_only"}}:
                names.append("TFS")
            # case {"model": {"name": "tfs"}}:
            #     names.append("TFS")
            # case {"model": {"name": "msa"}}:
            #     names.append("ZMSA")

            case {"model": {"name": "msa_encoder_only"}}:
                names.append("ZMSA")

    names, models = zip(*sorted(zip(names, models), key=lambda t: t[0], reverse=True))

    train_dataset, val_dataset = datasets("cpu", 64, "sin", 10)

    dl = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=True)
    (xc, yc, xt), yt = next(iter(dl))
    if args.use_random_xt:
        xt = torch.randn(2).repeat(len(xc), 1).unsqueeze(1)
        print(torch.argmin(torch.norm(xc - xt, dim=-1), dim=-1)[0])

    n = 0

    print("Computing gradients")
    for model in models:
        output = model(xc, yc, xt)
        o2 = output.detach().clone()
        o2[:, n, :] += 200.0
        loss = F.mse_loss(output, o2)
        print(loss.item())
        loss.backward()

    print("Plotting")
    fig = plt.figure(figsize=(4.5, 5))
    increment = len([n for n in names if n.startswith("Z")])

    gs = fig.add_gridspec(len(models) + increment + 1, 64, left=0.18)

    if args.use_random_xt:
        axs = [fig.add_subplot(gs[0, :1])]
        axs[0].imshow(
            F.mse_loss(output, o2, reduction="none")
            .sum(0, keepdims=True)
            .detach()
            .numpy(),
            cmap="gray",
            vmin=10000.0,
            vmax=10000.0,
        )
        axs[0].set_xticks([1])
    else:
        axs = [fig.add_subplot(gs[0, :])]

        axs[0].imshow(
            F.mse_loss(output, o2, reduction="none")
            .sum(0, keepdims=True)
            .detach()
            .numpy(),
            cmap="binary",
        )
        axs[0].set_xticks([0, 64])

    axs[0].set_yticks([0.0], ["Error"])
    i = 0
    for model, name_with_z in zip(models, names):
        name = name_with_z.replace("Z", "")

        grads = model.latents.grad[0].abs().sum(1).unsqueeze(0).detach().numpy()

        if name.startswith("MSA"):
            N = 64
            ax = fig.add_subplot(gs[i + 1, :N])
            ax.imshow(grads[:, :N], cmap="binary", vmin=0.0, vmax=1e-8)
            ax.set_yticks([0.0], [name + " cxt"])
            if N == 1:
                ax.set_xticks([1])
            else:
                ax.set_xticks([0, N])

            i += 1
            axs.append(ax)

            N = 1 if args.use_random_xt else 64
            cmap = "gray" if args.use_random_xt else "binary"

            ax = fig.add_subplot(gs[i + 1, :N])
            ax.imshow(
                grads[:, -N:],
                cmap=cmap,
                vmin=10000.0,
                vmax=10000.0,
            )
            ax.set_yticks([0.0], [name + " tgt"])
        else:
            N = grads.shape[1]
            ax = fig.add_subplot(gs[i + 1, :N])

            ax.imshow(grads, cmap="binary", vmin=0.0, vmax=1e-8)
            ax.set_yticks([0.0], [name])

        if N == 1:
            ax.set_xticks([1])
        else:
            ax.set_xticks([0, N])
        axs.append(ax)

        i += 1

    for ax in axs:
        ax.set_aspect("equal")
        for loc in ["top", "left", "right"]:
            ax.spines[loc].set_visible(False)

    rnxt_str = "_randxt" if args.use_random_xt else ""

    fig.savefig(f"grads{n}{rnxt_str}.pdf")
