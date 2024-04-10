# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Revert an error in the output experiment.

Train first:
    python train.py dataset=sin model=ablations/encoder_only/5000 freq=10

How to run:
    python scripts/revert.py --path <path_to_trained_baselines>

    example:
    python revert.py --path outputs/encoder_only-5000-0
"""
import argparse
import logging
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from torch.utils import data
from tqdm import tqdm

from msa.datasets.baseline_dataset import datasets
from msa.experiment import to
from msa.models.gen import GEN, GraphStructure, grid
from msa.models.msa import MSAEncoderOnly
from msa.models.transformer import EncoderOnly

# Ensure reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

load_dotenv()
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
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
                models.append(model)
            case {"model": {"name": "encoder_only"}}:
                model = EncoderOnly(**cfg["model"]["params"])
                models.append(model)
            case {"model": {"name": "msa_encoder_only"}}:
                model = MSAEncoderOnly(**cfg["model"]["params"])
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
            case {"model": {"name": "msa_encoder_only"}}:
                names.append("ZMSA")

    names, models, cfgs = zip(
        *sorted(zip(names, models, cfgs), key=lambda t: t[0], reverse=True)
    )

    train_dataset, val_dataset = datasets("cpu", 64, "sin", 10)

    train_dl = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dl = data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    results = []
    for name, model, cfg in zip(names, models, cfgs):
        print(name)
        with torch.no_grad():
            train_targets = [
                model(*inputs).detach().clone() for inputs, _ in tqdm(train_dl)
            ]
            val_targets = [
                model(*inputs).detach().clone() for inputs, _ in tqdm(val_dl)
            ]

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["experiment"]["lr"],
            weight_decay=cfg["experiment"]["wd"],
        )

        device = torch.device("cpu")
        to_device = partial(to, device=device)
        criterion = F.mse_loss

        losses = {"train": 0.0, "val": 0.0}
        nb_samples = {"train": 0, "val": 0}

        optimizer.zero_grad()

        k = 0
        for (inputs, _), targets in zip(train_dl, train_targets):
            k += 1
            targets = targets.to(device)
            targets[:, 1, :] += 5.0
            inputs = tuple(map(to_device, inputs))

            model.train()
            output = model(*inputs)
            loss = criterion(output, targets.clone())
            loss.backward()

            losses["train"] += loss.item() * len(targets)
            nb_samples["train"] += len(targets)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                for (inputs, t2), targets in zip(val_dl, val_targets):
                    targets = t2.to(device)
                    targets[:, 1, :] += 5.0
                    inputs = tuple(map(to_device, inputs))
                    model.eval()
                    output = model(*inputs)
                    loss = criterion(output, targets)

                    losses["val"] += loss.item() * len(targets)
                    nb_samples["val"] += len(targets)

                for phase in losses:
                    losses[phase] /= nb_samples[phase]
                print(losses)

            if losses["val"] < 0.01:
                break
        print(k)
        results.append(k)

    results = np.array(results)

    with open("revert.txt", "w") as f:
        f.write(str(results))

    results = results / results[0] * 100
    names = [
        "MSA",
        "TFS",
        "GEN 8x8",
        "GEN 7x7",
        "GEN 6x6",
        "GEN 5x5",
        "GEN 4x4",
        "GEN 3x3",
        "GEN 2x2",
        "GEN 1x1",
    ]

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.bar(names, results, color="#bdc3c7")

    # set the color of the first bar
    ax.patches[0].set_facecolor("C0")

    ax.set_yticklabels([f"{x:.0f}%" for x in ax.get_yticks()])

    for loc in ["top", "right"]:
        ax.spines[loc].set_visible(False)
    ax.set_xticklabels(names, rotation=45, ha="right", rotation_mode="anchor")

    ax.set_ylabel("Increase in \n # of Gradient Updates")

    fig.tight_layout()
    plt.savefig("revert.pdf")
