# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Experiment on error correction of the TFS/GEN models.

How to run:
    python error_bottleneck.py --path_tfs path/to/tfs.pt --path_gen path/to/gen.pt --path_msa path/to/msa.pt

parameters are optional, default values should be given in the env file.
"""
import argparse
import os
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.data as data
from hydra import compose, initialize
from tqdm import tqdm

from msa.datasets.baseline_dataset import datasets
from msa.experiment import to
from msa.models.gen import GEN, GraphStructure, grid
from msa.models.msa import MSAEncoderOnly
from msa.models.transformer import EncoderOnly

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


def load_models(path_msa, path_tfs, path_gen):
    """Load the models from the given paths.

    Args:
        path_tfs (str): Path to the transformer model
        path_gen (str): Path to the GEN model
        path_msa (str): Path to the MSA model

    Returns:
        tuple: (MSA, Transformer, GEN, config)
    """
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="trainer",
            overrides=[
                "dataset=sin",
                "dataset.f=10",
                "model=ablations/msa_encoder_only/5000",
            ],
        )
        print(cfg)
        msa = MSAEncoderOnly(**cfg.model.params)
        msa.load_state_dict(torch.load(path_msa, map_location="cpu"))
        msa.eval()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="trainer",
            overrides=[
                "dataset=sin",
                "dataset.f=10",
                "model=ablations/encoder_only/5000",
            ],
        )
        print(cfg)
        tfs = EncoderOnly(**cfg.model.params)
        tfs.load_state_dict(torch.load(path_tfs, map_location="cpu"))
        tfs.eval()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="trainer",
            overrides=["dataset=sin", "dataset.f=10", "model=baselines/gen/5000"],
        )
        print(cfg)
        gs = GraphStructure(*grid(cfg.model.grid_size), fixed=False)
        gen = GEN(gs, **cfg.model.params)
        gen.load_state_dict(torch.load(path_gen, map_location="cpu"))
        gen.eval()

    return msa, tfs, gen, cfg


def apply_artificial_loss(output, n=1):
    """Apply an artificial loss to the output of the model.

    The target is the same as the output, except that the value of the
    n-th point is increased by 10.

    Args:
        output (torch.Tensor): Output of the model
        n (int): Index of the point to modify

    Returns:
        tuple: (loss, output)
    """
    o2 = output.detach().clone()
    o2[:, n, :] += 10.0
    loss = F.mse_loss(output, o2)
    print(loss)
    loss.backward()
    return loss, o2


parser = argparse.ArgumentParser()
parser.add_argument("--path_msa", type=str)
parser.add_argument("--path_tfs", type=str)
parser.add_argument("--path_gen", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    msa, tfs, gen, cfg = load_models(args.path_msa, args.path_tfs, args.path_gen)

    train_dataset, val_dataset = datasets(
        "cpu", cfg.dataset.n_x, cfg.dataset.type, cfg.dataset.f
    )

    (xc, yc, xt), yt = train_dataset[0]
    xc, yc, xt, yt = xc.unsqueeze(0), yc.unsqueeze(0), xt.unsqueeze(0), yt.unsqueeze(0)

    output = tfs(xc, yc, xt)
    loss, _ = apply_artificial_loss(output)

    output = gen(xc, yc, xt)
    loss, o2 = apply_artificial_loss(output)

    results = []
    for model in [msa, tfs, gen]:
        bs = cfg.experiment.batch_size
        train_dl = data.DataLoader(train_dataset, batch_size=bs, shuffle=False)
        val_dl = data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

        model.eval()
        with torch.no_grad():
            train_targets = [
                model(*inputs).detach().clone() for inputs, _ in tqdm(train_dl)
            ]
            val_targets = [
                model(*inputs).detach().clone() for inputs, _ in tqdm(val_dl)
            ]

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.wd,
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

        results.append(k)
        print(k)
        print()
        print()
