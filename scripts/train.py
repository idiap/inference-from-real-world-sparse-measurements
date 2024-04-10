# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Train a model on the selected dataset.

How to run:
    python train.py dataset=wind model=tfs/100000

How to override the config:
    python train.py dataset=poisson model=gen/100000
"""
import logging
import os

import hydra
import torch
import torch.nn as nn
import torch.utils.data as data

from msa.datasets import (
    baseline_datasets,
    darcy_datasets,
    era5_datasets,
    navierstokes_datasets,
    poisson_datasets,
    wind_datasets,
)
from msa.datasets.sequential import DecimateContextDataset
from msa.experiment import run_exp
from msa.models import default_models, gen_models, geofno_models
from msa.models.gen import GraphStructure, grid, kmeans_from_dataset, neighbors_edges

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="trainer", version_base="1.2")
def main(cfg):
    """Training pipeline for all models and all baselines."""
    optimizer = None

    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.experiment.seed)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    model_name = cfg.model.name
    log.info("model_name = %s", model_name)
    metric = nn.MSELoss()

    if cfg.dataset.type == "poisson":
        train_dataset, val_dataset = poisson_datasets(cfg, device)
    elif cfg.dataset.type == "wind":
        train_dataset, val_dataset, metric = wind_datasets(cfg)
    elif cfg.dataset.type == "navierstokes":
        train_dataset, val_dataset = navierstokes_datasets()
    elif cfg.dataset.type == "era5":
        train_dataset, val_dataset, metric = era5_datasets()
    elif cfg.dataset.type == "darcy":
        train_dataset, val_dataset, metric = darcy_datasets()
    else:
        train_dataset, val_dataset = baseline_datasets(
            device, cfg.dataset.n_x, cfg.dataset.type, cfg.dataset.f
        )

    if "decimation_context" in cfg.dataset:
        d = cfg.dataset.decimation_context
        train_dataset = DecimateContextDataset(train_dataset, d)
        val_dataset = DecimateContextDataset(val_dataset, d)

    if model_name in default_models:
        model = default_models[model_name](**cfg.model.params)
    elif model_name in gen_models:
        if cfg.dataset.type == "wind":
            pos = kmeans_from_dataset(k=1000, dataset=train_dataset)
            graph = (pos, *neighbors_edges(pos, 3))
        else:
            graph = grid(cfg.model.grid_size)

            if cfg.dataset.type in ("navierstokes", "era5"):
                pos, s, r = graph
                graph = (pos * 2 - 1, s, r)

        gs = GraphStructure(*graph, fixed=False)
        model = gen_models[model_name](gs, **cfg.model.params)
        optimizer = torch.optim.Adam(
            model.optim_groups(cfg.model.pos_lr),
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.wd,
        )
    elif model_name.startswith("geofno"):
        model = geofno_models[model_name](**cfg.model.params)
        optimizer = torch.optim.Adam(
            model.optim_groups(cfg.model.iphi_lr),
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.wd,
        )

    log.debug("len(train_dataset_raw) = %d", len(train_dataset))
    log.debug("len(val_dataset_raw) = %d", len(val_dataset))

    g = torch.Generator().manual_seed(cfg.experiment.seed)

    bs = cfg.experiment.batch_size
    train_dl = data.DataLoader(train_dataset, batch_size=bs, shuffle=True, generator=g)
    val_dl = data.DataLoader(val_dataset, batch_size=bs, shuffle=True, generator=g)

    run_exp(model, train_dl, val_dl, cfg, metric, optimizer)


if __name__ == "__main__":
    main()
