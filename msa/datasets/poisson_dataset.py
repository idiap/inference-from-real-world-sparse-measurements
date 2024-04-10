# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Utilities to load the Poisson dataset."""
import shutil
from pathlib import Path

import hydra
import torch
from graph_element_networks.gen_datasets import FTDataset
from graph_element_networks.poisson_datasets import (
    PoissonSquareRoomInpDataset,
    PoissonSquareRoomOutDataset,
)
from torchvision.datasets.utils import download_and_extract_archive

from .utils import StackDataset


def datasets(cfg, device, train_data_frac=0.8):
    """Load the Poisson dataset.

    Args:
        cfg (omegaconf.DictConfig): The configuration.
        device (torch.device): The device to load the data on.
        train_data_frac (float): The fraction of the data to use for training.

    Returns:
        train_dataset (torch.utils.data.Dataset): The training dataset.
        val_dataset (torch.utils.data.Dataset): The validation dataset.
    """
    data_dir = Path(cfg.dataset.path)

    if not data_dir.exists():
        raise ValueError(
            """Please set config/datasets/poisson path to the data folder.
            You can get the data by running
            `python -m msa.datasets.poisson_dataset dataset=poisson dataset.create=download`
            or
            You can download the data from
            https://github.com/FerranAlet/graph_element_networks/tree/master/data
            """
        )

    full_dataset = FTDataset(
        inp_datasets=[PoissonSquareRoomInpDataset],
        inp_datasets_args=[{"dir_path": data_dir / "poisson_inp"}],
        out_datasets=[PoissonSquareRoomOutDataset],
        out_datasets_args=[{"file_path": data_dir / "poisson_out.hdf5"}],
    )
    num_rows = len(full_dataset)
    train_size = int(train_data_frac * num_rows)
    test_size = num_rows - train_size
    train_raw, val_raw = torch.utils.data.random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_xc = torch.cat([xc for (((xc, _),), _), _ in train_raw]).to(device)
    train_yc = torch.cat([yc for (((_, yc),), _), _ in train_raw]).to(device)
    train_xt = torch.cat([xt for (_, ((xt, _),)), _ in train_raw]).to(device)
    train_yt = torch.cat([yt for (_, ((_, yt),)), _ in train_raw]).to(device)

    train_inputs = StackDataset(train_xc, train_yc, train_xt)
    train_dataset = StackDataset(train_inputs, train_yt)

    val_xc = torch.cat([xc for (((xc, _),), _), _ in val_raw]).to(device)
    val_yc = torch.cat([yc for (((_, yc),), _), _ in val_raw]).to(device)
    val_xt = torch.cat([xt for (_, ((xt, _),)), _ in val_raw]).to(device)
    val_yt = torch.cat([yt for (_, ((_, yt),)), _ in val_raw]).to(device)

    val_inputs = StackDataset(val_xc, val_yc, val_xt)
    val_dataset = StackDataset(val_inputs, val_yt)

    return train_dataset, val_dataset


@hydra.main(config_path="../../configs", config_name="trainer", version_base="1.2")
def main(cfg):
    """Download the Poisson dataset."""
    print(cfg.dataset.path)

    if cfg.dataset.create == "download":
        out = Path(cfg.dataset.path)
        print(f"[web] -> {out}")

        print(out.absolute())

        if out.exists():
            print("Data already downloaded. Exiting...")
            return

        download_and_extract_archive(
            "https://github.com/FerranAlet/graph_element_networks/archive/refs/heads/master.zip",
            download_root=out,
            remove_finished=True,
        )
        # extract * from graph_element_networks-master/data to .

        shutil.move(out / "graph_element_networks-master/data/poisson_inp", out)
        shutil.move(out / "graph_element_networks-master/data/poisson_out.hdf5", out)

        shutil.rmtree(out / "graph_element_networks-master")

    else:
        print(
            """Download the data first by running
            `python -m msa.datasets.poisson_dataset dataset=poisson dataset.create=download`
            """
        )
        print("Download not set to download. Exiting...")


if __name__ == "__main__":
    main()
