# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Load and preprocess the wind dataset."""
from pathlib import Path

import hydra
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

from .sequential import AdaptDataset, CombineSequentialDataset, ContextSeqDataset
from .utils import Scale, ScaleRMSE
from .windspeed import BaseDataset, Day, Week


class ScaleDataset(Dataset):
    """Scale a dataset by mean and std."""

    def __init__(self, dataset, meanx, stdx, meany, stdy):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
            meanx (float): Mean of the x coordinates.
            stdx (float): Standard deviation of the x coordinates.
            meany (float): Mean of the y coordinates.
            stdy (float): Standard deviation of the y coordinates.
        """
        self.dataset = dataset
        self.meanx = meanx
        self.stdx = stdx
        self.meany = meany
        self.stdy = stdy

    def __getitem__(self, i):
        """Get an item.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Normalized Inputs and targets of the item.
        """
        (xc, yc, xt), yt = self.dataset[i]
        xc = (xc - self.meanx) / self.stdx
        xt = (xt - self.meanx) / self.stdx
        yc = (yc - self.meany) / self.stdy
        yt = (yt - self.meany) / self.stdy
        return (xc, yc, xt), yt

    def __len__(self):
        """Get the number of items.

        Returns:
            int: Number of items.
        """
        return len(self.dataset)


def get_mean_std(dataset):
    """Get the mean and std of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to split.

    Returns:
        tuple: Mean and std of the dataset.
    """
    xc = torch.cat([x[0][0] for x in dataset])
    xt = torch.cat([x[0][2] for x in dataset])
    x = torch.cat([xc, xt])
    yc = torch.cat([x[0][1] for x in dataset])
    yt = torch.cat([x[1] for x in dataset])
    y = torch.cat((yc, yt))
    return x.mean(0), x.std(0), y.mean(0), y.std(0)


def datasets(cfg):
    """Load the wind dataset.

    Returns:
        tuple: Train dataset, validation dataset, metric.
    """
    path = Path(cfg.dataset.path)

    if not path.exists():
        raise ValueError(
            """ Data folder (that can be set in config/dataset/wind.yaml) does not exists
            You need to download the dataset first running this file with --dataset download"
            `python -m msa.datasets.wind_dataset dataset.create=download`
            `python -m msa.datasets.wind_dataset dataset.create=default`
            """
        )
    tw = cfg.dataset.tw
    if tw is None or tw == 30:
        train_path = path / "train.pt"
        val_path = path / "val.pt"
        train_dataset = torch.load(train_path)
        val_dataset = torch.load(val_path)
    else:
        train_dataset = torch.load(path / f"/train{tw}.pt")
        val_dataset = torch.load(path / f"/val{tw}.pt")

    meanx, stdx, meany, stdy = get_mean_std(train_dataset)
    train_dataset = ScaleDataset(train_dataset, meanx, stdx, meany, stdy)
    val_dataset = ScaleDataset(val_dataset, meanx, stdx, meany, stdy)
    metric = ScaleRMSE(Scale(meany, stdy))

    return train_dataset, val_dataset, metric


@hydra.main(config_path="../../configs", config_name="trainer", version_base="1.2")
def main(cfg):
    """Download and preprocess the wind dataset."""
    print(cfg.dataset.path)
    out = Path(cfg.dataset.path)
    csvs = Path(cfg.dataset.path) / "SkySoft_WindSpeed/"

    if cfg.dataset.create == "download":
        print(f"[web] -> {csvs}")
        mirror = "https://zenodo.org/record/5074237/files/"
        tarname = "SkySoft_WindSpeed.tar.gz"
        md5 = "87b5f8c8e7faad810b43af6d67a49428"

        print(csvs.absolute())
        if not csvs.exists():
            download_and_extract_archive(
                mirror + tarname,
                download_root=out,
                filename=tarname,
                md5=md5,
            )
    elif cfg.dataset.create == "default":
        print(f"{csvs.absolute()} -> .pts")
        val = BaseDataset(csvs, Day(3, 3))
        val = AdaptDataset(ContextSeqDataset(val, 1 * 60, 30 * 60))
        torch.save(val, out / "val.pt")

        train = [BaseDataset(csvs, Week(i)) for i in [0, 1, 2, 4]]
        train = [ContextSeqDataset(d, 1 * 60, 30 * 60) for d in train]
        train = CombineSequentialDataset(train)
        train = AdaptDataset(train)
        torch.save(train, out / "train.pt")
    else:
        print(f"{csvs.absolute()} -> [tw].pts")

        tws = [10, 20, 60, 120, 240, 360]
        for tw in tws:
            val = BaseDataset(csvs, Day(3, 3))
            val = AdaptDataset(ContextSeqDataset(val, 1 * 60, tw * 60))
            torch.save(val, out / f"val{tw}.pt")

            train = [BaseDataset(csvs, Week(i)) for i in [0, 1, 2, 4]]
            train = [ContextSeqDataset(d, 1 * 60, tw * 60) for d in train]
            train = CombineSequentialDataset(train)
            train = AdaptDataset(train)
            torch.save(train, out / f"train{tw}.pt")


if __name__ == "__main__":
    main()
