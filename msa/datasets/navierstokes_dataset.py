# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""NavierStokes dataset."""
import os
from argparse import ArgumentParser
from pathlib import Path

import h5py
import pandas as pd
import torch
from dotenv import load_dotenv
from torchvision.datasets.utils import download_url

from msa.datasets.utils import StackDataset

from .utils import shapes

load_dotenv()


def h5toparticles(path):
    """Read a .h5 file and returns the particles as a numpy array.

    files are available here : https://github.com/pdebench/PDEBench 10Go each
    and contains the following keys :

    ['force', 'particles', 't', 'velocity']
    """
    print(f"Reading {path}")
    with h5py.File(path, "r") as f:
        particles = f["particles"][:]  # (4, 1000, 512, 512, 1)

    return particles


def particles2dataset(particles):
    """Convert a numpy array of particles to a tensors.

    Logic:
        - Start with a numpy array of particles (4, 1000, 512, 512, 1)
        - convert to tensors and squeeze
        - take pairs of images with 50 dt in between
        - sample 64, from context and 256 from target
    """
    particles = torch.from_numpy(particles).squeeze()
    x = torch.randint(0, 512, (4, 950, 64))
    y = torch.randint(0, 512, (4, 950, 64))
    a = torch.arange(4)[:, None, None].repeat(1, 950, 64)
    b = torch.arange(0, 950)[None, :, None].repeat(4, 1, 64)

    xc = torch.cat((x[..., None], y[..., None]), dim=3).div(256).sub(1).float()
    yc = particles[a, b, x, y]

    x = torch.randint(0, 512, (4, 950, 256))
    y = torch.randint(0, 512, (4, 950, 256))
    a = torch.arange(4)[:, None, None].repeat(1, 950, 256)
    b = torch.arange(50, 1000)[None, :, None].repeat(4, 1, 256)

    xt = torch.cat((x[..., None], y[..., None]), dim=3).div(256).sub(1).float()
    yt = particles[a, b, x, y]

    return (xc, yc, xt), yt


def download_and_process():
    """Download and process the navierstokes dataset.

    - read the csv files
    - get the corresponding url
    - download the file
    - process the file
    - save the file
    - delete source
    - repeat for all files
    """
    df = pd.read_csv(os.getenv("PDE_BENCH_CSV_PATH"))
    df = df[df["PDE"] == "NS_Incom"]
    print(f"Found {len(df)} files")

    already_downloaded = True

    for idx, (_, row) in enumerate(df.iterrows()):
        url = row["URL"]
        filename = row["Filename"]
        md5 = row["MD5"]
        path = Path() / filename

        if path.exists():
            print(f"File {filename} already exists, skipping...")
            already_downloaded = False
            continue

        if already_downloaded:
            continue

        if not path.exists():
            print(f"Downloading {url} {idx + 1}/{len(df)}")
            download_url(url, root=".", filename=filename, md5=md5)

        particles = h5toparticles(path)
        dataset = particles2dataset(particles)
        print(f"processed {filename}, res: {shapes(dataset)}, saving...")
        torch.save(dataset, filename.replace(".h5", ".pt"))

        os.remove(path)
        print()


def combine_files(files):
    """Combine data from a file list into a single dataset.

    Args:
        files (list): list of files to combine

    Returns:
        dataset (StackDataset): combined dataset
    """
    (xc, yc, xt), yt = torch.load(files[0])
    xc = xc.reshape(4 * 950, 64, 2)
    yc = yc.reshape(4 * 950, 64, 1)
    xt = xt.reshape(4 * 950, 256, 2)
    yt = yt.reshape(4 * 950, 256, 1)

    for f in files:
        (xc_, yc_, xt_), yt_ = torch.load(f)
        xc = torch.cat((xc, xc_.reshape(4 * 950, 64, 2)), dim=0)
        yc = torch.cat((yc, yc_.reshape(4 * 950, 64, 1)), dim=0)
        xt = torch.cat((xt, xt_.reshape(4 * 950, 256, 2)), dim=0)
        yt = torch.cat((yt, yt_.reshape(4 * 950, 256, 1)), dim=0)

    inputs = StackDataset(xc, yc, xt)
    dataset = StackDataset(inputs, yt)

    return dataset


def combine_dataset():
    """Combine the train and val dataset into a single dataset."""
    path = Path(os.getenv("NAVIER_STOKES_DATASET"))
    files = list(path.glob("*.pt"))

    files_trains = files[:200]
    files_val = files[200:]

    for files, name in zip([files_trains, files_val], ["train", "val"]):
        dataset = combine_files(files)
        torch.save(dataset, path / f"{name}.pt")


def datasets():
    """Return the train and val dataset."""
    path = Path(os.getenv("NAVIER_STOKES_DATASET"))
    train = torch.load(path / "train.pt")
    val = torch.load(path / "val.pt")

    return train, val


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_and_process()
    else:
        combine_dataset()
