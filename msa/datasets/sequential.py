# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Sequential datasets."""
import torch
import torch.utils.data as data
from more_itertools import pairwise


class SequentialDataset(data.Dataset):
    """Split a dataset into time slices."""

    def __init__(self, dataset, time_interval):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
            time_interval (float): Time interval between two slices.
        """
        inputs, _ = dataset[:]
        self.time_interval = time_interval

        time = inputs[:, 3].clone()
        mint, maxt = time.min(), time.max()
        intervals = int((maxt - mint) / self.time_interval)

        splits = torch.tensor(
            [time[0] + self.time_interval * i for i in range(intervals + 1)]
        )

        splits_index = torch.searchsorted(time, splits)
        splits_index = splits_index.tolist() + [-1]
        self.slices = [slice(a, b) for a, b in pairwise(splits_index) if a != b]
        self.sequences = tuple(dataset[sl] for sl in self.slices)

        assert all(len(s[0]) for s in self.sequences)

        assert all(
            (s[0][-1, 3] - s[0][0, 3]) <= self.time_interval + 5e-4
            for s in self.sequences
        )

    def __getitem__(self, index):
        """Get a time slice.

        Args:
            index (int): Index of the time slice.
        """
        return self.sequences[index]

    def __len__(self):
        """Get the number of time slices.

        Returns:
            int: Number of time slices.
        """
        return len(self.sequences)


class CombineSequentialDataset(data.Dataset):
    """Combine several sequential datasets."""

    def __init__(self, datasets):
        """Initialize the dataset.

        Args:
            datasets (list): List of sequential datasets.
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.bins = torch.tensor(self.lengths).cumsum(0)

    def __getitem__(self, index):
        """Get a the time slice in one of the datasets.

        Args:
            index (int): Index of the time slice.

        Returns:
            tuple: Inputs and targets of the time slice.
        """
        j = torch.searchsorted(self.bins, index + 1)
        i = index if j == 0 else index - self.bins[j - 1]

        return self.datasets[j][i]

    def __len__(self):
        """Get the number of time slices.

        Returns:
            int: Number of time slices.
        """
        return sum(self.lengths)


class PercentageDataset(data.Dataset):
    """Take a percentage of a dataset."""

    def __init__(self, dataset, percentage):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
            percentage (float): Percentage of the dataset to take.
        """
        self.dataset = dataset
        max_i = int(percentage * len(dataset))
        self.idxs = torch.randperm(len(dataset))[:max_i]

    def __getitem__(self, index):
        """Get an item.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Inputs and targets of the item.
        """
        return self.dataset[self.idxs[index]]

    def __len__(self):
        """Get the number of items.

        Returns:
            int: Number of items.
        """
        return len(self.idxs)


class DecimateContextDataset(data.Dataset):
    """Take one measure of of d in context of a dataset."""

    def __init__(self, dataset, decimation):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
            decimation (int): Decimation factor.
        """
        self.dataset = dataset
        self.decimation = decimation

    def __getitem__(self, index):
        """Get an item.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Inputs and targets of the item.
        """
        (xc, yc, xt), yt = self.dataset[index]
        return (xc[:: self.decimation], yc[:: self.decimation], xt), yt

    def __len__(self):
        """Get the number of items.

        Returns:
            int: Number of items.
        """
        return len(self.dataset)


def ContextSeqDataset(dataset, time_interval, time_window):
    """Combine a time slice with another with an interval of time_window.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to split.
        time_interval (float): Time interval between two slices.
        time_window (float): Time window of the context.
    """
    assert time_window % time_interval == 0
    seq_dataset = SequentialDataset(dataset, time_interval)
    offset = time_window // time_interval + 1
    return list(zip(seq_dataset[:-offset], seq_dataset[offset:]))


class AdaptDataset(data.Dataset):
    """Adapt a dataset ((xc,yc), (xtyt) to ((xc,yc,xt),yt."""

    def __init__(self, dataset):
        """Initialize the dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to split.
        """
        self.dataset = dataset

    def __getitem__(self, i):
        """Get an item.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Inputs and targets of the item.
        """
        (xc, yc), (xt, yt) = self.dataset[i]
        return (xc[:, :3], yc, xt[:, :3]), yt

    def __len__(self):
        """Get the number of items.

        Returns:
            int: Number of items.
        """
        return len(self.dataset)
