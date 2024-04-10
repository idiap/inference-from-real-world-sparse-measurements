# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Wind Speed Dataset."""
from collections import namedtuple

import pandas as pd
import torch
import torch.utils.data as data

from .windprocessing import process_df, select_day

Weeks = namedtuple("Weeks", ["files"])
Week = namedtuple("Week", ["file"])
Day = namedtuple("Day", ["file", "day"])


class BaseDataset(data.Dataset):
    """Base Dataset."""

    def __init__(self, folder, subset):
        """Initialize the dataset.

        Args:
            folder (str): path to the folder containing the data
            subset (Weeks, Week or Day): subset of the data to use
        """
        files = sorted(folder.glob("*.csv"))

        print("Starting Processing.")

        if isinstance(subset, Weeks):
            files = [files[i] for i in subset.files]
            data_list = []
            self.fid2indices = []
            for f in files:
                print("  - Processing file : ", str(f))
                df = pd.read_csv(f)
                data_in_file, fid = process_df(df)
                data_list.append(data_in_file)
                self.fid2indices.append(fid)

            self.count = [len(d) for d in data_list]

            data = torch.cat(data_list)
        else:
            df = pd.read_csv(files[subset.file])
            if isinstance(subset, Week):
                data, self.fid2indices = process_df(df)
            else:
                df = select_day(df, subset.day)
                data, self.fid2indices = process_df(df)

        self.inputs, self.targets = data[:, :4], data[:, 4:]

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: length of the dataset
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """Return the item at the given index.

        Args:
            index (int): index of the item to return

        Returns:
            tuple: position, values
        """
        return self.inputs[index], self.targets[index]
