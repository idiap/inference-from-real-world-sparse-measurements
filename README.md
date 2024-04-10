<!--
Copyright © <2023> Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>

SPDX-License-Identifier: LGPL-3.0-only
-->


# Inference from Real-World Sparse Measurements - MALAT

This code was developed as a part of the Innosuisse MALAT: Machine Learning for Air Traffic project, which is a partnership between SkySoft ATM and the Idiap Research Institute.

Main research partner: Pr. François Fleuret (UNIGE)

Project manager : Didier Berling (SkySoft ATM)

Author: Arnaud Pannatier <arnaud.pannatier@idiap.ch> (Idiap Research Institute).

For any questions/remarks about this work or this research, feel free to contact the author.


## Project Overview
This project contains the implementation of the Multi-Layer Self-Attention, a state-of-the-art model designed for wind nowcasting tasks. The code in this repository corresponds to part T2.4 of the Innosuisse project (the second part of the wind nowcasting task) and is also related to the publication `Inference from Real-World Sparse Measurements` accepted at Transactions on Machine Learning Research (TMLR) available at: [https://openreview.net/forum?id=y9IDfODRns](https://openreview.net/forum?id=y9IDfODRns).

The Extrinsic Transformer is specifically tailored for wind nowcasting, focusing on the training and evaluation of the transformer model. In addition to the core model, this repository also includes code for training and evaluating the transformer, GEN (Graph Element Networks), CNP (Conditional Neural Processes), and various baseline models. These models are applied to different tasks, including wind nowcasting, Poisson equation task, and other baselines to compare and showcase the capabilities of the Extrinsic Transformer.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Installation

The `environment.yml` contains all the required dependencies.
You should be able to create a working environment by running this command in the main folder.

```
conda env create -f environment.yml
```

## Usage

```python
import torch
from msa.modes.msa import MSA

m = MSA(
    dim_x = 3,
    dim_yc = 2,
    dim_yt = 2,
    dim_h = 128,
    nhead = 8,
    nlayers = 4,
    use_same_pos_enc=True,
    dropout=0.0
)

xc,yc, xt = torch.randn(10,1000, 3), torch.randn(10, 1000, 2), torch.randn(10, 2000,3)

yt = m(xc,yc,xt) # (10, 2000, 2)
```

This repository is structured as follows:

- `msa`: This package includes all the necessary models and dataset processing components, which will be installed automatically when the requirements are met.
- `configs`: These are the Hydra configuration folders that contain all the essential parameters for running the experiments and the baseline models.
- `scripts`: This folder serves as the entry point for training the model and conducting the experiments. Each file within the folder provides more detailed information in its respective docstring.

Models can be used out of the box:



### Configuration files

This repository uses Hydra for configuration management. The configuration files are located in the `configs` folder.

### Training

To train the model, you can use the `train.py` script. The script will train the model using the configuration file specified in the `config` argument.

to train msa on the wind nowcasting task, you can run the following command in the `scripts` folder:

```bash
python train.py
```

You might need to download the data first.

Data can be automatically downloaded by running the following command in the `scripts` folder:

```bash
python -m msa.dataset.wind dataset.create=download
python -m msa.dataset.wind dataset.create=default
```
The data is available here: [https://www.idiap.ch/en/scientific-research/data/skysoft](https://www.idiap.ch/en/scientific-research/data/skysoft)

For the Poisson equation task, you can run the following command in the `scripts` folder:

```bash
python train.py dataset=poisson
```

you might need to download the data first:

```bash
python -m msa.dataset.poisson dataset=poisson dataset.create=download
```

## Minimal GPU specs
This repository contains rather small models, running without problem in a few hours on modest GPUs (such as a few GTX 1080Tis).

## License

This project is licensed under the terms of the LGPL-3.0-only license.
