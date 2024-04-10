# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Experiment runner."""
import logging
import time
from functools import partial

import torch
from torch.nn.utils import clip_grad_norm_

log = logging.getLogger(__name__)


def validate(model, val_dl, criterion, device):
    """Validate the model.

    Args:
        model (torch.nn.Module): model to validate
        val_dl (torch.utils.data.DataLoader): validation dataloader
        criterion (torch.nn.Module): loss function
        device (torch.device): device to use

    Returns:
        float: validation loss
    """
    to_device = partial(to, device=device)
    nb_val_samples, acc_val_loss = 0, 0.0
    model.eval()
    for inputs, targets in val_dl:
        targets = targets.to(device)
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = map(to_device, inputs)
            output = model(*inputs)
        else:
            inputs = inputs.to(device)
            output = model(inputs)
        loss = criterion(output, targets)
        acc_val_loss += loss.item() * len(targets)
        nb_val_samples += len(targets)

    model.train()
    return acc_val_loss / nb_val_samples


def run_exp(model, train_dl, val_dl, cfg, criterion=None, optimizer=None):
    """Run an experiment.

    Args:
        model (torch.nn.Module): model to train
        train_dl (torch.utils.data.DataLoader): train data loader
        val_dl (torch.utils.data.DataLoader): validation data loader
        cfg (dict): configuration
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed(cfg.experiment.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    to_device = partial(to, device=device)

    log.info(cfg)
    log.info(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(device)
    criterion.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.experiment.lr,
            weight_decay=cfg.experiment.wd,
        )

    total_time = 0.0
    num_epochs = cfg.experiment.nb_epochs

    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed(cfg.experiment.seed)
    for epoch in range(num_epochs):
        tic = time.perf_counter()
        nb_train_samples, acc_train_loss = 0, 0.0

        model.train()
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(train_dl, 1):
            targets = targets.to(device)
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                inputs = map(to_device, inputs)
                output = model(*inputs)
            else:
                inputs = inputs.to(device)
                output = model(inputs)
            loss = criterion(output, targets)

            loss.backward()
            acc_train_loss += loss.item() * len(targets)
            nb_train_samples += len(targets)
            train_loss = acc_train_loss / nb_train_samples

            clip_grad_norm_(model.parameters(), 4.0)

            log.debug(
                "    train: %03d / %04d, %03d / %04d %.4f",
                epoch,
                num_epochs,
                i,
                len(train_dl),
                loss.item(),
            )
            optimizer.step()
            optimizer.zero_grad()

            if "steps" in cfg.experiment and i % cfg.experiment.steps == 0:
                val_loss = validate(model, val_dl, criterion, device)
                log.info("Step %05d : train, val = %4f %.4f", i, train_loss, val_loss)

        val_loss = validate(model, val_dl, criterion, device)
        tac = time.perf_counter()
        total_time += tac - tic

        log.info(
            "Epoch %03d / %03d: train, val, duration = %.6f %.6f %.4f",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            total_time / (epoch + 1),
        )

        if "freq" not in cfg.experiment or (epoch + 1) % cfg.experiment.freq == 0:
            torch.save(model.state_dict(), f"{cfg.name}-{epoch}-{val_loss:.4f}.pt")


def to(x, device):
    """Move object to device.

    Args:
        x (object): object to move
        device (torch.device): device to move to

    Returns:
        object: moved object
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to(y, device) for y in x]
    elif isinstance(x, tuple):
        return tuple(to(y, device) for y in x)
    else:
        return x
