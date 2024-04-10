# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Organize the models."""

from .gen import GEN, GENPYG, GENcatpos, GENCross, GENDist, GENnopos
from .gen_nograph import GEN_nograph
from .geofno import GEOFNO, GEOFNO_3layers
from .gka import GKA
from .lstm import LSTM
from .msa import MSA, MSAEncoderOnly
from .nps import CNP, CNPCross, CNPDist
from .perceiver import Perceiver
from .transformer import (
    TFS,
    EncoderOnly,
    TFSCross,
    TFSDist,
    TFSfull,
    TFSfullpn,
    TFSOne,
    TFSpos,
)

default_models = {
    "encoder_only": EncoderOnly,
    "gen_nograph": GEN_nograph,
    "gka": GKA,
    "msa_encoder_only": MSAEncoderOnly,
    "msa": MSA,
    "lstm": LSTM,
    "np": CNP,
    "npcross": CNPCross,
    "npdist": CNPDist,
    "perceiver": Perceiver,
    "tfs": TFS,
    "tfscross": TFSCross,
    "tfsdist": TFSDist,
    "tfsfull": TFSfull,
    "tfsfullpn": TFSfullpn,
    "tfsnormalfour": TFS,
    "tfsnormalone": TFS,
    "tfsnormaltwo": TFS,
    "tfsone": TFSOne,
    "tfspos": TFSpos,
}

gen_models = {
    "gen": GEN,
    "gencatpos": GENcatpos,
    "gencross": GENCross,
    "gendist": GENDist,
    "gennopos": GENnopos,
    "gennormalfour": GEN,
    "gennormalone": GEN,
    "gennormaltwo": GEN,
    "genpyg": GENPYG,
}

geofno_models = {"geofno": GEOFNO, "geofno3lay": GEOFNO_3layers}
