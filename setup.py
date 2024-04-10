# Copyright © <2023> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arnaud Pannatier <arnaud.pannatier@idiap.ch>
#
# SPDX-License-Identifier: LGPL-3.0-only
"""Setup for windnowcasting package."""

from setuptools import find_packages, setup

__author__ = "Arnaud Pannatier"
__copyright__ = "Copyright (c) 2023, Idiap Research Institute"
__license__ = "INNOSUISSE Project Agreement - 31933.1 IP-ICT MALAT"
__maintainer__ = "Arnaud Pannatier"
__email__ = "arnaud.pannatier@idiap.ch"
__url__ = "https://gitlab.idiap.ch/malat/inference-from-real-world-sparse-measurements"
__version__ = "1.0"
__description__ = (
    """Provide code for Inference from Real-World Sparse Measurements (TMLR)."""
)

if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()
    setup(
        name="msa",
        version=__version__,
        description=__description__,
        long_description=long_description,
        long_description_content_ytpe="text/x-markdown",
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        license=__license__,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
        ],
        packages=find_packages(include=["msa", "msa.*"]),
    )
