#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import setuptools
from importlib.util import find_spec
import warnings
import platform

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "numpy<1.19.0",
    "ruamel.yaml",
    "colorcet",
    "pillow",
    "py-cpuinfo==5.0.0",
    "tqdm",
]

if "tegra" in platform.platform():
    warnings.warn(
        "Not installing the following packages:\nopencv-python\ntensorflow\npandas\ntables\nPlease follow instructions on github to install opencv and tensorflow. If you want to use the benchmark_videos function to save poses from a video, then please install pandas and tables (pip install pandas tables)"
    )
else:
    install_requires.append("opencv-python")
    install_requires.append("tensorflow")
    install_requires.append("pandas")
    install_requires.append("tables")

setuptools.setup(
    name="deeplabcut-live",
    version="1.0",
    author="A. & M. Mathis Labs",
    author_email="admin@deeplabcut.org",
    description="Class to load exported DeepLabCut networks and perform pose estimation on single frames (from a camera feed)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut-live",
    python_requires=">=3.5, <3.8",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    package_data={"dlclive": ["check_install/*"]},
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ),
    entry_points={
        "console_scripts": [
            "dlc-live-test=dlclive.check_install.check_install:main",
            "dlc-live-benchmark=dlclive.benchmark:main",
        ]
    },
)
