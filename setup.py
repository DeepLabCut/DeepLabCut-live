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
    "numpy",
    "ruamel.yaml",
    "colorcet",
    "pillow",
    "py-cpuinfo==5.0.0",
    "tqdm",
]

if find_spec("cv2") is None:
    install_requires.append("opencv-python")
if find_spec("tensorflow") is None:
    warnings.warn(
        "tensorflow is not yet installed. Installing tensorflow CPU version. if you wish to use the GPU version, please run: pip install tensorflow-gpu==1.13.1"
    )
    install_requires.append("tensorflow==1.13.1")
if "tegra" in platform.platform():
    if find_spec("pandas") is None:
        warnings.warn(
            "Not installing pandas or pytables (these can take forever on NVIDIA Jetson boards). Please install these packages if you want to use the benchmark_videos function to save poses from a video."
        )
else:
    install_requires.append("pandas")
    install_requires.append("tables")

setuptools.setup(
    name="deeplabcut-live",
    version="0.0",
    author="A. & M. Mathis Labs",
    author_email="admin@deeplabcut.org",
    description="Class to load exported DeepLabCut networks and perform pose estimation on single frames (from a camera feed)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut-live",
    python_requires=">=3.5, <3.8",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points={
        "console_scripts": [
            "dlc-live-test=check_install.check_install:main",
            "dlc-live-bench=dlclive.bench:main",
            "dlc-live-benchmark=dlclive.benchmark:main",
        ]
    },
)
