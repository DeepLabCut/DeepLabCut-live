#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deeplabcut-live",
    version="0.0.b0",
    author="A. & M. Mathis Labs",
    author_email="alexander@deeplabcut.org",
    description="Class to load exported DLC networks and perform pose estimation on single frames (from camera feed)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepLabCut/DeepLabCut-live",
    python_requires = '>=3.5, <3.8',
    install_requires=['numpy', 'ruamel.yaml', 'colorcet', 'pillow'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ),
    entry_points = {'console_scripts' : ['dlclive_human_demo=demos.human.run_live_human']}
)
