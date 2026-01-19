#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Helpers for configuration file IO"""
from pathlib import Path

import ruamel.yaml


def read_yaml(file_path: str | Path) -> dict:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(
            f"The pose configuration file for the exported model at {str(file_path)} "
            "was not found. Please check the path to the exported model directory"
        )

    with open(file_path, "r") as f:
        cfg = ruamel.yaml.YAML(typ="safe", pure=True).load(f)

    return cfg
