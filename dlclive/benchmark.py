"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import csv
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import colorcet as cc
import cv2
import numpy as np
from PIL import ImageColor
from pip._internal.operations import freeze
import torch

try:
    import pandas as pd

    has_pandas = True
except ModuleNotFoundError as err:
    has_pandas = False

from dlclive import DLCLive
from dlclive import VERSION
from dlclive import __file__ as dlcfile

from dlclive.utils import decode_fourcc


def download_benchmarking_data(
    target_dir=".",
    url="https://huggingface.co/datasets/mwmathis/DLCspeed_benchmarking/resolve/main/Data-DLC-live-benchmark.zip",
):
    """
    Downloads and extracts DeepLabCut-Live benchmarking data (videos & DLC models).
    """
    import os
    import urllib.request
    from tqdm import tqdm
    import zipfile

    # Avoid nested folder issue
    if os.path.basename(os.path.normpath(target_dir)) == "Data-DLC-live-benchmark":
        target_dir = os.path.dirname(os.path.normpath(target_dir))
    os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists

    zip_path = os.path.join(target_dir, "Data-DLC-live-benchmark.zip")

    if os.path.exists(zip_path):
        print(f"{zip_path} already exists. Skipping download.")
    else:
        def show_progress(count, block_size, total_size):
            pbar.update(block_size)

        print(f"Downloading the benchmarking data from {url} ...")
        pbar = tqdm(unit="B", total=0, position=0, desc="Downloading")

        filename, _ = urllib.request.urlretrieve(url, filename=zip_path, reporthook=show_progress)
        pbar.close()

    print(f"Extracting {zip_path} to {target_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def get_system_info() -> dict:
    """
    Returns a summary of system information relevant to running benchmarking.

    Returns
    -------
    dict
        A dictionary containing the following system information:
        - host_name (str): Name of the machine.
        - op_sys (str): Operating system.
        - python (str): Path to the Python executable, indicating the conda/virtual
            environment in use.
        - device_type (str): Type of device used ('GPU' or 'CPU').
        - device (list): List containing the name of the GPU or CPU brand.
        - freeze (list): List of installed Python packages with their versions.
        - python_version (str): Version of Python in use.
        - git_hash (str or None): If installed from git repository, hash of HEAD commit.
        - dlclive_version (str): Version of the DLCLive package.
    """

    # Get OS and host name
    op_sys = platform.platform()
    host_name = platform.node().replace(" ", "")

    # Get Python executable path
    if platform.system() == "Windows":
        host_python = sys.executable.split(os.path.sep)[-2]
    else:
        host_python = sys.executable.split(os.path.sep)[-3]

    # Try to get git hash if possible
    git_hash = None
    dlc_basedir = os.path.dirname(os.path.dirname(__file__))
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dlc_basedir)
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        # Not installed from git repo, e.g., pypi
        pass

    # Get device info (GPU or CPU)
    if torch.cuda.is_available():
        dev_type = "GPU"
        dev = [torch.cuda.get_device_name(torch.cuda.current_device())]
    else:
        from cpuinfo import get_cpu_info

        dev_type = "CPU"
        dev = [get_cpu_info()["brand_raw"]]

    return {
        "host_name": host_name,
        "op_sys": op_sys,
        "python": host_python,
        "device_type": dev_type,
        "device": dev,
        "freeze": list(freeze.freeze()),
        "python_version": sys.version,
        "git_hash": git_hash,
        "dlclive_version": VERSION,
    }


def save_poses_to_files(video_path, save_dir, bodyparts, poses, timestamp):
    """
    Saves the detected keypoint poses from the video to CSV and HDF5 files.

    Parameters
    ----------
    video_path : str
        Path to the analyzed video file.
    save_dir : str
        Directory where the pose data files will be saved.
    bodyparts : list of str
        List of body part names corresponding to the keypoints.
    poses : list of dict
        List of dictionaries containing frame numbers and corresponding pose data.

    Returns
    -------
    None
    """

    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    csv_save_path = os.path.join(save_dir, f"{base_filename}_poses_{timestamp}.csv")
    h5_save_path = os.path.join(save_dir, f"{base_filename}_poses_{timestamp}.h5")

    # Save to CSV
    with open(csv_save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["frame"] + [
            f"{bp}_{axis}" for bp in bodyparts for axis in ["x", "y", "confidence"]
        ]
        writer.writerow(header)
        for entry in poses:
            frame_num = entry["frame"]
            pose = entry["pose"]["poses"][0][0]
            row = [frame_num] + [
                item.item() if isinstance(item, torch.Tensor) else item
                for kp in pose
                for item in kp
            ]
            writer.writerow(row)


import argparse
import os


def main():
    """Provides a command line interface to benchmark_videos function."""
    parser = argparse.ArgumentParser(
        description="Analyze a video using a DeepLabCut model and visualize keypoints."
    )
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("model_type", type=str, help="Type of the model (e.g., 'DLC').")
    parser.add_argument(
        "device", type=str, help="Device to run the model on (e.g., 'cuda' or 'cpu')."
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="FP32",
        help="Model precision (e.g., 'FP32', 'FP16').",
    )
    parser.add_argument(
        "-d", "--display", action="store_true", help="Display keypoints on the video."
    )
    parser.add_argument(
        "-c",
        "--pcutoff",
        type=float,
        default=0.5,
        help="Probability cutoff for keypoints visualization.",
    )
    parser.add_argument(
        "-dr",
        "--display-radius",
        type=int,
        default=5,
        help="Radius of keypoint circles in the display.",
    )
    parser.add_argument(
        "-r",
        "--resize",
        type=int,
        default=None,
        help="Resize video frames to [width, height].",
    )
    parser.add_argument(
        "-x",
        "--cropping",
        type=int,
        nargs=4,
        default=None,
        help="Cropping parameters [x1, x2, y1, y2].",
    )
    parser.add_argument(
        "-y",
        "--dynamic",
        type=float,
        nargs=3,
        default=[False, 0.5, 10],
        help="Dynamic cropping [flag, pcutoff, margin].",
    )
    parser.add_argument(
        "--save-poses", action="store_true", help="Save the keypoint poses to files."
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the output video with keypoints.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="model_predictions",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--draw-keypoint-names",
        action="store_true",
        help="Draw keypoint names on the video.",
    )
    parser.add_argument(
        "--cmap", type=str, default="bmy", help="Colormap for keypoints visualization."
    )
    parser.add_argument(
        "--no-sys-info",
        action="store_false",
        help="Do not print system info.",
        dest="get_sys_info",
    )

    args = parser.parse_args()

    # Call the benchmark_videos function with the parsed arguments
    benchmark_videos(
        video_path=args.video_path,
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        precision=args.precision,
        display=args.display,
        pcutoff=args.pcutoff,
        display_radius=args.display_radius,
        resize=tuple(args.resize) if args.resize else None,
        cropping=args.cropping,
        dynamic=tuple(args.dynamic),
        save_poses=args.save_poses,
        save_dir=args.save_dir,
        draw_keypoint_names=args.draw_keypoint_names,
        cmap=args.cmap,
        get_sys_info=args.get_sys_info,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
