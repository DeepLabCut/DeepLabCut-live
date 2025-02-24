import csv
import os
import platform
import subprocess
import sys
import time

import colorcet as cc
import cv2
import h5py
import numpy as np
import torch
from PIL import ImageColor
from pip._internal.operations import freeze

from dlclive import VERSION, DLCLive

# def download_benchmarking_data(
#     target_dir=".",
#     url="http://deeplabcut.rowland.harvard.edu/datasets/dlclivebenchmark.tar.gz",
# ):
#     """
#     Downloads a DeepLabCut-Live benchmarking Data (videos & DLC models).
#     """
#     import tarfile
#     import urllib.request

#     from tqdm import tqdm

#     def show_progress(count, block_size, total_size):
#         pbar.update(block_size)

#     def tarfilenamecutting(tarf):
#         """' auxfun to extract folder path
#         ie. /xyz-trainsetxyshufflez/
#         """
#         for memberid, member in enumerate(tarf.getmembers()):
#             if memberid == 0:
#                 parent = str(member.path)
#                 l = len(parent) + 1
#             if member.path.startswith(parent):
#                 member.path = member.path[l:]
#                 yield member

#     response = urllib.request.urlopen(url)
#     print(
#         "Downloading the benchmarking data from the DeepLabCut server @Harvard -> Go Crimson!!! {}....".format(
#             url
#         )
#     )
#     total_size = int(response.getheader("Content-Length"))
#     pbar = tqdm(unit="B", total=total_size, position=0)
#     filename, _ = urllib.request.urlretrieve(url, reporthook=show_progress)
#     with tarfile.open(filename, mode="r:gz") as tar:
#         tar.extractall(target_dir, members=tarfilenamecutting(tar))


def get_system_info() -> dict:
    """Return summary info for system running benchmark.

    Returns
    -------
    dict
        Dictionary containing the following system information:
        * ``host_name`` (str): name of machine
        * ``op_sys`` (str): operating system
        * ``python`` (str): path to python (which conda/virtual environment)
        * ``device`` (tuple): (device type (``'GPU'`` or ``'CPU'```), device information)
        * ``freeze`` (list): list of installed packages and versions
        * ``python_version`` (str): python version
        * ``git_hash`` (str, None): If installed from git repository, hash of HEAD commit
        * ``dlclive_version`` (str): dlclive version from :data:`dlclive.VERSION`
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


def analyze_video(
    video_path: str,
    model_path: str,
    model_type=str,
    device=str,
    display=True,
    pcutoff=0.5,
    display_radius=5,
    resize=None,
    cropping=None,  # Adding cropping to the function parameters
    dynamic=(False, 0.5, 10),
    save_poses=False,
    save_dir="model_predictions",
    draw_keypoint_names=False,
    cmap="bmy",
    get_sys_info=True,
):
    """
    Analyze a video to track keypoints using an imported DeepLabCut model, visualize keypoints on the video, and optionally save the keypoint data and the labelled video.

    Parameters:
    -----------
    video_path : str
        The path to the video file to be analyzed.
    dlc_live : DLCLive
        An instance of the DLCLive class.
    pcutoff : float, optional, default=0.5
        The probability cutoff value below which keypoints are not visualized.
    display_radius : int, optional, default=5
        The radius of the circles drawn to represent keypoints on the video frames.
    resize : tuple of int (width, height) or None, optional, default=None
        The size to which the frames should be resized. If None, the frames are not resized.
    cropping : list of int, optional, default=None
        Cropping parameters in pixel number: [x1, x2, y1, y2]
    save_poses : bool, optional, default=False
        Whether to save the detected poses to CSV and HDF5 files.
    save_dir : str, optional, default="model_predictions"
        The directory where the output video and pose data will be saved.
    draw_keypoint_names : bool, optional, default=False
        Whether to draw the names of the keypoints on the video frames.
    cmap : str, optional, default="bmy"
        The colormap from the colorcet library to use for keypoint visualization.

    Returns:
    --------
    poses : list of dict
        A list of dictionaries where each dictionary contains the frame number and the corresponding pose data.
    """
    # Create the DLCLive object with cropping
    dlc_live = DLCLive(
        path=model_path,
        model_type=model_type,
        device=device,
        display=display,
        resize=resize,
        cropping=cropping,  # Pass the cropping parameter
        dynamic=dynamic,
    )

    # Ensure save directory exists
    os.makedirs(name=save_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Start empty dict to save poses to for each frame
    poses = []
    frame_index = 0

    # Retrieve bodypart names and number of keypoints
    bodyparts = dlc_live.cfg["metadata"]["bodyparts"]
    num_keypoints = len(bodyparts)

    # Set colors and convert to RGB
    cmap_colors = getattr(cc, cmap)
    colors = [
        ImageColor.getrgb(color)
        for color in cmap_colors[:: int(len(cmap_colors) / num_keypoints)]
    ]

    # Define output video path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(save_dir, f"{video_name}_DLCLIVE_LABELLED.mp4")

    # Get video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vwriter = cv2.VideoWriter(
        filename=output_video_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=(frame_width, frame_height),
    )

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        # if frame_index == 0:
        #     pose = dlc_live.init_inference(frame)  # load DLC model
        try:
            # pose = dlc_live.get_pose(frame)
            if frame_index == 0:
                # dlc_live.dynamic = (False, dynamic[1], dynamic[2]) # TODO trying to fix issues with dynamic cropping jumping back and forth between dyanmic cropped and original image
                pose = dlc_live.init_inference(frame)  # load DLC model
            else:
                # dlc_live.dynamic = dynamic
                pose = dlc_live.get_pose(frame)
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            continue

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Frame {frame_index} processing time: {processing_time:.4f} seconds")

        poses.append({"frame": frame_index, "pose": pose})

        # Visualize keypoints
        this_pose = pose["poses"][0][0]
        for j in range(this_pose.shape[0]):
            if this_pose[j, 2] > pcutoff:
                x, y = map(int, this_pose[j, :2])
                cv2.circle(
                    frame,
                    center=(x, y),
                    radius=display_radius,
                    color=colors[j],
                    thickness=-1,
                )

                if draw_keypoint_names:
                    cv2.putText(
                        frame,
                        text=bodyparts[j],
                        org=(x + 10, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=colors[j],
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        vwriter.write(image=frame)
        frame_index += 1

    cap.release()
    vwriter.release()

    if get_sys_info:
        print(get_system_info())

    if save_poses:
        save_poses_to_files(video_path, save_dir, bodyparts, poses)

    return poses


def save_poses_to_files(video_path, save_dir, bodyparts, poses):
    """
    Save the keypoint poses detected in the video to CSV and HDF5 files.

    Parameters:
    -----------
    video_path : str
        The path to the video file that was analyzed.
    save_dir : str
        The directory where the pose data files will be saved.
    bodyparts : list of str
        A list of body part names corresponding to the keypoints.
    poses : list of dict
        A list of dictionaries where each dictionary contains the frame number and the corresponding pose data.

    Returns:
    --------
    None
    """
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    csv_save_path = os.path.join(save_dir, f"{base_filename}_poses.csv")
    h5_save_path = os.path.join(save_dir, f"{base_filename}_poses.h5")

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
            row = [frame_num] + [item for kp in pose for item in kp]
            writer.writerow(row)

    # Save to HDF5
    with h5py.File(h5_save_path, "w") as hf:
        hf.create_dataset(name="frames", data=[entry["frame"] for entry in poses])
        for i, bp in enumerate(bodyparts):
            hf.create_dataset(
                name=f"{bp}_x",
                data=[entry["pose"]["poses"][0][0][i, 0].item() for entry in poses],
            )
            hf.create_dataset(
                name=f"{bp}_y",
                data=[entry["pose"]["poses"][0][0][i, 1].item() for entry in poses],
            )
            hf.create_dataset(
                name=f"{bp}_confidence",
                data=[entry["pose"]["poses"][0][0][i, 2].item() for entry in poses],
            )
