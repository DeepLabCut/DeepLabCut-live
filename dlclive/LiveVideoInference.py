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


def analyze_live_video(
    model_path: str,
    model_type: str,
    device: str,
    camera: float = 0,
    experiment_name: str = "Test",
    precision: str = "FP32",
    snapshot: str = None,
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
    save_video=False,
):
    """
    Analyze a video to track keypoints using an imported DeepLabCut model, visualize keypoints on the video, and optionally save the keypoint data and the labelled video.

    Parameters:
    -----------
    camera : float, default=0 (webcam)
        The camera to record the live video from
    experiment_name : str, default = "Test"
        Prefix to label generated pose and video files
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
        precision=precision,
        snapshot=snapshot,
    )

    # Ensure save directory exists
    os.makedirs(name=save_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Error: Could not open video file {camera}")
        return

    # Start empty dict to save poses to for each frame
    poses, times = [], []
    frame_index = 0

    # Retrieve bodypart names and number of keypoints
    bodyparts = dlc_live.cfg["metadata"]["bodyparts"]
    num_keypoints = len(bodyparts)

    if save_video:
        # Set colors and convert to RGB
        cmap_colors = getattr(cc, cmap)
        colors = [
            ImageColor.getrgb(color)
            for color in cmap_colors[:: int(len(cmap_colors) / num_keypoints)]
        ]

        # Define output video path
        output_video_path = os.path.join(
            save_dir, f"{experiment_name}_DLCLIVE_LABELLED.mp4"
        )

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

        try:
            if frame_index == 0:
                pose = dlc_live.init_inference(frame)  # load DLC model
            else:
                pose = dlc_live.get_pose(frame)
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            continue

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Frame {frame_index} processing time: {processing_time:.4f} seconds")

        poses.append({"frame": frame_index, "pose": pose})
        if save_video:
            # Visualize keypoints
            this_pose = pose[0]["poses"][0][0]
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

        # Display the frame
        if display:
            cv2.imshow("DLCLive", frame)

        # Add key press check for quitting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    if save_video:
        vwriter.release()
    # cv2.destroyAllWindows()

    if get_sys_info:
        print(get_system_info())

    if save_poses:
        save_poses_to_files(experiment_name, save_dir, bodyparts, poses)

    return poses, times


def save_poses_to_files(experiment_name, save_dir, bodyparts, poses):
    """
    Save the keypoint poses detected in the video to CSV and HDF5 files.

    Parameters:
    -----------
    experiment_name : str
        Name of the experiment, used as a prefix for saving files.
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
    base_filename = os.path.splitext(os.path.basename(experiment_name))[0]
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
            pose_data = entry["pose"][0]["poses"][0][0]
            # Convert tensor data to numeric values
            row = [frame_num] + [
                item.item() if isinstance(item, torch.Tensor) else item
                for kp in pose_data
                for item in kp
            ]
            writer.writerow(row)

    # Save to HDF5
    with h5py.File(h5_save_path, "w") as hf:
        hf.create_dataset(name="frames", data=[entry["frame"] for entry in poses])
        for i, bp in enumerate(bodyparts):
            hf.create_dataset(
                name=f"{bp}_x",
                data=[
                    (
                        entry["pose"][0]["poses"][0][0][i, 0].item()
                        if isinstance(
                            entry["pose"][0]["poses"][0][0][i, 0], torch.Tensor
                        )
                        else entry["pose"][0]["poses"][0][0][i, 0]
                    )
                    for entry in poses
                ],
            )
            hf.create_dataset(
                name=f"{bp}_y",
                data=[
                    (
                        entry["pose"][0]["poses"][0][0][i, 1].item()
                        if isinstance(
                            entry["pose"][0]["poses"][0][0][i, 1], torch.Tensor
                        )
                        else entry["pose"][0]["poses"][0][0][i, 1]
                    )
                    for entry in poses
                ],
            )
            hf.create_dataset(
                name=f"{bp}_confidence",
                data=[
                    (
                        entry["pose"][0]["poses"][0][0][i, 2].item()
                        if isinstance(
                            entry["pose"][0]["poses"][0][0][i, 2], torch.Tensor
                        )
                        else entry["pose"][0]["poses"][0][0][i, 2]
                    )
                    for entry in poses
                ],
            )
