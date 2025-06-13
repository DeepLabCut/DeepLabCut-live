import csv
import platform
import subprocess
import sys
import time

import colorcet as cc
import cv2
import h5py
import numpy as np
from pathlib import Path
from PIL import ImageColor
from pip._internal.operations import freeze
import torch
from tqdm import tqdm

# torch import needs to switch order with "from pip._internal.operations import freeze" because of crash
# see https://github.com/pytorch/pytorch/issues/140914

from dlclive import DLCLive
from dlclive.version import VERSION


def get_system_info() -> dict:
    """
    Returns a summary of system information relevant to running benchmarking.

    Returns
    -------
    dict
        A dictionary containing the following system information:
        - host_name (str): Name of the machine.
        - op_sys (str): Operating system.
        - python (str): Path to the Python executable, indicating the conda/virtual environment in use.
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


def benchmark(
    video_path: str,
    model_path: str,
    model_type: str,
    device: str,
    single_animal: bool,
    save_dir=None,
    n_frames=1000,
    precision: str = "FP32",
    display=True,
    pcutoff=0.5,
    display_radius=3,
    resize=None,
    cropping=None,  # Adding cropping to the function parameters
    dynamic=(False, 0.5, 10),
    save_poses=False,
    draw_keypoint_names=False,
    cmap="bmy",
    get_sys_info=True,
    save_video=False,
):
    """
    Analyzes a video to track keypoints using a DeepLabCut model, and optionally saves the keypoint data and the labeled video.

    Parameters
    ----------
    video_path : str
        Path to the video file to be analyzed.
    model_path : str
        Path to the DeepLabCut model.
    model_type : str
        Type of the model (e.g., 'onnx').
    device : str
        Device to run the model on ('cpu' or 'cuda').
    single_animal: bool
        Whether the video contains only one animal (True) or multiple animals (False).
    save_dir : str, optional
        Directory to save output data and labeled video.
        If not specified, will use the directory of video_path, by default None
    n_frames : int, optional
        Number of frames to run inference on, by default 1000
    precision : str, optional, default='FP32'
        Precision type for the model ('FP32' or 'FP16').
    display : bool, optional, default=True
        Whether to display frame with labelled key points.
    pcutoff : float, optional, default=0.5
        Probability cutoff below which keypoints are not visualized.
    display_radius : int, optional, default=5
        Radius of circles drawn for keypoints on video frames.
    resize : tuple of int (width, height) or None, optional
        Resize dimensions for video frames. e.g. if resize = 0.5, the video will be processed in half the original size. If None, no resizing is applied.
    cropping : list of int or None, optional
        Cropping parameters [x1, x2, y1, y2] in pixels. If None, no cropping is applied.
    dynamic : tuple, optional, default=(False, 0.5, 10) (True/false), p cutoff, margin)
        Parameters for dynamic cropping. If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold), then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This window is expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detection treshold). The current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large enough given the movement of the animal).
    save_poses : bool, optional, default=False
        Whether to save the detected poses to CSV and HDF5 files.
    draw_keypoint_names : bool, optional, default=False
        Whether to display keypoint names on video frames in the saved video.
    cmap : str, optional, default='bmy'
        Colormap from the colorcet library for keypoint visualization.
    get_sys_info : bool, optional, default=True
        Whether to print system information.
    save_video : bool, optional, default=False
        Whether to save the labeled video.

    Returns
    -------
    tuple
        A tuple containing:
        - poses (list of dict): List of pose data for each frame.
        - times (list of float): List of inference times for each frame.
    """

    # Create the DLCLive object with cropping
    dlc_live = DLCLive(
        model_path=model_path,
        model_type=model_type,
        single_animal=single_animal,
        device=device,
        display=display,
        resize=resize,
        cropping=cropping,  # Pass the cropping parameter
        dynamic=dynamic,
        precision=precision,
        pcutoff=pcutoff,
        display_radius=display_radius,
        display_cmap=cmap,
    )

    if save_dir is None:
        save_dir = Path(video_path).resolve().parent
    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get the current date and time as a string
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Retrieve bodypart names and number of keypoints
    bodyparts = dlc_live.read_config()["metadata"]["bodyparts"]

    if save_video:
        colors, vwriter = setup_video_writer(
            video_path=video_path,
            save_dir=save_dir,
            timestamp=timestamp,
            num_keypoints=len(bodyparts),
            cmap=cmap,
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

    # Start empty dict to save poses to for each frame
    poses, times = [], []
    frame_index = 0

    total_n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = int(
        n_frames
        if (n_frames > 0) and n_frames < total_n_frames
        else total_n_frames
    )
    iterator = range(n_frames) if display else tqdm(range(n_frames))
    for i in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            start_time = time.perf_counter()
            if frame_index == 0:
                pose = dlc_live.init_inference(frame) # Loads model
            else:
                pose = dlc_live.get_pose(frame)
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            frame_index += 1
            continue

        inf_time = time.perf_counter() - start_time
        poses.append({"frame": frame_index, "pose": pose})
        times.append(inf_time)

        if save_video:
            draw_pose_and_write(
                frame=frame,
                pose=pose,
                colors=colors,
                bodyparts=bodyparts,
                pcutoff=pcutoff,
                display_radius=display_radius,
                draw_keypoint_names=draw_keypoint_names,
                vwriter=vwriter
            )

        frame_index += 1

    cap.release()
    if save_video:
        vwriter.release()

    if get_sys_info:
        print(get_system_info())

    if save_poses:
        individuals = dlc_live.read_config()["metadata"].get("individuals", [])
        n_individuals = len(individuals) or 1
        save_poses_to_files(video_path, save_dir, n_individuals, bodyparts, poses, timestamp=timestamp)

    return poses, times

def setup_video_writer(
    video_path:str,
    save_dir:str,
    timestamp:str,
    num_keypoints:int,
    cmap:str,
    fps:float,
    frame_size:tuple[int, int],
):
    # Set colors and convert to RGB
    cmap_colors = getattr(cc, cmap)
    colors = [
        ImageColor.getrgb(color)
        for color in cmap_colors[:: int(len(cmap_colors) / num_keypoints)]
    ]

    # Define output video path
    video_path = Path(video_path)
    video_name = video_path.stem  # filename without extension
    output_video_path = Path(save_dir) / f"{video_name}_DLCLIVE_LABELLED_{timestamp}.mp4"

    # Get video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vwriter = cv2.VideoWriter(
        filename=output_video_path,
        fourcc=fourcc,
        fps=fps,
        frameSize=frame_size,
    )

    return colors, vwriter

def draw_pose_and_write(
    frame: np.ndarray,
    pose: np.ndarray,
    colors: list[tuple[int, int, int]],
    bodyparts: list[str],
    pcutoff: float,
    display_radius: int,
    draw_keypoint_names: bool,
    vwriter: cv2.VideoWriter,
):
    if len(pose.shape) == 2:
        pose = pose[None]

    # Visualize keypoints
    for i in range(pose.shape[0]):
        for j in range(pose.shape[1]):
            if pose[i, j, 2] > pcutoff:
                x, y = map(int, pose[i, j, :2])
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

def save_poses_to_files(video_path, save_dir, n_individuals, bodyparts, poses, timestamp):
    """
    Saves the detected keypoint poses from the video to CSV and HDF5 files.

    Parameters
    ----------
    video_path : str
        Path to the analyzed video file.
    save_dir : str
        Directory where the pose data files will be saved.
    n_individuals: int
        Number of individuals
    bodyparts : list of str
        List of body part names corresponding to the keypoints.
    poses : list of dict
        List of dictionaries containing frame numbers and corresponding pose data.

    Returns
    -------
    None
    """
    import pandas as pd

    base_filename = Path(video_path).stem
    save_dir = Path(save_dir)
    h5_save_path = save_dir / f"{base_filename}_poses_{timestamp}.h5"
    csv_save_path = save_dir / f"{base_filename}_poses_{timestamp}.csv"

    poses_array = _create_poses_np_array(n_individuals, bodyparts, poses)
    flattened_poses = poses_array.reshape(poses_array.shape[0], -1)

    if n_individuals == 1:
        pdindex = pd.MultiIndex.from_product(
            [bodyparts, ["x", "y", "likelihood"]], names=["bodyparts", "coords"]
        )
    else:
        individuals = [f"individual_{i}" for i in range(n_individuals)]
        pdindex = pd.MultiIndex.from_product(
            [individuals, bodyparts, ["x", "y", "likelihood"]], names=["individuals", "bodyparts", "coords"]
        )

    pose_df = pd.DataFrame(flattened_poses, columns=pdindex)

    pose_df.to_hdf(h5_save_path, key="df_with_missing", mode="w")
    pose_df.to_csv(csv_save_path, index=False)

def _create_poses_np_array(n_individuals: int, bodyparts: list, poses: list):
    # Create numpy array with poses:
    max_frame = max(p["frame"] for p in poses)
    pose_target_shape = (n_individuals, len(bodyparts), 3)
    poses_array = np.full((max_frame + 1, *pose_target_shape), np.nan)

    for item in poses:
        frame = item["frame"]
        pose = item["pose"]
        if pose.ndim == 2:
            pose = pose[np.newaxis, :, :]
        padded_pose = np.full(pose_target_shape, np.nan)
        slices = tuple(slice(0, min(pose.shape[i], pose_target_shape[i])) for i in range(3))
        padded_pose[slices] = pose[slices]
        poses_array[frame] = padded_pose

    return poses_array


import argparse
import os


def main():
    """Provides a command line interface to analyze_video function."""

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

    # Call the analyze_video function with the parsed arguments
    benchmark(
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
