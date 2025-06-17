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
from tqdm import tqdm

from dlclive import DLCLive
from dlclive import VERSION
from dlclive import __file__ as dlcfile
from dlclive.factory import Engine
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


def benchmark(
    model_path: str,
    model_type: str,
    video_path: str,
    tf_config=None,
    device: str | None = None,
    resize: float | None = None,
    pixels: int | None = None,
    single_animal: bool = True,
    cropping=None,
    dynamic=(False, 0.5, 10),
    n_frames=1000,
    print_rate=False,
    precision: str = "FP32",
    display=True,
    pcutoff=0.5,
    display_radius=3,
    cmap="bmy",
    save_dir=None,
    save_poses=False,
    save_video=False,
    draw_keypoint_names=False,
):
    """
    Analyzes a video to track keypoints using a DeepLabCut model, and optionally saves the keypoint data and the labeled video.

    Parameters
    ----------
    model_path : str
        Path to the DeepLabCut model.
    model_type : str
        Which model to use. For the PyTorch engine, options are [`pytorch`]. For the
    video_path : str
        Path to the video file to be analyzed.
        TensorFlow engine, options are [`base`, `tensorrt`, `lite`].
    tf_config : :class:`tensorflow.ConfigProto`
        Tensorflow only. Tensorflow session configuration
    device : str
        Pytorch only. Device to run the model on ('cpu' or 'cuda').
    resize : float or None, optional
        Resize dimensions for video frames. e.g. if resize = 0.5, the video will be processed in half the original size. If None, no resizing is applied.
    pixels : int, optional
        downsize image to this number of pixels, maintaining aspect ratio.
        Can only use one of resize or pixels. If both are provided, will use pixels.
    single_animal: bool, optional, default=True
        Whether the video contains only one animal (True) or multiple animals (False).
    cropping : list of int or None, optional
        Cropping parameters [x1, x2, y1, y2] in pixels. If None, no cropping is applied.
    dynamic : tuple, optional, default=(False, 0.5, 10) (True/false), p cutoff, margin)
        Parameters for dynamic cropping. If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold), then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This window is expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detection treshold). The current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large enough given the movement of the animal).
    n_frames : int, optional
        Number of frames to run inference on, by default 1000
    print_rate: bool, optional, default=False
        Print the rate
    precision : str, optional, default='FP32'
        Precision type for the model ('FP32' or 'FP16').
    display : bool, optional, default=True
        Whether to display frame with labelled key points.
    pcutoff : float, optional, default=0.5
        Probability cutoff below which keypoints are not visualized.
    display_radius : int, optional, default=5
        Radius of circles drawn for keypoints on video frames.
    cmap : str, optional, default='bmy'
        Colormap from the colorcet library for keypoint visualization.
    save_dir : str, optional
        Directory to save output data and labeled video.
        If not specified, will use the directory of video_path, by default None
    save_poses : bool, optional, default=False
        Whether to save the detected poses to CSV and HDF5 files.
    save_video : bool, optional, default=False
        Whether to save the labeled video.
    draw_keypoint_names : bool, optional, default=False
        Whether to display keypoint names on video frames in the saved video.

    Returns
    -------
    tuple
        A tuple containing:
        - times (list of float): List of inference times for each frame.
        - im_size: tuple of two ints, corresponding to image size
        - metadata: dict
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    im_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if pixels is not None:
        resize = np.sqrt(pixels / (im_size[0] * im_size[1]))
    if resize is not None:
        im_size = (int(im_size[0] * resize), int(im_size[1] * resize))

    # Create the DLCLive object with cropping
    dlc_live = DLCLive(
        model_path=model_path,
        model_type=model_type,
        precision=precision,
        tf_config=tf_config,
        single_animal=single_animal,
        device=device,
        display=display,
        resize=resize,
        cropping=cropping,  # Pass the cropping parameter
        dynamic=dynamic,
        pcutoff=pcutoff,
        display_radius=display_radius,
        display_cmap=cmap,
    )

    if save_dir is None:
        save_dir = Path(video_path).resolve().parent
    else:
        save_dir = Path(save_dir)
    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get the current date and time as a string
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Retrieve bodypart names and number of keypoints
    engine = Engine.from_model_type(model_type)
    if engine == Engine.PYTORCH:
        bodyparts = dlc_live.read_config()["metadata"]["bodyparts"]
    else:
        bodyparts = dlc_live.read_config()["all_joints_names"]

    if save_video:
        colors, vwriter = setup_video_writer(
            video_path=video_path,
            save_dir=save_dir,
            timestamp=timestamp,
            num_keypoints=len(bodyparts),
            cmap=cmap,
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_size=im_size,
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
    iterator = range(n_frames) if print_rate or display else tqdm(range(n_frames))
    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            warnings.warn(
                (
                    "Did not complete {:d} frames. "
                    "There probably were not enough frames in the video {}."
                ).format(n_frames, video_path)
            )
            break

        start_time = time.perf_counter()
        if frame_index == 0:
            pose = dlc_live.init_inference(frame) # Loads model
        else:
            pose = dlc_live.get_pose(frame)

        inf_time = time.perf_counter() - start_time
        poses.append({"frame": frame_index, "pose": pose})
        times.append(inf_time)

        if print_rate:
            print("Inference rate = {:.3f} FPS".format(1 / inf_time), end="\r", flush=True)

        if save_video:
            draw_pose_and_write(
                frame=frame,
                pose=pose,
                resize=resize,
                colors=colors,
                bodyparts=bodyparts,
                pcutoff=pcutoff,
                display_radius=display_radius,
                draw_keypoint_names=draw_keypoint_names,
                vwriter=vwriter
            )

        frame_index += 1

    if print_rate:
        print("Mean inference rate: {:.3f} FPS".format(np.mean(1 / np.array(times)[1:])))

    metadata = _get_metadata(
        video_path=video_path,
        cap=cap,
        dlc_live=dlc_live
    )

    cap.release()

    dlc_live.close()

    if save_video:
        vwriter.release()

    if save_poses:
        if engine == Engine.PYTORCH:
            individuals = dlc_live.read_config()["metadata"].get("individuals", [])
        else:
            individuals = []
        n_individuals = len(individuals) or 1
        save_poses_to_files(video_path, save_dir, n_individuals, bodyparts, poses, timestamp=timestamp)

    return times, im_size, metadata


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
    resize: float,
    colors: list[tuple[int, int, int]],
    bodyparts: list[str],
    pcutoff: float,
    display_radius: int,
    draw_keypoint_names: bool,
    vwriter: cv2.VideoWriter,
):
    if len(pose.shape) == 2:
        pose = pose[None]

    if resize is not None and resize != 1.0:
        # Resize the frame
        frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        # Scale pose coordinates
        pose = pose.copy()
        pose[..., :2] *= resize

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


def _get_metadata(
    video_path: str,
    cap: cv2.VideoCapture,
    dlc_live: DLCLive
):
    try:
        fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    except:
        fourcc = ""
    try:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
    except:
        fps = None
    try:
        pix_fmt = decode_fourcc(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))
    except:
        pix_fmt = ""
    try:
        frame_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:
        frame_count = None
    try:
        orig_im_size = (
            round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    except:
        orig_im_size = None

    meta = {
        "video_path": video_path,
        "video_codec": fourcc,
        "video_pixel_format": pix_fmt,
        "video_fps": fps,
        "video_total_frames": frame_count,
        "original_frame_size": orig_im_size,
        "dlclive_params": dlc_live.parameterization,
    }
    return meta


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
