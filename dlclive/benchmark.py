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


def benchmark(
    path: str | Path,
    video_path: str | Path,
    single_animal: bool = True,
    resize: float | None = None,
    pixels: int | None = None,
    cropping: list[int] = None,
    dynamic: tuple[bool, float, int] = (False, 0.5, 10),
    n_frames: int = 1000,
    print_rate: bool = False,
    display: bool = False,
    pcutoff: float = 0.0,
    max_detections: int = 10,
    display_radius: int = 3,
    cmap: str = "bmy",
    save_poses: bool = False,
    save_video: bool = False,
    output: str | Path | None = None,
) -> tuple[np.ndarray, tuple, dict]:
    """Analyze DeepLabCut-live exported model on a video:

    Calculate inference time, display keypoints, or get poses/create a labeled video.

    Parameters
    ----------
    path : str
        path to exported DeepLabCut model
    video_path : str
        path to video file
    single_animal: bool
        to make code behave like DLCLive for tensorflow models
    resize : int, optional
        Resize factor. Can only use one of resize or pixels. If both are provided, will
        use pixels. by default None
    pixels : int, optional
        Downsize image to this number of pixels, maintaining aspect ratio. Can only use
        one of resize or pixels. If both are provided, will use pixels. by default None
    cropping : list of int
        cropping parameters in pixel number: [x1, x2, y1, y2]
    dynamic: triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that
        if an object is detected (i.e. any body part > detectiontreshold), then object
        boundaries are computed according to the smallest/largest x position and
        smallest/largest y position of all body parts. This  window is expanded by the
        margin and from then on only the posture within this crop is analyzed (until the
        object is lost, i.e. < detectiontreshold). The current position is utilized for
        updating the crop window for the next frame (this is why the margin is important
        and should be set large enough given the movement of the animal)
    n_frames : int, optional
        number of frames to run inference on, by default 1000
    print_rate : bool, optional
        flag to print inference rate frame by frame, by default False
    display : bool, optional
        flag to display keypoints on images. Useful for checking the accuracy of
        exported models.
    pcutoff : float, optional
        likelihood threshold to display keypoints
    max_detections: int
        for top-down models, the maximum number of individuals to detect in a frame
    display_radius : int, optional
        size (radius in pixels) of keypoint to display
    cmap : str, optional
        a string indicating the :package:`colorcet` colormap, `options here
        <https://colorcet.holoviz.org/>`, by default "bmy"
    save_poses : bool, optional
        flag to save poses to an hdf5 file. If True, operates similar to
        :function:`DeepLabCut.benchmark_videos`, by default False
    save_video : bool, optional
        flag to save a labeled video. If True, operates similar to
        :function:`DeepLabCut.create_labeled_video`, by default False
    output : str, optional
        path to directory to save pose and/or video file. If not specified, will use
        the directory of video_path, by default None

    Returns
    -------
    :class:`numpy.ndarray`
        vector of inference times
    tuple
        (image width, image height)
    dict
        metadata for video

    Example
    -------
    Return a vector of inference times for 10000 frames:
    dlclive.benchmark('/my/exported/model', 'my_video.avi', n_frames=10000)

    Return a vector of inference times, resizing images to half the width and height for inference
    dlclive.benchmark('/my/exported/model', 'my_video.avi', n_frames=10000, resize=0.5)

    Display keypoints to check the accuracy of an exported model
    dlclive.benchmark('/my/exported/model', 'my_video.avi', display=True)

    Analyze a video (save poses to hdf5) and create a labeled video, similar to :function:`DeepLabCut.benchmark_videos` and :function:`create_labeled_video`
    dlclive.benchmark('/my/exported/model', 'my_video.avi', save_poses=True, save_video=True)
    """
    path = Path(path)
    video_path = Path(video_path)
    if not video_path.exists():
        raise ValueError(f"Could not find video: {video_path}: check that it exists!")

    if output is None:
        output = video_path.parent
    else:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    # load video
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    n_frames = (
        n_frames
        if (n_frames > 0) and (n_frames < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        else (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    )
    n_frames = int(n_frames)
    im_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # get resize factor
    if pixels is not None:
        resize = np.sqrt(pixels / (im_size[0] * im_size[1]))

    if resize is not None:
        im_size = (int(im_size[0] * resize), int(im_size[1] * resize))

    # create video writer
    if save_video:
        colors = None
        out_vid_file = output / f"{video_path.stem}_DLCLIVE_LABELED.avi"
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(out_vid_file)
        print(fourcc)
        print(fps)
        print(im_size)
        vid_writer = cv2.VideoWriter(str(out_vid_file), fourcc, fps, im_size)

    # initialize DLCLive and perform inference
    inf_times = np.zeros(n_frames)
    poses = []

    live = DLCLive(
        model_path=path,
        single_animal=single_animal,
        resize=resize,
        cropping=cropping,
        dynamic=dynamic,
        display=display,
        max_detections=max_detections,
        pcutoff=pcutoff,
        display_radius=display_radius,
        display_cmap=cmap,
    )

    poses.append(live.init_inference(frame))

    iterator = range(n_frames)
    if print_rate or display:
        iterator = tqdm(iterator)

    for i in iterator:
        ret, frame = cap.read()
        if not ret:
            warnings.warn(
                f"Did not complete {n_frames:d} frames. There probably were not enough "
                f"frames in the video {video_path}."
            )
            break

        start_pose = time.time()
        poses.append(live.get_pose(frame))
        inf_times[i] = time.time() - start_pose
        if save_video:
            this_pose = poses[-1]

            if single_animal:
                # expand individual dimension
                this_pose = this_pose[None]

            num_idv, num_bpt = this_pose.shape[:2]
            num_colors = num_bpt

            if colors is None:
                all_colors = getattr(cc, cmap)
                colors = [
                    ImageColor.getcolor(c, "RGB")[::-1]
                    for c in all_colors[:: int(len(all_colors) / num_colors)]
                ]

            for j in range(num_idv):
                for k in range(num_bpt):
                    color_idx = k
                    if this_pose[j, k, 2] > pcutoff:
                        x = int(this_pose[j, k, 0])
                        y = int(this_pose[j, k, 1])
                        frame = cv2.circle(
                            frame,
                            (x, y),
                            display_radius,
                            colors[color_idx],
                            thickness=-1,
                        )

            if resize is not None:
                frame = cv2.resize(frame, im_size)
            vid_writer.write(frame)

        if print_rate:
            print(f"pose rate = {int(1 / inf_times[i]):d}")

    if print_rate:
        print(f"mean pose rate = {int(np.mean(1 / inf_times)):d}")

    # gather video and test parameterization
    # dont want to fail here so gracefully failing on exception --
    # eg. some packages of cv2 don't have CAP_PROP_CODEC_PIXEL_FORMAT
    try:
        fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    except:
        fourcc = ""

    try:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
    except Exception:
        fps = None

    try:
        pix_fmt = decode_fourcc(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))
    except Exception:
        pix_fmt = ""

    try:
        frame_count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception:
        frame_count = None

    try:
        orig_im_size = (
            round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    except Exception:
        orig_im_size = None

    meta = {
        "video_path": video_path,
        "video_codec": fourcc,
        "video_pixel_format": pix_fmt,
        "video_fps": fps,
        "video_total_frames": frame_count,
        "original_frame_size": orig_im_size,
        "dlclive_params": live.parameterization,
    }

    # close video
    cap.release()
    if save_video:
        vid_writer.release()

    if save_poses:
        bodyparts = live.cfg["metadata"]["bodyparts"]
        max_idv = np.max([p.shape[0] for p in poses])

        poses_array = -np.ones((len(poses), max_idv, len(bodyparts), 3))
        for i, p in enumerate(poses):
            num_det = len(p)
            poses_array[i, :num_det] = p
        poses = poses_array

        num_frames, num_idv, num_bpts = poses.shape[:3]
        individuals = [f"individual-{i}" for i in range(num_idv)]

        if has_pandas:
            poses = poses.reshape((num_frames, num_idv * num_bpts * 3))
            col_index = pd.MultiIndex.from_product(
                [individuals, bodyparts, ["x", "y", "likelihood"]],
                names=["individual", "bodyparts", "coords"],
            )
            pose_df = pd.DataFrame(poses, columns=col_index)

            out_dlc_file = output / (video_path.stem + "_DLCLIVE_POSES.h5")
            try:
                pose_df.to_hdf(out_dlc_file, key="df_with_missing", mode="w")
            except ImportError as err:
                print(
                    "Cannot export predictions to H5 file. Install ``pytables`` extra "
                    f"to export to HDF: {err}"
                )
            out_csv = Path(out_dlc_file).with_suffix(".csv")
            pose_df.to_csv(out_csv)

        else:
            warnings.warn(
                "Could not find installation of pandas; saving poses as a numpy array "
                "with the dimensions (n_frames, n_keypoints, [x, y, likelihood])."
            )
            np.save(str(output / (video_path.stem + "_DLCLIVE_POSES.npy")), poses)

    return inf_times, im_size, meta


def benchmark_videos(
    video_path: str,
    model_path: str,
    model_type: str,
    device: str,
    precision: str = "FP32",
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
    Analyzes a video to track keypoints using a DeepLabCut model, and optionally saves
    the keypoint data and the labeled video.

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
    precision : str, optional, default='FP32'
        Precision type for the model ('FP32' or 'FP16').
    display : bool, optional, default=True
        Whether to display frame with labelled key points.
    pcutoff : float, optional, default=0.5
        Probability cutoff below which keypoints are not visualized.
    display_radius : int, optional, default=5
        Radius of circles drawn for keypoints on video frames.
    resize : tuple of int (width, height) or None, optional
        Resize dimensions for video frames. e.g. if resize = 0.5, the video will be
        processed in half the original size. If None, no resizing is applied.
    cropping : list of int or None, optional
        Cropping parameters [x1, x2, y1, y2] in pixels. If None, no cropping is applied.
    dynamic : tuple, optional, default=(False, 0.5, 10) (True/false), p cutoff, margin)
        Parameters for dynamic cropping. If the state is true, then dynamic cropping
        will be performed. That means that if an object is detected (i.e. any body part
        > detectiontreshold), then object boundaries are computed according to the
        smallest/largest x position and smallest/largest y position of all body parts.
        This window is expanded by the margin and from then on only the posture within
        this crop is analyzed (until the object is lost, i.e. <detection treshold). The
        current position is used to update the crop window for the next frame
        (this is why the margin is important and should be set large enough given the
        movement of the animal).
    save_poses : bool, optional, default=False
        Whether to save the detected poses to CSV and HDF5 files.
    save_dir : str, optional, default='model_predictions'
        Directory to save output data and labeled video.
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
        device=device,
        display=display,
        resize=resize,
        cropping=cropping,  # Pass the cropping parameter
        dynamic=dynamic,
        precision=precision,
    )

    # Ensure save directory exists
    os.makedirs(name=save_dir, exist_ok=True)

    # Get the current date and time as a string
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Start empty dict to save poses to for each frame
    poses, times = [], []
    # Create variable indicate current frame. Later in the code +1 is added to frame_index
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
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(
            save_dir, f"{video_name}_DLCLIVE_LABELLED_{timestamp}.mp4"
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
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_index == 0:
        #     pose = dlc_live.init_inference(frame)  # load DLC model
        try:
            # pose = dlc_live.get_pose(frame)
            if frame_index == 0:
                # TODO trying to fix issues with dynamic cropping jumping back and forth
                #  between dyanmic cropped and original image
                # dlc_live.dynamic = (False, dynamic[1], dynamic[2])
                pose, inf_time = dlc_live.init_inference(frame)  # load DLC model
            else:
                # dlc_live.dynamic = dynamic
                pose, inf_time = dlc_live.get_pose(frame)
        except Exception as e:
            print(f"Error analyzing frame {frame_index}: {e}")
            continue

        poses.append({"frame": frame_index, "pose": pose})
        times.append(inf_time)

        if save_video:
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
    if save_video:
        vwriter.release()

    if get_sys_info:
        print(get_system_info())

    if save_poses:
        save_poses_to_files(video_path, save_dir, bodyparts, poses, timestamp=timestamp)

    return poses, times


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
