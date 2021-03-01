"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import platform
import os
import time
import sys
import warnings
import subprocess
import typing
import pickle
import colorcet as cc
from PIL import ImageColor
import ruamel

try:
    from pip._internal.operations import freeze
except ImportError:
    from pip.operations import freeze

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2

from dlclive import DLCLive
from dlclive import VERSION
from dlclive import __file__ as dlcfile

from dlclive.utils import decode_fourcc


def download_benchmarking_data(
    target_dir=".",
    url="http://deeplabcut.rowland.harvard.edu/datasets/dlclivebenchmark.tar.gz",
):
    """
    Downloads a DeepLabCut-Live benchmarking Data (videos & DLC models).
    """
    import urllib.request
    import tarfile
    from tqdm import tqdm

    def show_progress(count, block_size, total_size):
        pbar.update(block_size)

    def tarfilenamecutting(tarf):
        """' auxfun to extract folder path
        ie. /xyz-trainsetxyshufflez/
        """
        for memberid, member in enumerate(tarf.getmembers()):
            if memberid == 0:
                parent = str(member.path)
                l = len(parent) + 1
            if member.path.startswith(parent):
                member.path = member.path[l:]
                yield member

    response = urllib.request.urlopen(url)
    print(
        "Downloading the benchmarking data from the DeepLabCut server @Harvard -> Go Crimson!!! {}....".format(
            url
        )
    )
    total_size = int(response.getheader("Content-Length"))
    pbar = tqdm(unit="B", total=total_size, position=0)
    filename, _ = urllib.request.urlretrieve(url, reporthook=show_progress)
    with tarfile.open(filename, mode="r:gz") as tar:
        tar.extractall(target_dir, members=tarfilenamecutting(tar))


def get_system_info() -> dict:
    """ Return summary info for system running benchmark
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

    # get os

    op_sys = platform.platform()
    host_name = platform.node().replace(" ", "")

    # A string giving the absolute path of the executable binary for the Python interpreter, on systems where this makes sense.
    if platform.system() == "Windows":
        host_python = sys.executable.split(os.path.sep)[-2]
    else:
        host_python = sys.executable.split(os.path.sep)[-3]

    # try to get git hash if possible
    dlc_basedir = os.path.dirname(os.path.dirname(dlcfile))
    git_hash = None
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dlc_basedir
        )
        git_hash = git_hash.decode("utf-8").rstrip("\n")
    except subprocess.CalledProcessError:
        # not installed from git repo, eg. pypi
        # fine, pass quietly
        pass

    # get device info (GPU or CPU)
    dev = None
    if tf.test.is_gpu_available():
        gpu_name = tf.test.gpu_device_name()
        from tensorflow.python.client import device_lib

        dev_desc = [
            d.physical_device_desc
            for d in device_lib.list_local_devices()
            if d.name == gpu_name
        ]
        dev = [d.split(",")[1].split(":")[1].strip() for d in dev_desc]
        dev_type = "GPU"
    else:
        from cpuinfo import get_cpu_info

        dev = [get_cpu_info()["brand"]]
        dev_type = "CPU"

    return {
        "host_name": host_name,
        "op_sys": op_sys,
        "python": host_python,
        "device_type": dev_type,
        "device": dev,
        # pip freeze to get versions of all packages
        "freeze": list(freeze.freeze()),
        "python_version": sys.version,
        "git_hash": git_hash,
        "dlclive_version": VERSION,
    }


def benchmark(
    model_path,
    video_path,
    tf_config=None,
    resize=None,
    pixels=None,
    cropping=None,
    dynamic=(False, 0.5, 10),
    n_frames=1000,
    print_rate=False,
    display=False,
    pcutoff=0.0,
    display_radius=3,
    cmap="bmy",
    save_poses=False,
    save_video=False,
    output=None,
) -> typing.Tuple[np.ndarray, tuple, bool, dict]:
    """ Analyze DeepLabCut-live exported model on a video:
    Calculate inference time,
    display keypoints, or
    get poses/create a labeled video

    Parameters
    ----------
    model_path : str
        path to exported DeepLabCut model
    video_path : str
        path to video file
    tf_config : :class:`tensorflow.ConfigProto`
        tensorflow session configuration
    resize : int, optional
        resize factor. Can only use one of resize or pixels. If both are provided, will use pixels. by default None
    pixels : int, optional
        downsize image to this number of pixels, maintaining aspect ratio. Can only use one of resize or pixels. If both are provided, will use pixels. by default None
    cropping : list of int
        cropping parameters in pixel number: [x1, x2, y1, y2]
    dynamic: triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal)
    n_frames : int, optional
        number of frames to run inference on, by default 1000
    print_rate : bool, optional
        flat to print inference rate frame by frame, by default False
    display : bool, optional
        flag to display keypoints on images. Useful for checking the accuracy of exported models.
    pcutoff : float, optional
        likelihood threshold to display keypoints
    display_radius : int, optional
        size (radius in pixels) of keypoint to display
    cmap : str, optional
        a string indicating the :package:`colorcet` colormap, `options here <https://colorcet.holoviz.org/>`, by default "bmy"
    save_poses : bool, optional
        flag to save poses to an hdf5 file. If True, operates similar to :function:`DeepLabCut.benchmark_videos`, by default False
    save_video : bool, optional
        flag to save a labeled video. If True, operates similar to :function:`DeepLabCut.create_labeled_video`, by default False
    output : str, optional
        path to directory to save pose and/or video file. If not specified, will use the directory of video_path, by default None

    Returns
    -------
    :class:`numpy.ndarray`
        vector of inference times
    tuple
        (image width, image height)
    bool
        tensorflow inference flag
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

    ### load video

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    n_frames = (
        n_frames
        if (n_frames > 0) and (n_frames < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        else (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    )
    n_frames = int(n_frames)
    im_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### get resize factor

    if pixels is not None:
        resize = np.sqrt(pixels / (im_size[0] * im_size[1]))
    if resize is not None:
        im_size = (int(im_size[0] * resize), int(im_size[1] * resize))

    ### create video writer

    if save_video:
        colors = None
        out_dir = (
            output
            if output is not None
            else os.path.dirname(os.path.realpath(video_path))
        )
        out_vid_base = os.path.basename(video_path)
        out_vid_file = os.path.normpath(
            f"{out_dir}/{os.path.splitext(out_vid_base)[0]}_DLCLIVE_LABELED.avi"
        )
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        fps = cap.get(cv2.CAP_PROP_FPS)
        vwriter = cv2.VideoWriter(out_vid_file, fourcc, fps, im_size)

    ### check for pandas installation if using save_poses flag

    if save_poses:
        try:
            import pandas as pd

            use_pandas = True
        except:
            use_pandas = False
            warnings.warn(
                "Could not find installation of pandas; saving poses as a numpy array with the dimensions (n_frames, n_keypoints, [x, y, likelihood])."
            )

    ### initialize DLCLive and perform inference

    inf_times = np.zeros(n_frames)
    poses = []

    live = DLCLive(
        model_path,
        tf_config=tf_config,
        resize=resize,
        cropping=cropping,
        dynamic=dynamic,
        display=display,
        pcutoff=pcutoff,
        display_radius=display_radius,
        display_cmap=cmap,
    )

    poses.append(live.init_inference(frame))
    TFGPUinference = True if len(live.outputs) == 1 else False

    iterator = range(n_frames) if (print_rate) or (display) else tqdm(range(n_frames))
    for i in iterator:

        ret, frame = cap.read()

        if not ret:
            warnings.warn(
                "Did not complete {:d} frames. There probably were not enough frames in the video {}.".format(
                    n_frames, video_path
                )
            )
            break

        start_pose = time.time()
        poses.append(live.get_pose(frame))
        inf_times[i] = time.time() - start_pose

        if save_video:

            if colors is None:
                all_colors = getattr(cc, cmap)
                colors = [
                    ImageColor.getcolor(c, "RGB")[::-1]
                    for c in all_colors[:: int(len(all_colors) / poses[-1].shape[0])]
                ]

            this_pose = poses[-1]
            for j in range(this_pose.shape[0]):
                if this_pose[j, 2] > pcutoff:
                    x = int(this_pose[j, 0])
                    y = int(this_pose[j, 1])
                    frame = cv2.circle(
                        frame, (x, y), display_radius, colors[j], thickness=-1
                    )

            if resize is not None:
                frame = cv2.resize(frame, im_size)
            vwriter.write(frame)

        if print_rate:
            print("pose rate = {:d}".format(int(1 / inf_times[i])))

    if print_rate:
        print("mean pose rate = {:d}".format(int(np.mean(1 / inf_times))))

    ### gather video and test parameterization

    # dont want to fail here so gracefully failing on exception --
    # eg. some packages of cv2 don't have CAP_PROP_CODEC_PIXEL_FORMAT
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
        "dlclive_params": live.parameterization,
    }

    ### close video and tensorflow session

    cap.release()
    live.close()

    if save_video:
        vwriter.release()

    if save_poses:

        cfg_path = os.path.normpath(f"{model_path}/pose_cfg.yaml")
        ruamel_file = ruamel.yaml.YAML()
        dlc_cfg = ruamel_file.load(open(cfg_path, "r"))
        bodyparts = dlc_cfg["all_joints_names"]
        poses = np.array(poses)

        if use_pandas:

            poses = poses.reshape((poses.shape[0], poses.shape[1] * poses.shape[2]))
            pdindex = pd.MultiIndex.from_product(
                [bodyparts, ["x", "y", "likelihood"]], names=["bodyparts", "coords"]
            )
            pose_df = pd.DataFrame(poses, columns=pdindex)

            out_dir = (
                output
                if output is not None
                else os.path.dirname(os.path.realpath(video_path))
            )
            out_vid_base = os.path.basename(video_path)
            out_dlc_file = os.path.normpath(
                f"{out_dir}/{os.path.splitext(out_vid_base)[0]}_DLCLIVE_POSES.h5"
            )
            pose_df.to_hdf(out_dlc_file, key="df_with_missing", mode="w")

        else:

            out_vid_base = os.path.basename(video_path)
            out_dlc_file = os.path.normpath(
                f"{out_dir}/{os.path.splitext(out_vid_base)[0]}_DLCLIVE_POSES.npy"
            )
            np.save(out_dlc_file, poses)

    return inf_times, im_size, TFGPUinference, meta


def save_inf_times(
    sys_info, inf_times, im_size, TFGPUinference, model=None, meta=None, output=None
):
    """ Save inference time data collected using :function:`benchmark` with system information to a pickle file.
    This is primarily used through :function:`benchmark_videos`


    Parameters
    ----------
    sys_info : tuple
        system information generated by :func:`get_system_info`
    inf_times : :class:`numpy.ndarray`
        array of inference times generated by :func:`benchmark`
    im_size : tuple or :class:`numpy.ndarray`
        image size (width, height) for each benchmark run. If an array, each row corresponds to a row in inf_times
    TFGPUinference: bool
        flag if using tensorflow inference or numpy inference DLC model
    model: str, optional
        name of model
    meta : dict, optional
        metadata returned by :func:`benchmark`
    output : str, optional
        path to directory to save data. If None, uses pwd, by default None

    Returns
    -------
    bool
        flag indicating successful save
    """

    output = output if output is not None else os.getcwd()
    model_type = None
    if model is not None:
        if "resnet" in model:
            model_type = "resnet"
        elif "mobilenet" in model:
            model_type = "mobilenet"
        else:
            model_type = None

    fn_ind = 0
    base_name = (
        f"benchmark_{sys_info['host_name']}_{sys_info['device_type']}_{fn_ind}.pickle"
    )
    out_file = os.path.normpath(f"{output}/{base_name}")
    while os.path.isfile(out_file):
        fn_ind += 1
        base_name = f"benchmark_{sys_info['host_name']}_{sys_info['device_type']}_{fn_ind}.pickle"
        out_file = os.path.normpath(f"{output}/{base_name}")

    # summary stats (mean inference time & standard error of mean)
    stats = zip(
        np.mean(inf_times, 1),
        np.std(inf_times, 1) * 1.0 / np.sqrt(np.shape(inf_times)[1]),
    )

    # for stat in stats:
    #    print("Stats:", stat)

    data = {
        "model": model,
        "model_type": model_type,
        "TFGPUinference": TFGPUinference,
        "im_size": im_size,
        "inference_times": inf_times,
        "stats": stats,
    }

    data.update(sys_info)
    if meta:
        data.update(meta)

    os.makedirs(os.path.normpath(output), exist_ok=True)
    pickle.dump(data, open(out_file, "wb"))

    return True


def benchmark_videos(
    model_path,
    video_path,
    output=None,
    n_frames=1000,
    tf_config=None,
    resize=None,
    pixels=None,
    cropping=None,
    dynamic=(False, 0.5, 10),
    print_rate=False,
    display=False,
    pcutoff=0.5,
    display_radius=3,
    cmap="bmy",
    save_poses=False,
    save_video=False,
):
    """Analyze videos using DeepLabCut-live exported models.
    Analyze multiple videos and/or multiple options for the size of the video
    by specifying a resizing factor or the number of pixels to use in the image (keeping aspect ratio constant).
    Options to record inference times (to examine inference speed),
    display keypoints to visually check the accuracy,
    or save poses to an hdf5 file as in :function:`deeplabcut.benchmark_videos` and
    create a labeled video as in :function:`deeplabcut.create_labeled_video`.

    Parameters
    ----------
    model_path : str
        path to exported DeepLabCut model
    video_path : str or list
        path to video file or list of paths to video files
    output : str
        path to directory to save results
    tf_config : :class:`tensorflow.ConfigProto`
        tensorflow session configuration
    resize : int, optional
        resize factor. Can only use one of resize or pixels. If both are provided, will use pixels. by default None
    pixels : int, optional
        downsize image to this number of pixels, maintaining aspect ratio. Can only use one of resize or pixels. If both are provided, will use pixels. by default None
    cropping : list of int
        cropping parameters in pixel number: [x1, x2, y1, y2]
    dynamic: triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal)
    n_frames : int, optional
        number of frames to run inference on, by default 1000
    print_rate : bool, optional
        flat to print inference rate frame by frame, by default False
    display : bool, optional
        flag to display keypoints on images. Useful for checking the accuracy of exported models.
    pcutoff : float, optional
        likelihood threshold to display keypoints
    display_radius : int, optional
        size (radius in pixels) of keypoint to display
    cmap : str, optional
        a string indicating the :package:`colorcet` colormap, `options here <https://colorcet.holoviz.org/>`, by default "bmy"
    save_poses : bool, optional
        flag to save poses to an hdf5 file. If True, operates similar to :function:`DeepLabCut.benchmark_videos`, by default False
    save_video : bool, optional
        flag to save a labeled video. If True, operates similar to :function:`DeepLabCut.create_labeled_video`, by default False

    Example
    -------
    Return a vector of inference times for 10000 frames on one video or two videos:
    dlclive.benchmark_videos('/my/exported/model', 'my_video.avi', n_frames=10000)
    dlclive.benchmark_videos('/my/exported/model', ['my_video1.avi', 'my_video2.avi'], n_frames=10000)

    Return a vector of inference times, testing full size and resizing images to half the width and height for inference, for two videos
    dlclive.benchmark_videos('/my/exported/model', ['my_video1.avi', 'my_video2.avi'], n_frames=10000, resize=[1.0, 0.5])

    Display keypoints to check the accuracy of an exported model
    dlclive.benchmark_videos('/my/exported/model', 'my_video.avi', display=True)

    Analyze a video (save poses to hdf5) and create a labeled video, similar to :function:`DeepLabCut.benchmark_videos` and :function:`create_labeled_video`
    dlclive.benchmark_videos('/my/exported/model', 'my_video.avi', save_poses=True, save_video=True)
    """

    # convert video_paths to list

    video_path = video_path if type(video_path) is list else [video_path]

    # fix resize

    if pixels:
        pixels = pixels if type(pixels) is list else [pixels]
        resize = [None for p in pixels]
    elif resize:
        resize = resize if type(resize) is list else [resize]
        pixels = [None for r in resize]
    else:
        resize = [None]
        pixels = [None]

    # loop over videos

    for v in video_path:

        # initialize full inference times

        inf_times = []
        im_size_out = []

        for i in range(len(resize)):

            print(f"\nRun {i+1} / {len(resize)}\n")

            this_inf_times, this_im_size, TFGPUinference, meta = benchmark(
                model_path,
                v,
                tf_config=tf_config,
                resize=resize[i],
                pixels=pixels[i],
                cropping=cropping,
                dynamic=dynamic,
                n_frames=n_frames,
                print_rate=print_rate,
                display=display,
                pcutoff=pcutoff,
                display_radius=display_radius,
                cmap=cmap,
                save_poses=save_poses,
                save_video=save_video,
                output=output,
            )

            inf_times.append(this_inf_times)
            im_size_out.append(this_im_size)

        inf_times = np.array(inf_times)
        im_size_out = np.array(im_size_out)

        # save results

        if output is not None:
            sys_info = get_system_info()
            save_inf_times(
                sys_info,
                inf_times,
                im_size_out,
                TFGPUinference,
                model=os.path.basename(model_path),
                meta=meta,
                output=output,
            )


def main():
    """Provides a command line interface :function:`benchmark_videos`
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("video_path", type=str, nargs="+")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-n", "--n-frames", type=int, default=1000)
    parser.add_argument("-r", "--resize", type=float, nargs="+")
    parser.add_argument("-p", "--pixels", type=float, nargs="+")
    parser.add_argument("-v", "--print-rate", default=False, action="store_true")
    parser.add_argument("-d", "--display", default=False, action="store_true")
    parser.add_argument("-l", "--pcutoff", default=0.5, type=float)
    parser.add_argument("-s", "--display-radius", default=3, type=int)
    parser.add_argument("-c", "--cmap", type=str, default="bmy")
    parser.add_argument("--cropping", nargs="+", type=int, default=None)
    parser.add_argument("--dynamic", nargs="+", type=float, default=[])
    parser.add_argument("--save-poses", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    if (args.cropping) and (len(args.cropping) < 4):
        raise Exception(
            "Cropping not properly specified. Must provide 4 values: x1, x2, y1, y2"
        )

    if not args.dynamic:
        args.dynamic = (False, 0.5, 10)
    elif len(args.dynamic) < 3:
        raise Exception(
            "Dynamic cropping not properly specified. Must provide three values: 0 or 1 as boolean flag, pcutoff, and margin"
        )
    else:
        args.dynamic = (bool(args.dynamic[0]), args.dynamic[1], args.dynamic[2])

    benchmark_videos(
        args.model_path,
        args.video_path,
        output=args.output,
        resize=args.resize,
        pixels=args.pixels,
        cropping=args.cropping,
        dynamic=args.dynamic,
        n_frames=args.n_frames,
        print_rate=args.print_rate,
        display=args.display,
        pcutoff=args.pcutoff,
        display_radius=args.display_radius,
        cmap=args.cmap,
        save_poses=args.save_poses,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
