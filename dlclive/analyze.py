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
import argparse
import pickle
import colorcet as cc
from PIL import ImageColor
import ruamel
import pandas as pd

import cpuinfo
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2

from dlclive import DLCLive


def get_system_info():
    """ Return summary info for system running benchmark

    Returns
    -------
    str
        name of machine
    str
        operating system
    str
        path to python (which conda/virtual environment)
    str
        device name
    """


    ### get os

    op_sys = platform.platform()
    host_name = platform.node()
    if platform.system() == 'Windows':
        host_python = sys.executable.split(os.path.sep)[-2]
    else:
        host_python = sys.executable.split(os.path.sep)[-3]

    ### get device info (GPU or CPU)

    dev = None
    if tf.test.is_gpu_available():
        gpu_name = tf.test.gpu_device_name()
        from tensorflow.python.client import device_lib
        dev_desc = [d.physical_device_desc for d in device_lib.list_local_devices() if d.name == gpu_name]
        dev = [d.split(",")[1].split(':')[1].strip() for d in dev_desc]
        dev_type = "GPU"
    else:
        from cpuinfo import get_cpu_info
        dev = [get_cpu_info()['brand']]
        dev_type = "CPU"

    return host_name, op_sys, host_python, (dev_type, dev)


def analyze(model_path,
            video_path,
            tf_config=None,
            resize=None,
            pixels=None,
            n_frames=1000,
            print_rate=False,
            display=False,
            display_lik=0.0,
            display_radius=3,
            cmap='bmy',
            save_poses=False,
            save_video=False,
            output=None):
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
    n_frames : int, optional
        number of frames to run inference on, by default 1000
    print_rate : bool, optional
        flat to print inference rate frame by frame, by default False
    display : bool, optional
        flag to display keypoints on images. Useful for checking the accuracy of exported models.
    display_lik : float, optional
        likelihood threshold to display keypoints
    display_radius : int, optional
        size (radius in pixels) of keypoint to display
    cmap : str, optional
        a string indicating the :package:`colorcet` colormap, `options here <https://colorcet.holoviz.org/>`, by default "bmy"
    save_poses : bool, optional
        flag to save poses to an hdf5 file. If True, operates similar to :function:`DeepLabCut.analyze_videos`, by default False
    save_video : bool, optional
        flag to save a labeled video. If True, operates similar to :function:`DeepLabCut.create_labeled_video`, by default False
    output : str, optional
        path to directory to save pose and/or video file. If not specified, will use the directory of video_path, by default None

    Returns
    -------
    :class:`numpy.ndarray`
        vector of inference times
    float
        number of pixels in each image

    Example
    -------
    Return a vector of inference times for 10000 frames:
    dlclive.analyze('/my/exported/model', 'my_video.avi', n_frames=10000)

    Return a vector of inference times, resizing images to half the width and height for inference
    dlclive.analyze('/my/exported/model', 'my_video.avi', n_frames=10000, resize=0.5)

    Display keypoints to check the accuracy of an exported model
    dlclive.analyze('/my/exported/model', 'my_video.avi', display=True)

    Analyze a video (save poses to hdf5) and create a labeled video, similar to :function:`DeepLabCut.analyze_videos` and :function:`create_labeled_video`
    dlclive.analyze('/my/exported/model', 'my_video.avi', save_poses=True, save_video=True)
    """

    ### load video

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    n_frames = n_frames if n_frames < cap.get(cv2.CAP_PROP_FRAME_COUNT)-1 else cap.get(cv2.CAP_PROP_FRAME_COUNT)-1
    n_frames = int(n_frames)
    im_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    ### get resize factor

    if pixels is not None:
        resize = np.sqrt(pixels / (im_size[0] * im_size[1]))
    if resize is not None:
        im_size = (int(im_size[0]*resize), int(im_size[1]*resize))

    ### initialize live object

    live = DLCLive(model_path, tf_config=tf_config, resize=resize, display=display, display_lik=display_lik, display_radius=display_radius)
    live.init_inference(frame)
    TFGPUinference = True if len(live.outputs) == 1 else False

    ### create video writer

    if save_video:
        colors = None
        out_dir = output if output is not None else os.path.dirname(os.path.realpath(video_path))
        out_vid_base = os.path.basename(video_path)
        out_vid_file = os.path.normpath(f"{out_dir}/{os.path.splitext(video_path)[0]}_DLCLIVE_LABELED.avi")
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = cap.get(cv2.CAP_PROP_FPS)
        vwriter = cv2.VideoWriter(out_vid_file, fourcc, fps, im_size)

    ### perform inference

    iterator = range(n_frames) if (print_rate) or (display) else tqdm(range(n_frames))
    inf_times = np.zeros(n_frames)
    poses = []

    for i in iterator:

        ret, frame = cap.read()

        if not ret:
            warnings.warn("Did not complete {:d} frames. There probably were not enough frames in the video {}.".format(n_frames, video_path))
            break
        
        start_pose = time.time()
        poses.append(live.get_pose(frame))
        inf_times[i] = time.time() - start_pose

        if save_video:

            if colors is None:
                all_colors = getattr(cc, cmap)
                colors = [ImageColor.getcolor(c, "RGB")[::-1] for c in all_colors[::int(len(all_colors)/poses[-1].shape[0])]]

            this_pose = poses[-1]
            for j in range(this_pose.shape[0]):
                if this_pose[j, 2] > display_lik:
                    x = int(this_pose[j, 0])
                    y = int(this_pose[j, 1])
                    frame = cv2.circle(frame, (x, y), display_radius, colors[j], thickness=-1)
            
            if resize is not None:
                frame = cv2.resize(frame, im_size)
            vwriter.write(frame)
        
        if print_rate:
            print("pose rate = {:d}".format(int(1 / inf_times[i])))

    if print_rate:
        print("mean pose rate = {:d}".format(int(np.mean(1/inf_times))))

    ### close video and tensorflow session

    cap.release()
    live.close()

    if save_video:
        vwriter.release()

    if save_poses:

        cfg_path = os.path.normpath(f"{model_path}/pose_cfg.yaml")
        ruamel_file = ruamel.yaml.YAML()
        dlc_cfg = ruamel_file.load(open(cfg_path, 'r'))        
        bodyparts = dlc_cfg['all_joints_names']
        poses = np.array(poses)
        poses = poses.reshape((poses.shape[0], poses.shape[1]*poses.shape[2]))
        pdindex = pd.MultiIndex.from_product([bodyparts, ['x', 'y', 'likelihood']], names=['bodyparts', 'coords'])
        pose_df = pd.DataFrame(poses, columns=pdindex)

        out_dir = output if output is not None else os.path.dirname(os.path.realpath(video_path))
        out_vid_base = os.path.basename(video_path)
        out_dlc_file = os.path.normpath(f"{out_dir}/{os.path.splitext(video_path)[0]}_DLCLIVE_POSES.h5")
        pose_df.to_hdf(out_dlc_file, key='df_with_missing', mode='w')

    return inf_times, im_size[0]*im_size[1], TFGPUinference


def save_inf_times(sys_info,
                   inf_times,
                   pixels,
                   TFGPUinference,
                   model=None,
                   output=None):
    """ Save inference time data collected using :function:`analyze` with system information to a pickle file.
    This is primarily used through :function:`analyze_videos`
    
    Parameters
    ----------
    sys_info : tuple
        system information generated by :func:`get_system_info`
    inf_times : :class:`numpy.ndarray`
        array of inference times generated by :func:`analyze`
    pixels : float or :class:`numpy.ndarray`
        number of pixels for each benchmark run. If an array, each index corresponds to a row in inf_times
    TFGPUinference: bool
        flag if using tensorflow inference or numpy inference DLC model
    model: str, optional
        name of model
    output : str, optional
        path to directory to save data. If None, uses pwd, by default None
    
    Returns
    -------
    bool
        flag indicating successful save
    """

    output = output if output is not None else os.getcwd()
    host_name, op_sys, host_python, dev = sys_info
    host_name = host_name.replace(" ", "")
    model_type = None
    if model is not None:
        if 'resnet' in model:
            model_type = 'resnet'
        elif 'mobilenet' in model:
            model_type = 'mobilenet'
        else:
            model_type = None

    fn_ind = 0
    base_name = "benchmark_{}_{}_{}.pickle".format(host_name, dev[0], fn_ind)
    while os.path.isfile(os.path.normpath(output + '/' + base_name)):
        fn_ind += 1
        base_name = "benchmark_{}_{}_{}.pickle".format(host_name, dev[0], fn_ind)

    fn = os.path.normpath(output)

    data = {'host_name' : host_name,
            'op_sys' : op_sys,
            'python' : host_python,
            'device_type' : dev[0],
            'device' : dev[1],
            'model' : model,
            'model_type' : model_type,
            'TFGPUinference' : TFGPUinference,
            'pixels' : pixels,
            'inference_times' : inf_times}

    os.makedirs(os.path.normpath(output), exist_ok=True)
    pickle.dump(data, open(os.path.normpath(f"{output}/{base_name}"), 'wb'))

    return True


def analyze_videos(model_path,
                   video_path,
                   output=None,
                   n_frames=1000,
                   tf_config=None,
                   resize=None,
                   pixels=None,
                   print_rate=False,
                   display=False,
                   display_lik=0.5,
                   display_radius=3,
                   save_poses=False,
                   save_video=False,
                   cmap='bmy'):
    """Analyze videos using DeepLabCut-live exported models. 
    Analyze multiple videos and/or multiple options for the size of the video 
    by specifying a resizing factor or the number of pixels to use in the image (keeping aspect ratio constant).
    Options to record inference times (to examine inference speed), 
    display keypoints to visually check the accuracy, 
    or save poses to an hdf5 file as in :function:`deeplabcut.analyze_videos` and 
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
    n_frames : int, optional
        number of frames to run inference on, by default 1000
    print_rate : bool, optional
        flat to print inference rate frame by frame, by default False
    display : bool, optional
        flag to display keypoints on images. Useful for checking the accuracy of exported models.
    display_lik : float, optional
        likelihood threshold to display keypoints
    display_radius : int, optional
        size (radius in pixels) of keypoint to display
    cmap : str, optional
        a string indicating the :package:`colorcet` colormap, `options here <https://colorcet.holoviz.org/>`, by default "bmy"
    save_poses : bool, optional
        flag to save poses to an hdf5 file. If True, operates similar to :function:`DeepLabCut.analyze_videos`, by default False
    save_video : bool, optional
        flag to save a labeled video. If True, operates similar to :function:`DeepLabCut.create_labeled_video`, by default False

    Example
    -------
    Return a vector of inference times for 10000 frames on one video or two videos:
    dlclive.analyze_videos('/my/exported/model', 'my_video.avi', n_frames=10000)
    dlclive.analyze_videos('/my/exported/model', ['my_video1.avi', 'my_video2.avi'], n_frames=10000)

    Return a vector of inference times, testing full size and resizing images to half the width and height for inference, for two videos
    dlclive.analyze_videos('/my/exported/model', ['my_video1.avi', 'my_video2.avi'], n_frames=10000, resize=[1.0, 0.5])

    Display keypoints to check the accuracy of an exported model
    dlclive.analyze_videos('/my/exported/model', 'my_video.avi', display=True)

    Analyze a video (save poses to hdf5) and create a labeled video, similar to :function:`DeepLabCut.analyze_videos` and :function:`create_labeled_video`
    dlclive.analyze_videos('/my/exported/model', 'my_video.avi', save_poses=True, save_video=True)
    """

    ### convert video_paths to list

    video_path = video_path if type(video_path) is list else [video_path]

    ### fix resize

    if pixels:
        resize = [None for p in pixels]
    elif resize:
        pixels = [None for r in resize]
    else:
        resize = [None]
        pixels = [None]

    ### loop over videos

    for v in video_path:

        ### initialize full inference times

        inf_times = np.zeros((len(resize), n_frames))
        pixels_out = np.zeros(len(resize))

        for i in range(len(resize)):

            print("\nRun {:d} / {:d}\n".format(i+1, len(resize)))

            inf_times[i], pixels_out[i], TFGPUinference = analyze(model_path,
                                                                  v,
                                                                  tf_config=tf_config,
                                                                  resize=resize[i],
                                                                  pixels=pixels[i],
                                                                  n_frames=n_frames,
                                                                  print_rate=print_rate,
                                                                  display=display,
                                                                  display_lik=display_lik,
                                                                  display_radius=display_radius,
                                                                  cmap=cmap,
                                                                  save_poses=save_poses,
                                                                  save_video=save_video,
                                                                  output=output)

        ### save results

        if output is not None:
            sys_info = get_system_info()
            save_inf_times(sys_info, inf_times, pixels_out, TFGPUinference, model=os.path.basename(model_path), output=output)


def main():
    """Provides a command line interface :function:`analyze_videos`
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('video_path', type=str, nargs='+')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-n', '--n-frames', type=int, default=10000)
    parser.add_argument('-r', '--resize', type=float, nargs='+')
    parser.add_argument('-p', '--pixels', type=float, nargs='+')
    parser.add_argument('-v', '--print-rate', default=False, action='store_true')
    parser.add_argument('-d', '--display', default=False, action='store_true')
    parser.add_argument('-l', '--display-lik', default=0.5, type=float)
    parser.add_argument('-s', '--display-radius', default=3, type=int)
    parser.add_argument('-c', '--cmap', type=str, default='bmy')
    parser.add_argument('--save-poses', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    args = parser.parse_args()


    analyze_videos(args.model_path,
                   args.video_path,
                   output=args.output,
                   resize=args.resize,
                   pixels=args.pixels,
                   n_frames=args.n_frames,
                   print_rate=args.print_rate,
                   display=args.display,
                   display_lik=args.display_lik,
                   display_radius=args.display_radius,
                   cmap=args.cmap,
                   save_poses=args.save_poses,
                   save_video=args.save_video)


if __name__ == "__main__":
    main()
