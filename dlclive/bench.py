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
import subprocess
import typing
import warnings

try:
    from pip._internal.operations import freeze
except ImportError:
    from pip.operations import freeze
import cpuinfo
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import cv2

from dlclive import DLCLive
from dlclive import VERSION
from dlclive import __file__ as dlcfile

from dlclive.utils import decode_fourcc

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


    ### get os

    op_sys = platform.platform()
    host_name = platform.node().replace(' ', '')

    # A string giving the absolute path of the executable binary for the Python interpreter, on systems where this makes sense.
    if platform.system() == 'Windows':
        host_python = sys.executable.split(os.path.sep)[-2]
    else:
        host_python = sys.executable.split(os.path.sep)[-3]

    # try to get git hash if possible
    dlc_basedir = os.path.dirname(os.path.dirname(dlcfile))
    git_hash = None
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dlc_basedir)
        git_hash = git_hash.decode('utf-8').rstrip('\n')
    except subprocess.CalledProcessError:
        # not installed from git repo, eg. pypi
        # fine, pass quietly
        pass

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
        dev = get_cpu_info() #[get_cpu_info()['brand']]
        dev_type = "CPU"

    # return a dictionary rather than a tuple for inspectability's sake
    return {
        'host_name': host_name,
        'op_sys'   : op_sys,
        'python': host_python,
        'device_type': dev_type,
        'device': dev,
        'freeze': list(freeze.freeze()), # pip freeze to get versions of all packages
        'python_version': sys.version,
        'git_hash': git_hash,
        'dlclive_version': VERSION
    }

def run_benchmark(model_path, video_path, tf_config=None,
                  resize=None, pixels=None, n_frames=10000,
                  print_rate=False, display=False, pcutoff=0.0,
                  display_radius=3) -> typing.Tuple[np.ndarray, int, bool, dict]:
    """ Benchmark on inference times for a given DLC model and video

    Parameters
    ----------
    model_path : str
        path to exported DeepLabCut model
    video_path : str
        path to video file
    resize : int, optional
        resize factor. Can only use one of resize or pixels. If both are provided, will use pixels. by default None
    pixels : int, optional
        downsize image to this number of pixels, maintaining aspect ratio. Can only use one of resize or pixels. If both are provided, will use pixels. by default None
    n_frames : int, optional
        number of frames to run inference on, by default 10000
    print_rate : bool, optional
        flat to print inference rate frame by frame, by default False

    Returns
    -------
    :class:`numpy.ndarray`
        vector of inference times
    float
        number of pixels in each image
    """

    ### load video

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    im_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### get resize factor

    if pixels is not None:
        resize = np.sqrt(pixels / (im_size[0] * im_size[1]))
    else:
        resize = resize if resize is not None else 1

    ### initialize live object

    live = DLCLive(model_path, tf_config=tf_config, resize=resize, display=display, pcutoff=pcutoff, display_radius=display_radius)
    live.init_inference(frame)
    TFGPUinference = True if len(live.outputs) == 1 else False

    ### perform inference

    iterator = range(n_frames) if (print_rate) or (display) else tqdm(range(n_frames))
    inf_times = np.zeros(n_frames)

    for i in iterator:

        ret, frame = cap.read()

        if not ret:
            warnings.warn("Did not complete {:d} frames. There probably were not enough frames in the video {}.".format(n_frames, video_path))
            break
        
        start_pose = time.time()
        live.get_pose(frame)
        inf_times[i] = time.time() - start_pose

        if print_rate:
            print("pose rate = {:d}".format(int(1 / inf_times[i])))

    if print_rate:
        print("mean pose rate = {:d}".format(int(np.mean(1/inf_times))))

    ### close video and tensorflow session

    # gather video and test parameterization

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



    meta = {
        'video_path': video_path,
        'video_codec': fourcc,
        'video_pixel_format': pix_fmt,
        'video_fps': fps,
        'video_total_frames': frame_count,
        'resize': resize,
        'original_frame_size': im_size,
        'resized_frame_size': (im_size[0]*resize, im_size[1]*resize),
        'pixels': pixels,
        'dlclive_params': live.parameterization
    }

    cap.release()
    live.close()

    return inf_times, resize*im_size[0] * resize*im_size[1], TFGPUinference, meta

def get_savebenchmarkfn(sys_info ,i, fn_ind, out_dir=None):
    ''' get filename to save data (definitions see save_benchmark)'''
    out_dir = out_dir if out_dir is not None else os.getcwd()
    base_name = "benchmark_{}_{}_{}_{}.pickle".format(sys_info['host_name'], sys_info['device_type'], fn_ind, i)
    datafilename = out_dir + '/' + base_name
    return datafilename

def save_benchmark(sys_info: dict,
                   inf_times: np.ndarray,
                   pixels: typing.Union[np.ndarray, float],
                   iter: int,
                   TFGPUinference: bool = None,
                   model: str = None,
                   out_dir: str = None,
                   meta: dict=None):
    """ Save benchmarking data with system information to a pickle file

    Parameters
    ----------
    sys_info : dict
        system information generated by :func:`get_system_info`
    inf_times : :class:`numpy.ndarray`
        array of inference times generated by :func:`run_benchmark`
    pixels : float or :class:`numpy.ndarray`
        number of pixels for each benchmark run. If an array, each index corresponds to a row in inf_times
    i: integer
        number of the specific instance of experiment (so every part is saved individually)
    TFGPUinference: bool
        flag if using tensorflow inference or numpy inference DLC model
    model: str, optional
        name of model
    out_dir : str, optional
        path to directory to save data. If None, uses pwd, by default None
    meta: dict, optional
        metadata returned form run_benchmark

    Returns
    -------
    bool
        flag indicating successful save
    """

    out_dir = out_dir if out_dir is not None else os.getcwd()

    model_type = None
    if model is not None:
        if 'resnet' in model:
            model_type = 'resnet'
        elif 'mobilenet' in model:
            model_type = 'mobilenet'
        else:
            model_type = None

    fn_ind = 0
    base_name = "benchmark_{}_{}_{}_{}.pickle".format(sys_info['host_name'],
                                                      sys_info['device'][0],
                                                      fn_ind,
                                                      iter)
    while os.path.isfile(os.path.normpath(out_dir + '/' + base_name)):
        fn_ind += 1
        base_name = "benchmark_{}_{}_{}_{}.pickle".format(sys_info['host_name'],
                                                          sys_info['device'][0],
                                                          fn_ind,
                                                          iter)

    data = {'model': model,
            'model_type': model_type,
            'TFGPUinference': TFGPUinference,
            'pixels': pixels,
            'inference_times': inf_times}

    data.update(sys_info)

    if meta:
        data.update(meta)

    data.update(sys_info)

    datafilename = os.path.normpath(f"{out_dir}/{base_name}")
    pickle.dump(data, open(os.path.normpath(datafilename), 'wb'))

    return True

def read_pickle(filename):
    """ Read the pickle file """
    with open(filename, "rb") as handle:
        return pickle.load(handle)

def benchmark_model_by_size(model_path, video_path, output=None, n_frames=10000, tf_config=None, resize=None, pixels=None, print_rate=False, display=False, pcutoff=0.5, display_radius=3):
    """Benchmark DLC model by image size

    Parameters
    ----------
    model_path : str
        path to exported DLC model
    video_path : str
        path to video file
    fn_ind : integer
        auxiliary variable for creating a unique identifier for saving the models
    out_dir : str, optional
        directory to save data, will not save data if None, by default None
    n_frames : int, optional
        number of frames to run, by default 10000
    resize : list, optional
        list of resize factors (as floats), can only use one of resize or pixels, if both specified will use pixels, by default None
    pixels : list, optional
        list of pixel image sizes (as ints), can only use one of resize or pixels, if both specified will use pixels, by default None
    print_rate : bool, optional
        flag to print frame by frame inference rate, by default False

    Example
    --------
    Linux/MacOs
    dlclive.bench.benchmark_model_by_size('/path/to/pbfiles/', '/pathto/video.mp4', n_frames=10000, print_rate=True)
    """

    ### fix resize

    if pixels:
        resize = [None for p in pixels]
    elif resize:
        pixels = [None for r in resize]
    else:
        resize = [None]
        pixels = [None]

    ### initialize full inference times

    # get system info once, shouldn't change between runs
    sys_info = get_system_info()

    for i in range(len(resize)):

        print("\nRun {:d} / {:d}\n".format(i+1, len(resize)))

        inf_times, pixels_out, TFGPUinference, benchmark_meta = run_benchmark(
            model_path,
            video_path,
            tf_config=tf_config,
            resize=resize[i],
            pixels=pixels[i],
            n_frames=n_frames,
            print_rate=print_rate,
            display=display,
            pcutoff=pcutoff,
            display_radius=display_radius)

        #TODO: check if a part has already been complted?

        ### saving results intermediately
        save_benchmark(sys_info, inf_times, pixels_out, i, TFGPUinference,
                       model=os.path.basename(model_path),
                       out_dir = output,
                       meta=benchmark_meta)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('video_path', type=str)
    parser.add_argument('-o', '--output', type=str, default=os.getcwd())
    parser.add_argument('-n', '--n-frames', type=int, default=10000)
    parser.add_argument('-r', '--resize', type=float, nargs='+')
    parser.add_argument('-p', '--pixels', type=float, nargs='+')
    parser.add_argument('-v', '--print_rate', default=False, action='store_true')
    parser.add_argument('-d', '--display', default=False, action='store_true')
    parser.add_argument('-l', '--pcutoff', default=0.5, type=float)
    parser.add_argument('-s', '--display-radius', default=3, type=int)
    args = parser.parse_args()


    benchmark_model_by_size(args.model_path,
                            args.video_path,
                            output=args.output,
                            resize=args.resize,
                            pixels=args.pixels,
                            n_frames=args.n_frames,
                            print_rate=args.print_rate,
                            display=args.display,
                            pcutoff=args.pcutoff,
                            display_radius=args.display_radius)


if __name__ == "__main__":
    main()
