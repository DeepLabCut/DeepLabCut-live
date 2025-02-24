"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import glob
import os
import time
import typing
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import deeplabcut as dlc
import numpy as np
import onnx
import onnxruntime as ort
import ruamel.yaml
import torch
from deeplabcut.pose_estimation_pytorch.models import PoseModel

from dlclive import utils
from dlclive.display import Display
from dlclive.exceptions import DLCLiveError, DLCLiveWarning
from dlclive.pose import (argmax_pose_predict, extract_cnn_output,
                          multi_pose_predict)
from dlclive.predictor import HeatmapPredictor

if typing.TYPE_CHECKING:
    from dlclive.processor import Processor


# TODO:
# graph.py the main element to import TF model - convert to pytorch implementation
# add pcutoffn to docstring

# Q:     What is the best way to test the code as we go?
# Q: if self.pose is not None: - ask Niels to go through this!

# Q: what exactly does model_type reference?
# Q: is precision a type of qunatization?
# Q: for dynamic: First key points are predicted, then dynamic cropping is performed to 'single out' the animal, and then pose is estimated, we think. What is the difference from key point prediction to pose prediction?
# Q: what is the processor? see processor code F12 from init file - what is the 'user defined process' - could it be that if mouse = standing, perform some action? or is the process the prediction of a certain pose/set of keypoints
# Q: why have the convert2rgb function, is the stream coming from the camera different from the input needed to DLC live?
# Q: what is the parameter 'cfg'?

# What do these do?
#        self.inputs = None
#        self.outputs = None
#        self.tflite_interpreter = None
#        self.pose = None
#        self.is_initialized = False
#        self.sess = None


class DLCLive(object):
    """
    Object that loads a DLC network and performs inference on single images (e.g. images captured from a camera feed)

    Parameters
    -----------

    path : string
        Full path to exported model directory

    model_type: string, optional
        which model to use: 'base', 'tensorrt' for tensorrt optimized graph, 'lite' for tensorflow lite optimized graph

    precision : string, optional
        precision of model weights, only for model_type='tensorrt'. Can be 'FP16' (default), 'FP32', or 'INT8'

    cropping : list of int
        cropping parameters in pixel number: [x1, x2, y1, y2] #A: Maybe this is the dynamic cropping of each frame to speed of processing, so instead of analyzing the whole frame, it analyses only the part of the frame where the animal is

    dynamic: triple containing (state, detectiontreshold, margin) #A: margin adds some space so the 'bbox' isn't too narrow around the animal'. First key points are predicted, then dynamic cropping is performed to 'single out' the animal, and then pose is estimated, we think.
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal).

    resize : float, optional
        Factor to resize the image.
        For example, resize=0.5 will downsize both the height and width of the image by a factor of 2.

    processor: dlc pose processor object, optional #A: this is possibly the 'predictor' - or is it what enables use on jetson boards?
        User-defined processor object. Must contain two methods: process and save.
        The 'process' method takes in a pose, performs some processing, and returns processed pose.
        The 'save' method saves any valuable data created by or used by the processor
        Processors can be used for two main purposes:
        i) to run a forward predicting model that will predict the future pose from past history of poses (history can be stored in the processor object, but is not stored in this DLCLive object)
        ii) to trigger external hardware based on pose estimation (e.g. see 'TeensyLaser' processor)

    convert2rgb : bool, optional
        boolean flag to convert frames from BGR to RGB color scheme

    display : bool, optional
        Display frames with DeepLabCut labels?
        This is useful for testing model accuracy and cropping parameters, but it is very slow.

    display_lik : float, optional
        Likelihood threshold for display

    display_raidus : int, optional
        radius for keypoint display in pixels, default=3
    """

    PARAMETERS = (
        "path",
        "cfg",
        "model_type",
        "precision",
        "cropping",
        "dynamic",
        "resize",
        "processor",
    )

    def __init__(
        self,
        path: str,
        snapshot: str = None,
        model_type: str = "onnx",
        precision: str = "FP32",
        device: str = "cpu",
        cropping: Optional[List[int]] = None,
        dynamic: Tuple[bool, float, float] = (False, 0.5, 10),
        resize: Optional[float] = None,
        convert2rgb: bool = True,
        processor: Optional["Processor"] = None,
        display: typing.Union[bool, Display] = False,
        pcutoff: float = 0.5,
        display_radius: int = 3,
        display_cmap: str = "bmy",
    ):

        self.path = path
        self.model_type = model_type
        self.device = device
        self.snapshot = snapshot
        self.precision = precision
        self.cropping = cropping
        self.dynamic = dynamic
        self.dynamic_cropping = None
        self.resize = resize
        self.processor = processor
        self.convert2rgb = convert2rgb
        if isinstance(display, Display):
            self.display = display
        elif display:
            self.display = Display(
                pcutoff=pcutoff, radius=display_radius, cmap=display_cmap
            )
        else:
            self.display = None

        self.cfg = None
        self.cfg_path = None
        self.sess = None
        self.pose_model = None
        self.predictor = None
        self.pose = None

        if self.model_type == "pytorch" and (self.snapshot) is None:
            raise DLCLiveError(
                f"The selected model type is '{self.model_type}', but no snapshot was provided"
            )
        if self.model_type == "pytorch" and (self.device) == "tensorrt":
            raise DLCLiveError(
                f"The selected model type is '{self.model_type}' is not enabled by the selected runtime {self.device}"
            )
        self.read_config()

    def read_config(self):
        """Reads configuration yaml file

        Raises
        ------
        FileNotFoundError
            error thrown if pose configuration file does nott exist
        """

        cfg_path = Path(self.path).resolve() / "pytorch_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"The pose configuration file for the exported model at {str(cfg_path)} was not found. Please check the path to the exported model directory"
            )

        ruamel_file = ruamel.yaml.YAML()
        self.cfg = ruamel_file.load(open(str(cfg_path), "r"))

    @property
    def parameterization(
        self,
    ) -> (
        dict
    ):  # A: constructs a dictionary based on the object attributes based on the list of parameters
        """
        Return
        Returns
        -------
        """
        return {param: getattr(self, param) for param in self.PARAMETERS}

    def process_frame(self, frame):
        """
        Crops an image according to the object's cropping and dynamic properties.

        Parameters
        -----------
        frame :class:`numpy.ndarray`
            image as a numpy array

        Returns
        ----------
        frame :class:`numpy.ndarray`
            processed frame: convert type, crop, convert color
        """

        # ! NORMALISATION ??

        if self.cropping:
            frame = frame[
                self.cropping[2] : self.cropping[3], self.cropping[0] : self.cropping[1]
            ]
        if self.dynamic[0]:

            if self.pose is not None:
                detected = self.pose["poses"][0][0][:, 2] > self.dynamic[1]

                # if np.any(detected.numpy()):
                if torch.any(detected):
                    # if detected.any():  # Use PyTorch's any() method

                    x = self.pose["poses"][0][0][detected, 0]
                    y = self.pose["poses"][0][0][detected, 1]

                    x1 = int(max([0, int(torch.amin(x)) - self.dynamic[2]]))
                    x2 = int(
                        min([frame.shape[1], int(torch.amax(x)) + self.dynamic[2]])
                    )
                    y1 = int(max([0, int(torch.amin(y)) - self.dynamic[2]]))
                    y2 = int(
                        min([frame.shape[0], int(torch.amax(y)) + self.dynamic[2]])
                    )
                    self.dynamic_cropping = [x1, x2, y1, y2]

                    frame = frame[y1:y2, x1:x2]

                else:

                    self.dynamic_cropping = None

        if self.resize != 1:
            frame = utils.resize_frame(frame, self.resize)

        if self.convert2rgb:
            frame = utils.img_to_rgb(frame)

        return frame

    def load_model(self):
        if self.model_type == "pytorch":
            # Requires DLC 3.0 to be imported
            model_path = os.path.join(self.path, self.snapshot)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    "The model file {} does not exist.".format(model_path)
                )
            weights = torch.load(model_path, map_location=torch.device(self.device))
            self.pose_model = PoseModel.build(self.cfg["model"])
            self.pose_model.load_state_dict(weights["model"])
            self.pose_model = self.pose_model.to(self.device)
            self.pose_model.eval()

        elif self.model_type == "onnx":
            model_paths = glob.glob(os.path.normpath(self.path + "/*.onnx"))
            if self.precision == "FP16":
                model_path = [
                    model_paths[i]
                    for i in range(len(model_paths))
                    if "fp16" in model_paths[i]
                ][0]
                print(model_path)
            else:
                model_path = model_paths[0]
            opts = ort.SessionOptions()
            opts.enable_profiling = False
            if self.device == "cuda":
                self.sess = ort.InferenceSession(
                    model_path, opts, providers=["CUDAExecutionProvider"]
                )
                print(self.sess)
            elif self.device == "cpu":
                self.sess = ort.InferenceSession(
                    model_path, opts, providers=["CPUExecutionProvider"]
                )
            # ! TODO implement if statements for choice of tensorrt engine options (precision, and caching)
            elif self.device == "tensorrt":
                provider = [
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "./trt_engines",
                        },
                    )
                ]
                self.sess = ort.InferenceSession(model_path, opts, providers=provider)
            self.predictor = HeatmapPredictor.build(self.cfg)

            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    "The model file {} does not exist.".format(model_path)
                )

        else:
            raise DLCLiveError(
                "model_type = {} is not supported. model_type must be 'pytorch' or 'onnx'".format(
                    self.model_type
                )
            )

    def init_inference(self, frame=None, **kwargs):
        """
        Load model and perform inference on first frame -- the first inference is usually very slow.

        Parameters
        -----------
        frame :class:`numpy.ndarray`
            image as a numpy array

        Returns
        --------
        pose :class:`numpy.ndarray`
            the pose estimated by DeepLabCut for the input image
        """

        # load model
        self.load_model()

        inf_time = 0.0
        # get pose of first frame (first inference is often very slow)
        if frame is not None:
            pose, inf_time = self.get_pose(frame, **kwargs)
        else:
            pose = None

        return pose, inf_time

    def get_pose(self, frame=None, **kwargs):
        """
        Get the pose of an image

        Parameters
        -----------
        frame :class:`numpy.ndarray`
            image as a numpy array

        Returns
        --------
        pose :class:`numpy.ndarray`
            the pose estimated by DeepLabCut for the input image
        """
        inf_time = 0.0
        if frame is None:
            raise DLCLiveError("No frame provided for live pose estimation")

        if frame is not None:
            if frame.ndim >= 2:
                self.convert2rgb = True

            processed_frame = self.process_frame(frame)

        if self.model_type == "pytorch":
            frame = torch.Tensor(processed_frame)
            frame = frame.permute(2, 0, 1).unsqueeze(0)
            frame = frame.to(self.device)

            with torch.no_grad():
                start = time.time()
                outputs = self.pose_model(frame)
                end = time.time()
                inf_time = end - start

            self.pose = self.pose_model.get_predictions(outputs)
            self.pose = self.pose["bodypart"]

        elif self.model_type == "onnx":
            if self.precision == "FP32":
                frame = processed_frame.astype(np.float32)
            elif self.precision == "FP16":
                frame = processed_frame.astype(np.float16)

            frame = np.transpose(frame, (2, 0, 1))
            frame = np.expand_dims(frame, axis=0)

            ort_inputs = {self.sess.get_inputs()[0].name: frame}

            start = time.time()
            outputs = self.sess.run(None, ort_inputs)
            end = time.time()
            inf_time = end - start

            outputs = {
                "heatmap": torch.Tensor(outputs[0]),
                "locref": torch.Tensor(outputs[1]),
            }

            self.pose = self.predictor(outputs=outputs)

        else:
            raise DLCLiveError(
                "model_type = {} is not supported. model_type must be 'pytorch' or 'onnx'".format(
                    self.model_type
                )
            )

        # display image if display=True before correcting pose for cropping/resizing

        if self.display is not None:
            self.display.display_frame(processed_frame, self.pose)

        # if frame is cropped, convert pose coordinates to original frame coordinates

        if self.resize is not None:
            self.pose["poses"][0][0][:, :2] *= 1 / self.resize

        if self.cropping is not None:
            self.pose["poses"][0][0][0] += self.cropping[0]
            self.pose["poses"][0][0][0] += self.cropping[2]

        if self.dynamic_cropping is not None:
            self.pose["poses"][0][0][:, 0] += self.dynamic_cropping[0]
            self.pose["poses"][0][0][:, 1] += self.dynamic_cropping[2]

        # process the pose

        if self.processor:
            self.pose = self.processor.process(self.pose, **kwargs)

        return self.pose, inf_time

    # def close(self):
    #     """ Close tensorflow session
    #     """

    #     self.sess.close()
    #     self.sess = None
    #     self.is_initialized = False
    #     if self.display is not None:
    #         self.display.destroy()
