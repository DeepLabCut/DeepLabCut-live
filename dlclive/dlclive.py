"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import glob
import os

# import tensorflow as tf
import typing
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import deeplabcut as dlc
import numpy as np
import ruamel.yaml
import torch
from deeplabcut.pose_estimation_pytorch.models import PoseModel

from dlclive import utils
from dlclive.display import Display
from dlclive.exceptions import DLCLiveError, DLCLiveWarning
from dlclive.pose import argmax_pose_predict, extract_cnn_output, multi_pose_predict

# try:
#     TFVER = [int(v) for v in tf.__version__.split(".")]
#     if TFVER[1] < 14:
#         from tensorflow.contrib.tensorrt import trt_convert as trt
#     else:
#         from tensorflow.python.compiler.tensorrt import trt_convert as trt
# except Exception:
#     pass

# from dlclive.graph import (
#     read_graph,
#     finalize_graph,
#     get_output_nodes,
#     get_output_tensors,
#     extract_graph,
# )


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
        model_path: str = None,
        model_type: str = "base",
        precision: str = "FP32",
        tf_config=None,
        pytorch_cfg=str,
        snapshot=str,
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

        self.path = model_path
        self.cfg = None  # type: typing.Optional[dict]
        self.model_type = model_type
        self.tf_config = tf_config
        self.pytorch_cfg = pytorch_cfg
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

        self.sess = None
        self.inputs = None
        self.outputs = None
        self.tflite_interpreter = None
        self.pose = None
        self.is_initialized = False

        # checks

        # if self.model_type == "tflite" and self.dynamic[0]:
        #     self.dynamic = (False, *self.dynamic[1:])
        #     warnings.warn(
        #         "Dynamic cropping is not supported for tensorflow lite inference. Dynamic cropping will not be used...",
        #         DLCLiveWarning,
        #     )

        self.read_config()

    def read_config(self):
        """Reads configuration yaml file

        Raises
        ------
        FileNotFoundError
            error thrown if pose configuration file does nott exist
        """

        cfg_path = (
            Path(self.pytorch_cfg).resolve() / "pytorch_config.yaml"
        )  # TODO TF_ref - replace by pytorch config - consider importing read_config function from DLC 3 - and the new config may have both detector and 'normal' config - e.g batch size could refer both to detector and key points. should be handled in the read_config from DLC3
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

    def process_frame(self, frame):  #'self' holds all the arguments
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

        if frame.dtype != np.uint8:

            frame = utils.convert_to_ubyte(frame)

        if self.cropping:  # if cropping is specified, it will be applied

            frame = frame[  # A: this produces a cropped image based on incoming coordinates x1,x2,y1,y2
                self.cropping[2] : self.cropping[3], self.cropping[0] : self.cropping[1]
            ]

        if self.dynamic[
            0
        ]:  # to go through this if statement, the boolean would have to be = True. for it to react to false you'd have to write if not self.dynamic[0]

            if self.pose is not None:

                detected = self.pose[:, 2] > self.dynamic[1]

                if np.any(detected):

                    x = self.pose[detected, 0]
                    y = self.pose[detected, 1]

                    x1 = int(
                        max([0, int(np.amin(x)) - self.dynamic[2]])
                    )  # We think it is dtected if keypoint likelihood exceeds the dynamic threshold for dynamic cropping
                    x2 = int(min([frame.shape[1], int(np.amax(x)) + self.dynamic[2]]))
                    y1 = int(max([0, int(np.amin(y)) - self.dynamic[2]]))
                    y2 = int(min([frame.shape[0], int(np.amax(y)) + self.dynamic[2]]))
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
        self.read_config()
        weights = torch.load(
            self.snapshot, map_location=torch.device("cpu")
        )  # added this to run on CPU
        print("Loaded weights")
        print(self.cfg)
        pose_model = PoseModel.build(self.cfg["model"])
        print("Built pose model")
        pose_model.load_state_dict(weights["model"])
        print("Loaded pretrained weights")
        return pose_model

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

        # get model file

        model_file = glob.glob(os.path.normpath(self.pytorch_cfg + "/*.pt"))[
            0
        ]  # TODO TF_ref - maybe .pb format will be changed when using pytorch
        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                "The model file {} does not exist.".format(model_file)
            )

        # process frame

        # ! TODO replace this if statement
        # if frame is None and (self.model_type == "tflite"):
        #     raise DLCLiveError(
        #         "No image was passed to initialize inference. An image must be passed to the init_inference method"
        #     )

        if frame is not None:
            if frame.ndim == 2:
                self.convert2rgb = True
            processed_frame = self.process_frame(frame)

        # load model

        # if self.model_type == "base":

        #     graph_def = read_graph(model_file)
        #     graph = finalize_graph(graph_def)
        #     self.sess, self.inputs, self.outputs = extract_graph(
        #         graph, tf_config=self.tf_config
        #     )

        # elif self.model_type == "tflite":

        #     ###
        #     # the frame size needed to initialize the tflite model as
        #     # tflite does not support saving a model with dynamic input size
        #     ###

        #     # get input and output tensor names from graph_def
        #     graph_def = read_graph(model_file)
        #     graph = finalize_graph(graph_def)
        #     output_nodes = get_output_nodes(graph)
        #     output_nodes = [on.replace("DLC/", "") for on in output_nodes]

        #     tf_version_2 = tf.__version__[0] == '2'

        #     if tf_version_2:
        #         converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        #             model_file,
        #             ["Placeholder"],
        #             output_nodes,
        #             input_shapes={"Placeholder": [1, processed_frame.shape[0], processed_frame.shape[1], 3]},
        #         )
        #     else:
        #         converter = tf.lite.TFLiteConverter.from_frozen_graph(
        #             model_file,
        #             ["Placeholder"],
        #             output_nodes,
        #             input_shapes={"Placeholder": [1, processed_frame.shape[0], processed_frame.shape[1], 3]},
        #         )

        #     try:
        #         tflite_model = converter.convert()
        #     except Exception:
        #         raise DLCLiveError(
        #             (
        #                 "This model cannot be converted to tensorflow lite format. "
        #                 "To use tensorflow lite for live inference, "
        #                 "make sure to set TFGPUinference=False "
        #                 "when exporting the model from DeepLabCut"
        #             )
        #         )

        #     self.tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
        #     self.tflite_interpreter.allocate_tensors()
        #     self.inputs = self.tflite_interpreter.get_input_details()
        #     self.outputs = self.tflite_interpreter.get_output_details()

        # elif self.model_type == "tensorrt":

        #     graph_def = read_graph(model_file)
        #     graph = finalize_graph(graph_def)
        #     output_tensors = get_output_tensors(graph)
        #     output_tensors = [ot.replace("DLC/", "") for ot in output_tensors]

        #     if (TFVER[0] > 1) | (TFVER[0] == 1 & TFVER[1] >= 14):
        #         converter = trt.TrtGraphConverter(
        #             input_graph_def=graph_def,
        #             nodes_blacklist=output_tensors,
        #             is_dynamic_op=True,
        #         )
        #         graph_def = converter.convert()
        #     else:
        #         graph_def = trt.create_inference_graph(
        #             input_graph_def=graph_def,
        #             outputs=output_tensors,
        #             max_batch_size=1,
        #             precision_mode=self.precision,
        #             is_dynamic_op=True,
        #         )

        #     graph = finalize_graph(graph_def)
        #     self.sess, self.inputs, self.outputs = extract_graph(
        #         graph, tf_config=self.tf_config
        #     )

        # else:

        #     raise DLCLiveError(
        #         "model_type = {} is not supported. model_type must be 'base', 'tflite', or 'tensorrt'".format(
        #             self.model_type
        #         )
        #     )

        # get pose of first frame (first inference is often very slow)

        if frame is not None:
            pose = self.get_pose(frame, **kwargs)
        else:
            pose = None

        self.is_initialized = True

        return pose

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

        if frame is None:
            raise DLCLiveError("No frame provided for live pose estimation")

        frame = self.process_frame(frame)

        # if self.model_type in ["base", "tensorrt"]:

        #     pose_output = self.sess.run(
        #         self.outputs, feed_dict={self.inputs: np.expand_dims(frame, axis=0)}
        #     )

        # elif self.model_type == "tflite":

        #     self.tflite_interpreter.set_tensor(
        #         self.inputs[0]["index"],
        #         np.expand_dims(frame, axis=0).astype(np.float32),
        #     )
        #     self.tflite_interpreter.invoke()

        #     if len(self.outputs) > 1:
        #         pose_output = [
        #             self.tflite_interpreter.get_tensor(self.outputs[0]["index"]),
        #             self.tflite_interpreter.get_tensor(self.outputs[1]["index"]),
        #         ]
        #     else:
        #         pose_output = self.tflite_interpreter.get_tensor(
        #             self.outputs[0]["index"]
        #         )

        # else:

        #     raise DLCLiveError(
        #         "model_type = {} is not supported. model_type must be 'base', 'tflite', or 'tensorrt'".format(
        #             self.model_type
        #         )
        #     )

        # check if using TFGPUinference flag
        # if not, get pose from network output

        # ! to be replaced
        """
        if len(pose_output) > 1:
            scmap, locref = extract_cnn_output(
                pose_output, self.cfg
            )  # scmap = the heatmaps of likelihood of different key points being placed at different positions in the frame. locref = think it is something with refining how the 'peaks' in the heatmap is created.
            num_outputs = self.cfg.get("num_outputs", 1)
            if num_outputs > 1:  # seems to be if detecting more than 1 key point
                self.pose = multi_pose_predict(
                    scmap, locref, self.cfg["stride"], num_outputs
                )
            else:
                self.pose = argmax_pose_predict(scmap, locref, self.cfg["stride"])
        else:
            pose = np.array(pose_output[0])
            self.pose = pose[:, [1, 0, 2]]

        # display image if display=True before correcting pose for cropping/resizing

        if self.display is not None:
            self.display.display_frame(frame, self.pose)

        # if frame is cropped, convert pose coordinates to original frame coordinates

        if self.resize is not None:
            self.pose[:, :2] *= 1 / self.resize

        if self.cropping is not None:
            self.pose[:, 0] += self.cropping[0]
            self.pose[:, 1] += self.cropping[2]

        if self.dynamic_cropping is not None:
            self.pose[:, 0] += self.dynamic_cropping[0]
            self.pose[:, 1] += self.dynamic_cropping[2]

        # process the pose

        if self.processor:
            self.pose = self.processor.process(self.pose, **kwargs)"""

        # Mock pose
        num_individuals = 1
        num_kpts = 3

        # ! Multi animal OR single animal: display only supports single for now

        # self.pose = np.ones((num_individuals, num_kpts, 3))   # Multi animal
        # self.pose = np.ones((num_kpts, 3))                    # Single animal

        # mock_frame = np.ones((1, 3, 128, 128))
        frame = torch.Tensor(frame).permute(2, 0, 1)

        # Pytorch pose prediction
        pose_model = self.load_model()
        outputs = pose_model(frame)
        self.pose = pose_model.get_predictions(outputs)

        # debug
        print(pose_model)
        # print(self.pose, self.pose["bodypart"]["poses"].shape())
        return self.pose

    # def close(self):
    #     """ Close tensorflow session
    #     """

    #     self.sess.close()
    #     self.sess = None
    #     self.is_initialized = False
    #     if self.display is not None:
    #         self.display.destroy()
