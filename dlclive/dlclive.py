"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import os
import ruamel.yaml
import glob
import warnings
import numpy as np
import tensorflow as tf

try:
    TFVER = [int(v) for v in tf.__version__.split(".")]
    if TFVER[1] < 14:
        from tensorflow.contrib.tensorrt import trt_convert as trt
    else:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
except Exception:
    pass

from dlclive.graph import (
    read_graph,
    finalize_graph,
    get_output_nodes,
    get_output_tensors,
    extract_graph,
)
from dlclive.pose import extract_cnn_output, argmax_pose_predict, multi_pose_predict
from dlclive.display import Display
from dlclive import utils
from dlclive.exceptions import DLCLiveError, DLCLiveWarning


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
        cropping parameters in pixel number: [x1, x2, y1, y2]

    dynamic: triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that if an object is detected (i.e. any body part > detectiontreshold),
        then object boundaries are computed according to the smallest/largest x position and smallest/largest y position of all body parts. This  window is
        expanded by the margin and from then on only the posture within this crop is analyzed (until the object is lost, i.e. <detectiontreshold). The
        current position is utilized for updating the crop window for the next frame (this is why the margin is important and should be set large
        enough given the movement of the animal).

    resize : float, optional
        Factor to resize the image.
        For example, resize=0.5 will downsize both the height and width of the image by a factor of 2.

    processor: dlc pose processor object, optional
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
        model_path,
        model_type="base",
        precision="FP32",
        tf_config=None,
        cropping=None,
        dynamic=(False, 0.5, 10),
        resize=None,
        convert2rgb=True,
        processor=None,
        display=False,
        pcutoff=0.5,
        display_radius=3,
        display_cmap="bmy",
    ):

        self.path = model_path
        self.cfg = None
        self.model_type = model_type
        self.tf_config = tf_config
        self.precision = precision
        self.cropping = cropping
        self.dynamic = dynamic
        self.dynamic_cropping = None
        self.resize = resize
        self.processor = processor
        self.convert2rgb = convert2rgb
        self.display = (
            Display(pcutoff=pcutoff, radius=display_radius, cmap=display_cmap)
            if display
            else None
        )

        self.sess = None
        self.inputs = None
        self.outputs = None
        self.tflite_interpreter = None
        self.pose = None
        self.is_initialized = False

        # checks

        if self.model_type == "tflite" and self.dynamic[0]:
            self.dynamic[0] = False
            warnings.warn(
                "Dynamic cropping is not supported for tensorflow lite inference. Dynamic cropping will not be used...",
                DLCLiveWarning,
            )

        self.read_config()

    def read_config(self):
        """ Reads configuration yaml file

        Raises
        ------
        FileNotFoundError
            error thrown if pose configuration file does nott exist
        """

        cfg_path = os.path.normpath(self.path + "/pose_cfg.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"The pose configuration file for the exported model at {cfg_path} was not found. Please check the path to the exported model directory"
            )

        ruamel_file = ruamel.yaml.YAML()
        self.cfg = ruamel_file.load(open(cfg_path, "r"))

    @property
    def parameterization(self) -> dict:
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

        if frame.dtype != np.uint8:

            frame = utils.convert_to_ubyte(frame)

        if self.cropping:

            frame = frame[
                self.cropping[2] : self.cropping[3], self.cropping[0] : self.cropping[1]
            ]

        if self.dynamic[0]:

            if self.pose is not None:

                detected = self.pose[:, 2] > self.dynamic[1]

                if np.any(detected):

                    x = self.pose[detected, 0]
                    y = self.pose[detected, 1]

                    x1 = int(max([0, int(np.amin(x)) - self.dynamic[2]]))
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

        model_file = glob.glob(os.path.normpath(self.path + "/*.pb"))[0]
        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                "The model file {} does not exist.".format(model_file)
            )

        # process frame

        if frame is None and (self.model_type == "tflite"):
            raise DLCLiveError(
                "No image was passed to initialize inference. An image must be passed to the init_inference method"
            )

        if frame is not None:
            if frame.ndim == 2:
                self.convert2rgb = True
            frame = self.process_frame(frame)

        # load model

        if self.model_type == "base":

            graph_def = read_graph(model_file)
            graph = finalize_graph(graph_def)
            self.sess, self.inputs, self.outputs = extract_graph(
                graph, tf_config=self.tf_config
            )

        elif self.model_type == "tflite":

            ###
            # the frame size needed to initialize the tflite model as
            # tflite does not support saving a model with dynamic input size
            ###

            # get input and output tensor names from graph_def
            graph_def = read_graph(model_file)
            graph = finalize_graph(graph_def)
            output_nodes = get_output_nodes(graph)
            output_nodes = [on.replace("DLC/", "") for on in output_nodes]
            converter = tf.lite.TFLiteConverter.from_frozen_graph(
                model_file,
                ["Placeholder"],
                output_nodes,
                input_shapes={"Placeholder": [1, frame.shape[0], frame.shape[1], 3]},
            )
            try:
                tflite_model = converter.convert()
            except Exception:
                raise DLCLiveError(
                    (
                        "This model cannot be converted to tensorflow lite format. "
                        "To use tensorflow lite for live inference, "
                        "make sure to set TFGPUinference=False "
                        "when exporting the model from DeepLabCut"
                    )
                )

            self.tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
            self.tflite_interpreter.allocate_tensors()
            self.inputs = self.tflite_interpreter.get_input_details()
            self.outputs = self.tflite_interpreter.get_output_details()

        elif self.model_type == "tensorrt":

            graph_def = read_graph(model_file)
            graph = finalize_graph(graph_def)
            output_tensors = get_output_tensors(graph)
            output_tensors = [ot.replace("DLC/", "") for ot in output_tensors]

            if (TFVER[0] > 1) | (TFVER[0] == 1 & TFVER[1] >= 14):
                converter = trt.TrtGraphConverter(
                    input_graph_def=graph_def,
                    nodes_blacklist=output_tensors,
                    is_dynamic_op=True,
                )
                graph_def = converter.convert()
            else:
                graph_def = trt.create_inference_graph(
                    input_graph_def=graph_def,
                    outputs=output_tensors,
                    max_batch_size=1,
                    precision_mode=self.precision,
                    is_dynamic_op=True,
                )

            graph = finalize_graph(graph_def)
            self.sess, self.inputs, self.outputs = extract_graph(
                graph, tf_config=self.tf_config
            )

        else:

            raise DLCLiveError(
                "model_type = {} is not supported. model_type must be 'base', 'tflite', or 'tensorrt'".format(
                    self.model_type
                )
            )

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

        if self.model_type in ["base", "tensorrt"]:

            pose_output = self.sess.run(
                self.outputs, feed_dict={self.inputs: np.expand_dims(frame, axis=0)}
            )

        elif self.model_type == "tflite":

            self.tflite_interpreter.set_tensor(
                self.inputs[0]["index"],
                np.expand_dims(frame, axis=0).astype(np.float32),
            )
            self.tflite_interpreter.invoke()

            if len(self.outputs) > 1:
                pose_output = [
                    self.tflite_interpreter.get_tensor(self.outputs[0]["index"]),
                    self.tflite_interpreter.get_tensor(self.outputs[1]["index"]),
                ]
            else:
                pose_output = self.tflite_interpreter.get_tensor(
                    self.outputs[0]["index"]
                )

        else:

            raise DLCLiveError(
                "model_type = {} is not supported. model_type must be 'base', 'tflite', or 'tensorrt'".format(
                    self.model_type
                )
            )

        # check if using TFGPUinference flag
        # if not, get pose from network output

        if len(pose_output) > 1:
            scmap, locref = extract_cnn_output(pose_output, self.cfg)
            num_outputs = self.cfg.get("num_outputs", 1)
            if num_outputs > 1:
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
            self.pose = self.processor.process(self.pose, **kwargs)

        return self.pose

    def close(self):
        """ Close tensorflow session
        """

        self.sess.close()
        self.sess = None
        self.is_initialized = False
        if self.display is not None:
            self.display.destroy()
