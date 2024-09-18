#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""TensorFlow runners for DeepLabCut-Live"""
import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from dlclive.core.config import read_yaml
from dlclive.core.runner import BaseRunner
from dlclive.exceptions import DLCLiveError
from dlclive.pose_estimation_tensorflow.graph import (
    extract_graph,
    finalize_graph,
    get_output_nodes,
    get_output_tensors,
    read_graph,
)
from dlclive.pose_estimation_tensorflow.pose import (
    argmax_pose_predict,
    extract_cnn_output,
    multi_pose_predict,
)


class TensorFlowRunner(BaseRunner):
    """TensorFlow runner for live pose estimation using DeepLabCut-Live.

    Args:
        path: The path to the model to run inference with.

    Attributes:
        path: The path to the model to run inference with.
    """

    def __init__(
        self,
        path: str | Path,
        model_type: str = "base",
        tf_config: Any = None,
    ) -> None:
        super().__init__(path)
        self.cfg = self.read_config()
        self.model_type = model_type
        self.tf_config = tf_config
        self.precision = "FP32"
        self.sess = None
        self.inputs = None
        self.outputs = None
        self.tflite_interpreter = None

    def close(self) -> None:
        """Clears any resources used by the runner."""
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def get_pose(self, frame: np.ndarray, **kwargs) -> np.ndarray:
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
                f"model_type={self.model_type} is not supported. model_type must be "
                f"'base', 'tflite', or 'tensorrt'"
            )

        # check if using TFGPUinference flag
        # if not, get pose from network output
        if len(pose_output) > 1:
            scmap, locref = extract_cnn_output(pose_output, self.cfg)
            num_outputs = self.cfg.get("num_outputs", 1)
            if num_outputs > 1:
                pose = multi_pose_predict(
                    scmap, locref, self.cfg["stride"], num_outputs
                )
            else:
                pose = argmax_pose_predict(scmap, locref, self.cfg["stride"])
        else:
            pose = np.array(pose_output[0])
            pose = pose[:, [1, 0, 2]]

        return pose

    def init_inference(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        model_file = glob.glob(os.path.normpath(str(self.path) + "/*.pb"))[0]

        tf_ver = tf.__version__
        tf_version_2 = tf_ver[0] == "2"

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
            placeholder_shape = [1, frame.shape[0], frame.shape[1], 3]

            if tf_version_2:
                converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
                    model_file,
                    ["Placeholder"],
                    output_nodes,
                    input_shapes={"Placeholder": placeholder_shape},
                )
            else:
                converter = tf.lite.TFLiteConverter.from_frozen_graph(
                    model_file,
                    ["Placeholder"],
                    output_nodes,
                    input_shapes={"Placeholder": placeholder_shape},
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

            if (tf_ver[0] > 1) | (tf_ver[0] == 1 & tf_ver[1] >= 14):
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
                f"model_type={self.model_type} is not supported. model_type must be "
                "'base', 'tflite', or 'tensorrt'"
            )

        return self.get_pose(frame, **kwargs)

    def read_config(self) -> dict:
        """Reads the configuration file"""
        return read_yaml(self.path / "pose_cfg.yaml")
