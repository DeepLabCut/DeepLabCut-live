"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import dlclive.factory as factory
import dlclive.utils as utils
from dlclive.core.runner import BaseRunner
from dlclive.display import Display
from dlclive.exceptions import DLCLiveError
from dlclive.processor import Processor


class DLCLive:
    """
    Class that loads a DLC network and performs inference on single images (e.g.
    images captured from a camera feed)

    Parameters
    -----------

    model_path: Path
        Full path to exported model (created when `deeplabcut.export_model(...)` was
        called). For PyTorch models, this is a single model file. For TensorFlow models,
        this is a directory containing the model snapshots.

    model_type: string, optional
        Which model to use. For the PyTorch engine, options are [`pytorch`]. For the
        TensorFlow engine, options are [`base`, `tensorrt`, `lite`].

    precision: string, optional
        Precision of model weights, for model_type "pytorch" and "tensorrt". Options
        are, for different model_types:
            "pytorch": {"FP32", "FP16"}
            "tensorrt": {"FP32", "FP16", "INT8"}

    tf_config:
        TensorFlow only. Optional ConfigProto for the TensorFlow session.

    single_animal: bool, default=True
        PyTorch only.

    device: str, optional, default=None
        PyTorch only.

    top_down_config: dict, optional, default=None

    top_down_dynamic: dict, optional, default=None

    cropping: list of int
        Cropping parameters in pixel number: [x1, x2, y1, y2]

    dynamic: triple containing (state, detectiontreshold, margin)
        If the state is true, then dynamic cropping will be performed. That means that
        if an object is detected (i.e. any body part > detectiontreshold), then object
        boundaries are computed according to the smallest/largest x position and
        smallest/largest y position of all body parts. This window is expanded by the
        margin and from then on only the posture within this crop is analyzed (until the
        object is lost, i.e. <detectiontreshold). The current position is utilized for
        updating the crop window for the next frame (this is why the margin is important
        and should be set large enough given the movement of the animal).

    resize: float, optional
        Factor to resize the image.
        For example, resize=0.5 will downsize both the height and width of the image by
        a factor of 2.

    processor: dlc pose processor object, optional
        User-defined processor object. Must contain two methods: process and save.
        The 'process' method takes in a pose, performs some processing, and returns
        processed pose.
        The 'save' method saves any valuable data created by or used by the processor
        Processors can be used for two main purposes:
            i) to run a forward predicting model that will predict the future pose from
            past history of poses (history can be stored in the processor object, but is
            not stored in this DLCLive object)
            ii) to trigger external hardware based on pose estimation (e.g. see
            'TeensyLaser' processor)

    convert2rgb: bool, optional
        boolean flag to convert frames from BGR to RGB color scheme

    display: bool, optional
        Open a display to show predicted pose in frames with DeepLabCut labels.
        This is useful for testing model accuracy and cropping parameters, but it is
        very slow.

    pcutoff: float, default=0.5
        Only used when display=True. The score threshold for displaying a bodypart in
        the display.

    display_radius: int, default=3
        Only used when display=True. Radius for keypoint display in pixels, default=3

    display_cmap: str, optional
        Only used when display=True. String indicating the Matplotlib colormap to use.
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
        model_path: str | Path,
        model_type: str = "base",
        precision: str = "FP32",
        tf_config: Any = None,
        single_animal: bool = True,
        device: str | None = None,
        top_down_config: dict | None = None,
        top_down_dynamic: dict | None = None,
        cropping: list[int] | None = None,
        dynamic: tuple[bool, float, float] = (False, 0.5, 10),
        resize: float | None = None,
        convert2rgb: bool = True,
        processor: Processor | None = None,
        display: bool | Display = False,
        pcutoff: float = 0.5,
        display_radius: int = 3,
        display_cmap: str = "bmy",
    ):
        self.path = Path(model_path)
        self.runner: BaseRunner = factory.build_runner(
            model_type,
            model_path,
            precision=precision,
            tf_config=tf_config,
            single_animal=single_animal,
            device=device,
            dynamic=top_down_dynamic,
            top_down_config=top_down_config,
        )
        self.is_initialized = False

        self.model_type = model_type
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

    @property
    def cfg(self) -> dict | None:
        return self.runner.cfg

    def read_config(self) -> None:
        """Reads configuration yaml file

        Raises
        ------
        FileNotFoundError
            error thrown if pose configuration file does not exist
        """
        self.runner.read_config()

    @property
    def parameterization(
        self,
    ) -> dict:
        return {param: getattr(self, param) for param in self.PARAMETERS}

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
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
        if self.cropping:
            frame = frame[
                self.cropping[2] : self.cropping[3], self.cropping[0] : self.cropping[1]
            ]

        if self.dynamic[0]:
            if self.pose is not None:
                # Deal with PyTorch multi-animal models
                if len(self.pose.shape) == 3:
                    if len(self.pose) == 0:
                        pose = np.zeros((1, 3))
                    elif len(self.pose) == 1:
                        pose = self.pose[0]
                    else:
                        raise ValueError(
                            "Cannot use Dynamic Cropping - more than 1 individual found"
                        )

                else:
                    pose = self.pose

                detected = pose[:, 2] >= self.dynamic[1]
                if np.any(detected):
                    h, w = frame.shape[0], frame.shape[1]

                    x = pose[detected, 0]
                    y = pose[detected, 1]
                    xmin, xmax = int(np.min(x)), int(np.max(x))
                    ymin, ymax = int(np.min(y)), int(np.max(y))

                    x1 = max([0, xmin - self.dynamic[2]])
                    x2 = min([w, xmax + self.dynamic[2]])
                    y1 = max([0, ymin - self.dynamic[2]])
                    y2 = min([h, ymax + self.dynamic[2]])

                    self.dynamic_cropping = [x1, x2, y1, y2]
                    frame = frame[y1:y2, x1:x2]

                else:
                    self.dynamic_cropping = None

        if self.resize != 1:
            frame = utils.resize_frame(frame, self.resize)

        if self.convert2rgb:
            frame = utils.img_to_rgb(frame)

        return frame

    def init_inference(self, frame=None, **kwargs) -> np.ndarray:
        """
        Load model and perform inference on first frame -- the first inference is
        usually very slow.

        Parameters
        -----------
        frame :class:`numpy.ndarray`
            image as a numpy array

        Returns
        --------
        pose: the pose estimated by DeepLabCut for the input image
        """
        if frame is None:
            raise DLCLiveError("No frame provided to initialize inference.")

        if frame.ndim >= 2:
            self.convert2rgb = True

        processed_frame = self.process_frame(frame)
        self.pose = self.runner.init_inference(processed_frame)
        self.is_initialized = True
        return self._post_process_pose(processed_frame, **kwargs)

    def get_pose(self, frame: np.ndarray | None = None, **kwargs) -> np.ndarray:
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
        inf_time:class: `float`
            the pose inference time
        """
        if frame is None:
            raise DLCLiveError("No frame provided for live pose estimation")

        if frame.ndim >= 2:
            self.convert2rgb = True

        processed_frame = self.process_frame(frame)
        self.pose = self.runner.get_pose(processed_frame)
        return self._post_process_pose(processed_frame, **kwargs)

    def _post_process_pose(self, processed_frame: np.ndarray, **kwargs) -> np.ndarray:
        """Post-processes the frame and pose."""
        # display image if display=True before correcting pose for cropping/resizing
        if self.display is not None:
            self.display.display_frame(processed_frame, self.pose)

        # if frame is cropped, convert pose coordinates to original frame coordinates
        if self.resize is not None:
            self.pose[..., :2] *= 1 / self.resize

        if self.cropping is not None:
            self.pose[..., 0] += self.cropping[0]
            self.pose[..., 1] += self.cropping[2]

        if self.dynamic_cropping is not None:
            self.pose[..., 0] += self.dynamic_cropping[0]
            self.pose[..., 1] += self.dynamic_cropping[2]

        # process the pose
        if self.processor:
            self.pose = self.processor.process(self.pose, **kwargs)

        return self.pose

    def close(self) -> None:
        self.is_initialized = False
        self.runner.close()
        if self.display is not None:
            self.display.destroy()
