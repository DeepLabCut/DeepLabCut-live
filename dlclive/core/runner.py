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
"""Base runner for DeepLabCut-Live"""
import abc
from pathlib import Path

import numpy as np


class BaseRunner(abc.ABC):
    """Base runner for live pose estimation using DeepLabCut-Live.

    Args:
        path: The path to the model to run inference with.

    Attributes:
        cfg: The pose configuration data.
        path: The path to the model to run inference with.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.cfg = None

    @abc.abstractmethod
    def close(self) -> None:
        """Clears any resources used by the runner."""
        pass

    @abc.abstractmethod
    def get_pose(self, frame: np.ndarray | None, **kwargs) -> np.ndarray | None:
        """
        Abstract method to calculate and retrieve the pose of an object or system
        based on the given input frame of data. This method must be implemented
        by any subclass inheriting from this abstract base class to define the
        specific approach for pose estimation.

        Parameters
        ----------
        frame : np.ndarray
            The input data or image frame used for estimating the pose. Typically
            represents visual data such as video or image frames.
        kwargs : dict, optional
            Additional keyword arguments that may be required for specific pose
            estimation techniques implemented in the subclass.

        Returns
        -------
        np.ndarray
            The estimated pose resulting from the pose estimation process. The
            structure of the array may depend on the specific implementation
            but typically represents transformations or coordinates.
        """
        pass

    @abc.abstractmethod
    def init_inference(self, frame: np.ndarray | None, **kwargs) -> np.ndarray | None:
        """
        Initializes inference process on the provided frame.

        This method serves as an abstract base method, meant to be implemented by
        subclasses. It takes an input image frame and optional additional parameters
        to set up and perform inference. The method must return a processed result
        as a numpy array.

        Parameters
        ----------
        frame : np.ndarray
            The input image frame for which inference needs to be set up.
        kwargs : dict, optional
            Additional parameters that may be required for specific implementation
            of the inference initialization.

        Returns
        -------
        np.ndarray
            The result of the inference after being initialized and processed.
        """
        pass

    @abc.abstractmethod
    def read_config(self) -> dict:
        """
        Reads the pose configuration file.

        Returns
        -------
        dict
            The runner configuration

        Raises:
            FileNotFoundError: if the pose configuration file does not exist
        """
        pass
