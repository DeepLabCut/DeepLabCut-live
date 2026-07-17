"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import logging
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dlclive.engine import Engine
from dlclive.exceptions import DLCLiveWarning


def get_torch(required: bool = False, feature: str | None = None):
    """Lazily import torch.

    Args:
        required: If True, raise a clear error when torch is unavailable.
        feature: Optional feature name to include in error messages.

    Returns:
        The imported torch module, or None when unavailable and not required.
    """
    try:
        import torch

        return torch
    except (ImportError, ModuleNotFoundError) as exc:
        if required:
            context = f" for {feature}" if feature else ""
            raise ModuleNotFoundError(
                f"PyTorch is required{context} but is not installed. "
                "Install it with: pip install deeplabcut-live[pytorch]"
            ) from exc
        return None


def get_tensorflow(required: bool = False, feature: str | None = None):
    """Lazily import tensorflow.

    Args:
        required: If True, raise a clear error when tensorflow is unavailable.
        feature: Optional feature name to include in error messages.

    Returns:
        The imported tensorflow module, or None when unavailable and not required.
    """
    try:
        import tensorflow as tf

        return tf
    except (ImportError, ModuleNotFoundError) as exc:
        if required:
            context = f" for {feature}" if feature else ""
            raise ModuleNotFoundError(
                f"TensorFlow is required{context} but is not installed. "
                "Install it with: pip install deeplabcut-live[tf]"
            ) from exc
        return None


def convert_to_ubyte(frame: np.ndarray) -> np.ndarray:
    """Converts an image to unsigned 8-bit integer numpy array.
        If scikit-image is installed, uses skimage.img_as_ubyte, otherwise, uses a similar custom function.

    Parameters
    ----------
    frame:
        an image as a numpy array

    Returns
    -------
    :class: `numpy.ndarray`
        image converted to uint8
    """
    return _img_as_ubyte_np(frame)


def resize_frame(frame: np.ndarray, resize=None) -> np.ndarray:
    """Resizes an image using OpenCV.

    Parameters
    ----------
    frame:
        an image as a numpy array
    """

    if (resize is not None) and (resize != 1):
        new_x = int(frame.shape[0] * resize)
        new_y = int(frame.shape[1] * resize)
        return cv2.resize(frame, (new_y, new_x))

    else:
        return frame


def img_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an image to RGB using OpenCV.

    Parameters
    ----------
    frame : :class:`numpy.ndarray
        an image as a numpy array
    """

    if frame.ndim == 2:
        return gray_to_rgb(frame)

    elif frame.ndim == 3:
        return bgr_to_rgb(frame)

    else:
        warnings.warn(
            f"Image has {frame.ndim} dimensions. Must be 2 or 3 dimensions to convert to RGB",
            DLCLiveWarning,
            stacklevel=2,
        )
        return frame


def gray_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an image from grayscale to RGB using OpenCV.

    Parameters
    ----------
    frame : :class:`numpy.ndarray
        an image as a numpy array
    """

    return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an image from BGR to RGB using OpenCV.

    Parameters
    ----------
    frame : :class:`numpy.ndarray
        an image as a numpy array
    """

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _img_as_ubyte_np(frame: np.ndarray) -> np.ndarray:
    """Converts an image as a numpy array to unsinged 8-bit integer.
        As in scikit-image img_as_ubyte, converts negative pixels to 0 and converts range to [0, 255]

    Parameters
    ----------
    image : :class:`numpy.ndarray`
        an image as a numpy array

    Returns
    -------
    :class:`numpy.ndarray`
        image converted to uint8
    """

    frame = np.array(frame)
    im_type = frame.dtype.type

    # check if already ubyte
    if np.issubdtype(im_type, np.uint8):
        return frame

    # if floating
    elif np.issubdtype(im_type, np.floating):
        if (np.min(frame) < -1) or (np.max(frame) > 1):
            raise ValueError("Images of type float must be between -1 and 1.")

        frame *= 255
        frame = np.rint(frame)
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    # if integer
    elif np.issubdtype(im_type, np.integer):
        im_type_info = np.iinfo(im_type)
        frame *= 255 / im_type_info.max
        frame[frame < 0] = 0
        return frame.astype(np.uint8)

    else:
        raise TypeError(f"image of type {im_type} could not be converted to ubyte")


def decode_fourcc(cc):
    """
    Convert float fourcc code from opencv to characters.
    If decode fails, returns empty string.
    https://stackoverflow.com/a/49138893
    Arguments:
        cc (float, int): fourcc code from opencv
    Returns:
         str: Character format of fourcc code

    Examples:
        >>> vid = cv2.VideoCapture('/some/video/path.avi')
        >>> decode_fourcc(vid.get(cv2.CAP_PROP_FOURCC))
        'DIVX'
    """
    try:
        decoded = "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        decoded = ""

    return decoded


def get_available_backends() -> list[Engine]:
    """
    Check which backends (TensorFlow or PyTorch) are installed.

    Returns:
        list[str]: List of installed backends. Possible values: ["tensorflow"], ["pytorch"],
                   or ["tensorflow", "pytorch"]. Returns an empty list if neither is installed.

    Warns:
        DLCLiveWarning: If neither TensorFlow nor PyTorch is installed.
    """
    backends = []

    if get_tensorflow(required=False) is not None:
        backends.append(Engine.TENSORFLOW)

    if get_torch(required=False) is not None:
        backends.append(Engine.PYTORCH)

    if not backends:
        warnings.warn(
            "Neither TensorFlow nor PyTorch is installed. One of these is required to use DLCLive!"
            "Install with: pip install deeplabcut-live[tf] or pip install deeplabcut-live[pytorch]",
            DLCLiveWarning,
            stacklevel=2,
        )

    return backends


def download_file(url: str, filepath: str, chunk_size: int = 8192) -> None:
    """
    Download a file from a URL with progress bar and error handling.

    Args:
        url: URL to download from
        filepath: Local path to save the file
        chunk_size: Size of chunks to read (default: 8192 bytes)

    Raises:
        urllib.error.URLError: If the download fails
        IOError: If the file cannot be written
    """
    filepath = Path(filepath)

    # Check if file already exists
    if filepath.exists():
        print(f"File already exists at {filepath}, skipping download.")
        return

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Open the URL
        with urllib.request.urlopen(url) as response:
            # Get file size if available
            total_size = int(response.headers.get("Content-Length", 0))

            # Create progress bar if file size is known
            if total_size > 0:
                pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")
            else:
                pbar = None
                print("Downloading...")

            # Download and write file
            downloaded = 0
            with open(filepath, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))

            if pbar:
                pbar.close()

            # Verify file was written
            if not filepath.exists() or filepath.stat().st_size == 0:
                raise OSError(f"Downloaded file is empty or was not written to {filepath}")

            print(f"Successfully downloaded to {filepath}")

    except urllib.error.HTTPError as e:
        raise urllib.error.URLError(f"HTTP error {e.code}: {e.reason} when downloading from {url}") from e
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"Failed to download from {url}: {e.reason}") from e
    except OSError as e:
        raise OSError(f"Failed to write file to {filepath}") from e


class DebugTimingStats:
    """Tiny timing accumulator for camera worker performance diagnostics.

    Usage:
        with stats.measure("read"):
            frame, ts = backend.read()

    Logs aggregate timings once per log_interval seconds.
    """

    def __init__(
        self, info: str, *, logger: logging.Logger | None = None, log_interval: float = 1.0, enabled: bool = True
    ):
        self.info = info
        self.log_interval = float(log_interval)
        self.enabled = bool(enabled)
        self.logger = logger or logging.getLogger(__name__)
        if self.enabled:  # force logger to proper level
            if not self.logger.isEnabledFor(logging.DEBUG):
                self.logger.setLevel(logging.DEBUG)

        self._last_log = time.perf_counter()
        self._frames = 0
        self._timeouts = 0
        self._errors = 0
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    class _Measure:
        def __init__(self, parent: "DebugTimingStats", name: str):
            self.parent = parent
            self.name = name
            self.t0 = 0.0
            self.elapsed = 0.0

        def __enter__(self):
            if self.parent.enabled:
                self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            if not self.parent.enabled:
                return False

            self.elapsed = time.perf_counter() - self.t0
            self.parent._totals[self.name] = self.parent._totals.get(self.name, 0.0) + self.elapsed
            self.parent._counts[self.name] = self.parent._counts.get(self.name, 0) + 1
            return False

    def measure(self, name: str):
        return self._Measure(self, name)

    def note_frame(self) -> None:
        if self.enabled:
            self._frames += 1

    def note_timeout(self) -> None:
        if self.enabled:
            self._timeouts += 1

    def note_error(self) -> None:
        if self.enabled:
            self._errors += 1

    def maybe_log(self) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        elapsed = now - self._last_log
        if elapsed < self.log_interval:
            return

        fps = self._frames / max(elapsed, 1e-9)

        parts = [
            f"[Worker {self.info}]",
            f"fps={fps:.1f}",
            f"frames={self._frames}",
        ]

        if self._timeouts:
            parts.append(f"timeouts={self._timeouts}")
        if self._errors:
            parts.append(f"errors={self._errors}")

        for name in sorted(self._totals):
            count = max(self._counts.get(name, 0), 1)
            avg_ms = 1000.0 * self._totals[name] / count
            parts.append(f"avg_{name}_ms={avg_ms:.3f}")

        self.logger.debug(" ".join(parts))

        self._last_log = now
        self._frames = 0
        self._timeouts = 0
        self._errors = 0
        self._totals.clear()
        self._counts.clear()
