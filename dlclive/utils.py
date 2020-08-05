"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import numpy as np
import warnings
from dlclive.exceptions import DLCLiveWarning

try:
    import skimage

    SK_IM = True
except Exception:
    SK_IM = False

try:
    import cv2

    OPEN_CV = True
except Exception:
    from PIL import Image

    OPEN_CV = False
    warnings.warn(
        "OpenCV is not installed. Using pillow for image processing, which is slower.",
        DLCLiveWarning,
    )


def convert_to_ubyte(frame):
    """ Converts an image to unsigned 8-bit integer numpy array. 
        If scikit-image is installed, uses skimage.img_as_ubyte, otherwise, uses a similar custom function.

    Parameters
    ----------
    image : :class:`numpy.ndarray`
        an image as a numpy array

    Returns
    -------
    :class:`numpy.ndarray`
        image converted to uint8
    """

    if SK_IM:
        return skimage.img_as_ubyte(frame)
    else:
        return _img_as_ubyte_np(frame)


def resize_frame(frame, resize=None):
    """ Resizes an image. Uses OpenCV if installed, otherwise, uses pillow

    Parameters
    ----------
    image : :class:`numpy.ndarray`
        an image as a numpy array
    """

    if (resize is not None) and (resize != 1):

        if OPEN_CV:

            new_x = int(frame.shape[0] * resize)
            new_y = int(frame.shape[1] * resize)
            return cv2.resize(frame, (new_y, new_x))

        else:

            img = Image.fromarray(frame)
            img = img.resize((new_y, new_x))
            return np.asarray(img)

    else:

        return frame


def img_to_rgb(frame):
    """ Convert an image to RGB. Uses OpenCV is installed, otherwise uses pillow.

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
        )
        return frame


def gray_to_rgb(frame):
    """ Convert an image from grayscale to RGB. Uses OpenCV is installed, otherwise uses pillow.

    Parameters
    ----------
    frame : :class:`numpy.ndarray
        an image as a numpy array
    """

    if OPEN_CV:

        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    else:

        img = Image.fromarray(frame)
        img = img.convert("RGB")
        return np.asarray(img)


def bgr_to_rgb(frame):
    """ Convert an image from BGR to RGB. Uses OpenCV is installed, otherwise uses pillow.

    Parameters
    ----------
    frame : :class:`numpy.ndarray
        an image as a numpy array
    """

    if OPEN_CV:

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    else:

        img = Image.fromarray(frame)
        img = img.convert("RGB")
        return np.asarray(img)


def _img_as_ubyte_np(frame):
    """ Converts an image as a numpy array to unsinged 8-bit integer.
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

        raise TypeError(
            "image of type {} could not be converted to ubyte".format(im_type)
        )


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
    except:
        decoded = ""

    return decoded
