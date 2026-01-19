"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def fix_bbox_aspect_ratio(
    bbox: tuple[float, float, float, float] | np.ndarray | torch.Tensor,
    margin: int | float,
    out_w: int,
    out_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    w += 2 * margin
    h += 2 * margin

    input_ratio = w / h
    output_ratio = out_w / out_h
    if input_ratio > output_ratio:  # h/w < h0/w0 => h' = w * h0/w0
        h = w / output_ratio
    elif input_ratio < output_ratio:  # w/h < w0/h0 => w' = h * w0/h0
        w = h * output_ratio

    # cx,cy,w,h will now give the right ratio -> check if padding is needed
    x1, y1 = int(round(cx - (w / 2))), int(round(cy - (h / 2)))
    x2, y2 = int(round(cx + (w / 2))), int(round(cy + (h / 2)))

    return x1, y1, x2, y2


def crop_corners(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    center_padding: bool = True,
) -> tuple[int, int, int, int, int, int, int, int]:
    """"""
    x1, y1, x2, y2 = bbox
    img_w, img_h = image_size

    # pad symmetrically - compute total padding across axis
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    if x1 < 0:
        pad_left = -x1
        x1 = 0
    if x2 > img_w:
        pad_right = x2 - img_w
        x2 = img_w
    if y1 < 0:
        pad_top = -y1
        y1 = 0
    if y2 > img_h:
        pad_bottom = y2 - img_h
        y2 = img_h

    pad_x = pad_left + pad_right
    pad_y = pad_top + pad_bottom
    if center_padding:
        pad_left = pad_x // 2
        pad_top = pad_y // 2

    return x1, y1, x2, y2, pad_left, pad_top, pad_x, pad_y


def top_down_crop(
    image: np.ndarray | torch.Tensor,
    bbox: tuple[float, float, float, float] | np.ndarray | torch.Tensor,
    output_size: tuple[int, int],
    margin: int = 0,
    center_padding: bool = False,
) -> tuple[np.array, tuple[int, int], tuple[float, float]]:
    """
    Crops images around bounding boxes for top-down pose estimation. Computes offsets so
    that coordinates in the original image can be mapped to the cropped one;

        x_cropped = (x - offset_x) / scale_x
        x_cropped = (y - offset_y) / scale_y

    Bounding boxes are expected to be in COCO-format (xywh).

    Args:
        image: (h, w, c) the image to crop
        bbox: (4,) the bounding box to crop around
        output_size: the (width, height) of the output cropped image
        margin: a margin to add around the bounding box before cropping
        center_padding: whether to center the image in the padding if any is needed

    Returns:
        cropped_image, (offset_x, offset_y), (scale_x, scale_y)
    """
    image_h, image_w, c = image.shape
    img_size = (image_w, image_h)
    out_w, out_h = output_size

    bbox = fix_bbox_aspect_ratio(bbox, margin, out_w, out_h)
    x1, y1, x2, y2, pad_left, pad_top, pad_x, pad_y = crop_corners(
        bbox, img_size, center_padding
    )
    w, h = x2 - x1, y2 - y1
    crop_w, crop_h = w + pad_x, h + pad_y

    # crop the pixels we care about
    image_crop = np.zeros((crop_h, crop_w, c), dtype=image.dtype)
    image_crop[pad_top : pad_top + h, pad_left : pad_left + w] = image[y1:y2, x1:x2]

    # resize the cropped image
    image = cv2.resize(image_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # compute scale and offset
    offset = x1 - pad_left, y1 - pad_top
    scale = crop_w / out_w, crop_h / out_h
    return image, offset, scale


def top_down_crop_torch(
    image: torch.Tensor,
    bbox: tuple[float, float, float, float] | torch.Tensor,
    output_size: tuple[int, int],
    margin: int = 0,
) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
    """"""
    out_w, out_h = output_size

    x1, y1, x2, y2 = fix_bbox_aspect_ratio(bbox, margin, out_w, out_h)
    h, w = x2 - x1, y2 - y1

    F.resized_crop(image, y1, x1, h, w, [out_h, out_w])

    scale = w / out_w, h / out_h
    offset = x1, y1
    crop = F.resized_crop(image, y1, x1, h, w, [out_h, out_w])
    return crop, offset, scale


class AutoPadToDivisor(torch.nn.Module):
    def __init__(self, pad_height_divisor: int = 1, pad_width_divisor: int = 1):
        super().__init__()
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Accepts either (C, H, W) or (N, C, H, W)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # add batch dim

        assert img.ndim == 4, f"Expected 4D tensor, got shape {img.shape}"
        _, _, h, w = img.shape

        target_h = ((h + self.pad_height_divisor - 1) // self.pad_height_divisor) * self.pad_height_divisor
        target_w = ((w + self.pad_width_divisor - 1) // self.pad_width_divisor) * self.pad_width_divisor

        pad_h = target_h - h
        pad_w = target_w - w

        # Pad (left, top, right, bottom)
        padding = (0, 0, pad_w, pad_h)

        # Warning: this method returns the batched image, regardless if its input was batched or not
        return F.pad(img, padding, padding_mode="reflect")