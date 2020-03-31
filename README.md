# DeepLabCut-live

This package contains a DeepLabCut inference pipeline that has minimal (software) dependencies. Thus, it is as easy to install as possible (in particular on unypical systems like embeeded systems). This package contains a DLCLive class which enables pose estimation online to provide feedback. This object loads and prepares a DLC network for inference, and will return the predicted pose for single images.

To perform processing on poses (such as predicting the future pose of an animal given it's current pose, or to trigger external hardware like send TTL pulses to a laser for optogenetic stimulation), this object takes in a `Processor` object. Processor objects must contain two methods: process and save.
- The `process` method takes in a pose, performs some processing, and returns processed pose.
- The `save` method saves any valuable data created by or used by the processor
For examples, please see the [processor directory](processor)

###### Note :: this object does not record video or capture images from a camera. This must be done separately.

DLCLive parameters:
  - path = string; full path to the exported DLC model directory
  - model_type = string; the type of model to use for inference. Types include:
    - 'base' = the base DeepLabCut model
    - 'tensorrt' = apply [tensorrt](https://developer.nvidia.com/tensorrt) optimizations to model
    - 'tflite' = use [tensorflow lite](https://www.tensorflow.org/lite) inference (in progress...)
  - cropping = list of int, optional; cropping parameters in pixel number: [x1, x2, y1, y2]
  - dynamic = tuple, optional; defines parameters for dynamic cropping of images
    - index 0 = use dynamic cropping, bool
    - index 1 = detection threshold, float
    - index 2 = margin (in pixels) around identified points, int
  - resize = float, optional; factor by which to resize image (resize=0.5 downsizes both width and height of image by half). Can be used to downsize large images for faster inference
  - processor = dlc pose processor object, optional
  - display = bool, optional; display processed image with DeepLabCut points? Can be used to troubleshoot cropping and resizing parameters, but is very slow


### Installation:

Please see our instruction manual to install on a [Windows or Linux machine](docs/install_desktop.md) or on a [NVIDIA Jetson Development Board](docs/install_jetson.md)


### Instructions for use:

1. Initialize processor (if desired)
2. Initialize the DLCLive object
3. Perform pose estimation

```
from dlclive import DLCLive, Processor
dlc_proc = Processor()
dlc_live = DLCLive(<path to exported model directory>, processor=dlc_proc)
dlc_live.get_pose(<your image>)
```
