# DeepLabCut-live SDK<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="350" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">

![PyPI - Python Version](https://img.shields.io/pypi/v/deeplabcut-live)
[![License](https://img.shields.io/pypi/l/deeplabcutcore.svg)](https://github.com/DeepLabCut/deeplabcutlive/raw/master/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deeplabcut-live?color=purple)
![Python package](https://github.com/DeepLabCut/DeepLabCut-live/workflows/Python%20package/badge.svg)

This package contains a DeepLabCut inference pipeline for real-time applications that has minimal (software) dependencies. Thus, it is as easy to install as possible (in particular, on atypical systems like NVIDIA Jetson boards).

**Performance:** If you would like to see estimates on how your model might perform given a video size, neural network type, and hardware, please see: https://deeplabcut.github.io/DLC-inferencespeed-benchmark/ And, consider submitting your results too! https://github.com/DeepLabCut/DLC-inferencespeed-benchmark

**What this SDK provides:** This package provides a `DLCLive` class which enables pose estimation online to provide feedback. This object loads and prepares a DeepLabCut network for inference, and will return the predicted pose for single images.

To perform processing on poses (such as predicting the future pose of an animal given it's current pose, or to trigger external hardware like send TTL pulses to a laser for optogenetic stimulation), this object takes in a `Processor` object. Processor objects must contain two methods: process and save.

- The `process` method takes in a pose, performs some processing, and returns processed pose.
- The `save` method saves any valuable data created by or used by the processor

For more details and examples, see documentation [here](dlclive/processor/README.md).

###### Note :: alone, this object does not record video or capture images from a camera. This must be done separately, i.e. see our [DeepLabCut-live GUI](https://github.com/gkane26/DeepLabCut-live-GUI).


### Installation:

Please see our instruction manual to install on a [Windows or Linux machine](docs/install_desktop.md) or on a [NVIDIA Jetson Development Board](docs/install_jetson.md)

- available on pypi as: `pip install deeplabcut-live`


### Quick Start: instructions for use:

1. Initialize `Processor` (if desired)
2. Initialize the `DLCLive` object
3. Perform pose estimation!

```python
from dlclive import DLCLive, Processor
dlc_proc = Processor()
dlc_live = DLCLive(<path to exported model directory>, processor=dlc_proc)
dlc_live.init_inference(<your image>)
dlc_live.get_pose(<your image>)
```

`DLCLive` **parameters:**

  - `path` = string; full path to the exported DLC model directory
  - `model_type` = string; the type of model to use for inference. Types include:
      - `base` = the base DeepLabCut model
      - `tensorrt` = apply [tensor-rt](https://developer.nvidia.com/tensorrt) optimizations to model
      - `tflite` = use [tensorflow lite](https://www.tensorflow.org/lite) inference (in progress...)
  - `cropping` = list of int, optional; cropping parameters in pixel number: [x1, x2, y1, y2]
  - `dynamic` = tuple, optional; defines parameters for dynamic cropping of images
      - `index 0` = use dynamic cropping, bool
      - `index 1` = detection threshold, float
      - `index 2` = margin (in pixels) around identified points, int
  - `resize` = float, optional; factor by which to resize image (resize=0.5 downsizes both width and height of image by half). Can be used to downsize large images for faster inference
  - `processor` = dlc pose processor object, optional
  - `display` = bool, optional; display processed image with DeepLabCut points? Can be used to troubleshoot cropping and resizing parameters, but is very slow

`DLCLive` **inputs:**

  - `<path to exported model directory>` = path to the folder that has the `.pb` files that you acquire after running `deeplabcut.export_model`
  - `<your image>` = is a numpy array of each frame


### Benchmarking/Analyzing Exported DeepLabCut Models

DeepLabCut-live offers some analysis tools that allow users to peform the following operations on videos, from python or from the command line: 
1. Test inference speed across a range of image sizes, downsizing images by specifying the `resize` or `pixels` parameter. Using the `pixels` parameter will resize images to the desired number of `pixels`, without changing the aspect ratio. Results will be saved (along with system info) to a pickle file if you specify an output directory.
##### python
```python
dlclive.benchmark_videos('/path/to/exported/model', ['/path/to/video1', '/path/to/video2'], output='/path/to/output', resize=[1.0, 0.75, '0.5'])
```
##### command line
```
dlc-live-benchmark /path/to/exported/model /path/to/video1 /path/to/video2 -o /path/to/output -r 1.0 0.75 0.5
```

2. Display keypoints to visually inspect the accuracy of exported models on different image sizes (note, this is slow and only for testing purposes):

##### python
```python
dlclive.benchmark_videos('/path/to/exported/model', '/path/to/video', resize=0.5, display=True, pcutoff=0.5, display_radius=4, cmap='bmy')
```
##### command line
```
dlc-live-benchmark /path/to/exported/model /path/to/video -r 0.5 --display --pcutoff 0.5 --display-radius 4 --cmap bmy
```

3. Analyze and create a labeled video using the exported model and desired resize parameters. This option functions similar to `deeplabcut.benchmark_videos` and `deeplabcut.create_labeled_video` (note, this is slow and only for testing purposes).

##### python
```python
dlclive.benchmark_videos('/path/to/exported/model', '/path/to/video', resize=[1.0, 0.75, 0.5], pcutoff=0.5, display_radius=4, cmap='bmy', save_poses=True, save_video=True)
```
##### command line
```
dlc-live-benchmark /path/to/exported/model /path/to/video -r 0.5 --pcutoff 0.5 --display-radius 4 --cmap bmy --save-poses --save-video
```
  
### Citation:

If you find our code helpful, please consider citing:
```
@Article{Kane2020dlclive,
  author    = {Kane, Gary and Lopes, Gonçalo and Sanders, Jonny and Mathis, Alexander and Mathis, Mackenzie},
  title     = {Real-time DeepLabCut for closed-loop feedback based on posture},
  journal   = {BioRxiv},
  year      = {2020},
}
```
