# DeepLabCut-live! SDK<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1606082050387-M8M1CFI5DFUZCBAAUI0W/ke17ZwdGBToddI8pDm48kLuMKy7Ws6mFofiFehYynfdZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpzp2tFVMcEgqZM8QO7VXXQogrsLnYKC4n4YnYuHC1HMRWygQlqMNAoTF9HaycikLeg/DLClive.png?format=750w" width="350" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
![PyPI - Python Version](https://img.shields.io/pypi/v/deeplabcut-live)
[![Downloads](https://pepy.tech/badge/deeplabcut-live)](https://pepy.tech/project/deeplabcut-live)
[![Downloads](https://pepy.tech/badge/deeplabcut-live/month)](https://pepy.tech/project/deeplabcut-live)
![Python package](https://github.com/DeepLabCut/DeepLabCut-live/workflows/Python%20package/badge.svg)
[![GitHub stars](https://img.shields.io/github/stars/DeepLabCut/DeepLabCut-live.svg?style=social&label=Star)](https://github.com/DeepLabCut/DeepLabCut-live)
[![GitHub forks](https://img.shields.io/github/forks/DeepLabCut/DeepLabCut-live.svg?style=social&label=Fork)](https://github.com/DeepLabCut/DeepLabCut-live)
[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut)
[![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Twitter Follow](https://img.shields.io/twitter/follow/DeepLabCut.svg?label=DeepLabCut&style=social)](https://twitter.com/DeepLabCut)

This package contains a [DeepLabCut](http://www.mousemotorlab.org/deeplabcut) inference pipeline for real-time applications that has minimal (software) dependencies. Thus, it is as easy to install as possible (in particular, on atypical systems like [NVIDIA Jetson boards](https://developer.nvidia.com/buy-jetson)).

**Performance:** If you would like to see estimates on how your model should perform given different video sizes, neural network type, and hardware, please see: https://deeplabcut.github.io/DLC-inferencespeed-benchmark/

If you have different hardware, please consider submitting your results too! https://github.com/DeepLabCut/DLC-inferencespeed-benchmark

**What this SDK provides:** This package provides a `DLCLive` class which enables pose estimation online to provide feedback. This object loads and prepares a DeepLabCut network for inference, and will return the predicted pose for single images. 

To perform processing on poses (such as predicting the future pose of an animal given it's current pose, or to trigger external hardware like send TTL pulses to a laser for optogenetic stimulation), this object takes in a `Processor` object. Processor objects must contain two methods: process and save.

- The `process` method takes in a pose, performs some processing, and returns processed pose.
- The `save` method saves any valuable data created by or used by the processor

For more details and examples, see documentation [here](dlclive/processor/README.md).

###### 🔥🔥🔥🔥🔥 Note :: alone, this object does not record video or capture images from a camera. This must be done separately, i.e. see our [DeepLabCut-live GUI](https://github.com/gkane26/DeepLabCut-live-GUI).🔥🔥🔥

### News! 
- March 2022: DeepLabCut-Live! 1.0.2 supports poetry installation `poetry install deeplabcut-live`, thanks to PR #60.
- March 2021: DeepLabCut-Live! [**version 1.0** is released](https://pypi.org/project/deeplabcut-live/), with support for tensorflow 1 and tensorflow 2!
- Feb 2021: DeepLabCut-Live! was featured in **Nature Methods**: ["Real-time behavioral analysis"](https://www.nature.com/articles/s41592-021-01072-z)
- Jan 2021: full **eLife** paper is published: ["Real-time, low-latency closed-loop feedback using markerless posture tracking"](https://elifesciences.org/articles/61909)
- Dec 2020: we talked to **RTS Suisse Radio** about DLC-Live!: ["Capture animal movements in real time"](https://www.rts.ch/play/radio/cqfd/audio/capturer-les-mouvements-des-animaux-en-temps-reel?id=11782529)


### Installation:

Please see our instruction manual to install on a [Windows or Linux machine](docs/install_desktop.md) or on a [NVIDIA Jetson Development Board](docs/install_jetson.md). Note, this code works with tensorflow (TF) 1 or TF 2 models, but TF requires that whatever version you exported your model with, you must import with the same version (i.e., export with TF1.13, then use TF1.13 with DlC-Live; export with TF2.3, then use TF2.3 with DLC-live).

- available on pypi as: `pip install deeplabcut-live`

Note, you can then test your installation by running:

`dlc-live-test`

If installed properly, this script will i) create a temporary folder ii) download the full_dog model from the [DeepLabCut Model Zoo](http://www.mousemotorlab.org/dlc-modelzoo), iii) download a short video clip of a dog, and iv) run inference while displaying keypoints. v) remove the temporary folder.

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1606081086014-TG9GWH63ZGGOO7K779G3/ke17ZwdGBToddI8pDm48kHiSoSToKfKUI9t99vKErWoUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcOoIGycwr1shdgJWzLuxyzjLbSRGBFFxjYMBr42yCvRK5HHsLZWtMlAHzDU294nCd/dlclivetest.png?format=1000w" width="650" title="DLC-live-test" alt="DLC LIVE TEST" align="center" vspace = "50">

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


### Benchmarking/Analyzing your exported DeepLabCut models

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

## License:

This project is licensed under the GNU AGPLv3. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, we ask that you please cite us! This software is available for licensing via the EPFL Technology Transfer Office (https://tto.epfl.ch/, info.tto@epfl.ch).

## Community Support, Developers, & Help:

This is an actively developed package and we welcome community development and involvement.

- If you want to contribute to the code, please read our guide [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/CONTRIBUTING.md), which is provided at the main repository of DeepLabCut.

- We are a community partner on the [![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&amp;url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fdeeplabcut.json&amp;query=%24.topic_list.tags.0.topic_count&amp;colorB=brightgreen&amp;&amp;suffix=%20topics&amp;logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/deeplabcut). Please post help and support questions on the forum with the tag DeepLabCut. Check out their mission statement [Scientific Community Image Forum: A discussion forum for scientific image software](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000340).

- If you encounter a previously unreported bug/code issue, please post here (we encourage you to search issues first): https://github.com/DeepLabCut/DeepLabCut-live/issues

- For quick discussions here: [![Gitter](https://badges.gitter.im/DeepLabCut/community.svg)](https://gitter.im/DeepLabCut/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

### Reference:

If you utilize our tool, please [cite Kane et al, eLife 2020](https://elifesciences.org/articles/61909). The preprint is available here: https://www.biorxiv.org/content/10.1101/2020.08.04.236422v2

```
@Article{Kane2020dlclive,
  author    = {Kane, Gary and Lopes, Gonçalo and Sanders, Jonny and Mathis, Alexander and Mathis, Mackenzie},
  title     = {Real-time, low-latency closed-loop feedback using markerless posture tracking},
  journal   = {eLife},
  year      = {2020},
}
```

