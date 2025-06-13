### Install DeepLabCut-live on a desktop (Windows/Ubuntu)

We recommend that you install DeepLabCut-live in a conda environment (It is a standard
python package though, and other distributions will also likely work). In this case,
please install [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) 
(recommended) or Anaconda.

If you have an Nvidia GPU and want to use its capabilities, you'll need to [install CUDA
](https://developer.nvidia.com/cuda-downloads) first (check that CUDA is installed - 
checkout the installation guide for [linux](
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) or [Windows](
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

Create a conda environment with python 3.10 or 3.11, and install 
[`pytables`](https://www.pytables.org/usersguide/installation.html), `torch` and 
`torchvision`. Make sure you [install the correct `torch` and `torchvision` versions
for your compute platform](https://pytorch.org/get-started/locally/)!

```
conda create -n dlc-live python=3.11 
conda activate dlc-live
conda install -c conda-forge pytables==3.8.0

# Installs PyTorch on Linux with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Activate the conda environment, install the DeepLabCut-live package, then test the
installation:

```
conda activate dlc-live
pip install deeplabcut-live
dlc-live-test
```

Note, you can also just run the test:

`dlc-live-test`

If installed properly, this script will i) create a temporary folder ii) download the
full_dog model from the [DeepLabCut Model Zoo](
http://www.mousemotorlab.org/dlc-modelzoo), iii) download a short video clip of
a dog, and iv) run inference while displaying keypoints. v) remove the temporary folder.

Please note, you also should have curl installed on your computer (typically this is
already installed on your system), but just in case, just run `sudo apt install curl`
