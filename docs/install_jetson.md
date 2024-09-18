### Install DeepLabCut-live on a NVIDIA Jetson Development Kit

First, please follow NVIDIA's specific instructions to setup your Jetson Development Kit
(see [Jetson Development Kit User Guides](https://developer.nvidia.com/embedded/learn/getting-started-jetson)). Once you have installed the NVIDIA
Jetpack on your Jetson Development Kit, make sure all system libraries are up-to-date.
In a terminal, run:

```
sudo apt-get update
sudo apt-get upgrade
```

Lastly, please test that CUDA is installed properly by running: `nvcc --version`. The
output should say the version of CUDA installed on your Jetson.

#### Install python, virtualenv, and tensorflow

We highly recommend installing DeepLabCut-live in a virtual environment. Please run the
following command to install system dependencies needed to run python, to create virtual
environments, and to run tensorflow:

```
sudo apt-get update
sudo apt-get install libhdf5-serial-dev \
                     hdf5-tools \
                     libhdf5-dev \
                     zlib1g-dev \
                     zip \
                     libjpeg8-dev \
                     liblapack-dev \
                     libblas-dev \
                     gfortran \
                     python3-pip \
                     python3-venv \
                     python3-tk \
                     curl
```

#### Create a virtual environment

Next, create a virtual environment called `dlc-live`, activate the `dlc-live`
environment, and update it's package manager:

```
python3 -m venv dlc-live
source dlc-live/bin/activate
pip install -U pip testresources setuptools
```

#### Install DeepLabCut-live dependencies

First, install `python` dependencies to run `PyTorch` (from [NVIDIA instructions to 
install PyTorch for Jetson Platform](
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)). 
_This may take ~15-30 minutes._

```
pip3 install numpy==1.16.1 \
             future==0.17.1 \
             mock==3.0.5 \
             h5py==2.9.0 \
             keras_preprocessing==1.0.5 \
             keras_applications==1.0.8 \
             gast==0.2.2 \
             futures \
             protobuf \
             pybind11
```

Next, install PyTorch >= 2.0. This command will depend on the version of Jetpack you are
using. If you are uncertain, please refer to [NVIDIA's instructions](
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).
To install PyTorch >= 2.0

```
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/<torch_version_desired>
```

Currently, the only available PyTorch version that can be used is 
`torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl`. 


Lastly, copy the opencv-python bindings into your virtual environment:

```
cp -r /usr/lib/python3.12/dist-packages ~/dlc-live/lib/python3.12/dist-packages
```

#### Install the DeepLabCut-live package

Finally, please install DeepLabCut-live from PyPi (_this will take 3-5 mins_), then 
test the installation:

```
pip install deeplabcut-live
dlc-live-test
```

If installed properly, this script will i) download the full_dog model from the
DeepLabCut Model Zoo, ii) download a short video clip of a dog, and iii) run inference
while displaying keypoints.
