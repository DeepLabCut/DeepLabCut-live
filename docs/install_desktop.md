### Install DeepLabCut-live on a desktop (Windows/Ubuntu)

We recommend that you install DeepLabCut-live in a conda environment. First, please install Anaconda:

- [Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Linux](https://docs.anaconda.com/anaconda/install/linux/)

Create a conda environment with python 3.7 and tensorflow:

```
conda create -n dlc-live python=3.7 tensorflow-gpu==1.13.1 # if using GPU
conda create -n dlc-live python=3.7 tensorflow==1.13.1 # if not using GPU
```

Activate the conda environment, install the DeepLabCut-live package, then test the installation:

```
conda activate dlc-live
pip install deeplabcut-live
dlc-live-test
```

If installed properly, this script will i) download the full_dog model from the DeepLabCut Model Zoo, ii) download a short video clip of a dog, and iii) run inference while displaying keypoints.
