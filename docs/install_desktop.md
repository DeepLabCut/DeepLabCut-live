### Install DeepLabCut-live on a desktop (Windows/Ubuntu)

We recommend that you install DeepLabCut-live in a conda environment. First, please install Anaconda:
- [Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Linux](https://docs.anaconda.com/anaconda/install/linux/)

Create a conda environment with python 3.7 and tensorflow:
```
conda create -n dlc-live python=3.7 tensorflow-gpu==1.13.1 # if using GPU
conda create -n dlc-live python=3.7 tensorflow==1.13.1 # if not using GPU
```

Activate the conda environment and install the DeepLabCut-live package:
```
conda activate dlc-live
pip install git+https://github.com/AlexEMG/DeepLabCut-live.git
```

We also recommend that you have OpenCV installed:
```
pip install opencv-python
```
