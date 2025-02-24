### Install DeepLabCut-live on a desktop (Windows/Ubuntu)

We recommend that you install DeepLabCut-live in a conda environment (It is a standard python package though, and other distributions will also likely work). In this case, please install Anaconda:

- [Windows](https://docs.anaconda.com/anaconda/install/windows/)
- [Linux](https://docs.anaconda.com/anaconda/install/linux/)

Create a conda environment with python 3.7 and tensorflow:

New version:
```
conda create -n dlc-live python=3.8
conda activate dlc-live
conda install -c conda-forge pytables==3.8.0
pip install "tensorflow-macos<2.13.0" "tensorflow-metal" "tensorpack>=0.11" "tf_slim>=1.1.0"
pip install deeplabcut-live
dlc-live-test
```

Activate the conda environment, install the DeepLabCut-live package, then test the installation:

```
conda activate dlc-live
pip install deeplabcut-live
dlc-live-test
```

Note, you can also just run the test:

`dlc-live-test`

If installed properly, this script will i) create a temporary folder ii) download the full_dog model from the [DeepLabCut Model Zoo](http://www.mousemotorlab.org/dlc-modelzoo), iii) download a short video clip of a dog, and iv) run inference while displaying keypoints. v) remove the temporary folder.

Please note, you also should have curl installed on your computer (typically this is already installed on your system), but just in case, just run `sudo apt install curl`
