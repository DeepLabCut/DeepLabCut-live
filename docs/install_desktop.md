# Installation Guide

DeepLabCut-live can be installed using several methods. **For most users, we recommend installing from PyPI using `uv` or `pip`** - this is the simplest and most reliable method.



## System Requirements

- Python 3.10, 3.11, or 3.12
- For GPU support: CUDA-compatible NVIDIA GPU (see [CUDA installation guide](https://developer.nvidia.com/cuda-downloads))


### Important Notes

- **Required dependencies**: You must install either PyTorch or TensorFlow. Choose the one that matches your exported model:
  - PyTorch models: Use `deeplabcut-live[pytorch]`
  - TensorFlow models: Use `deeplabcut-live[tf]`
- **Tensorflow support**: Tensorflow is not supported for python versions >=3.11 on Windows machines, please use python 3.10 instead.
- **Conda environments**: When using `pip`, it's recommended to install in a separate conda environment to avoid dependency conflicts.
- **uv sync**: If you have cloned the repository and are using `uv`, you can use `uv sync --extra pytorch` (or `--extra tf`) to install all dependencies automatically.

## 1. Quick Start (recommended)

Install from PyPI with PyTorch or TensorFlow:
   ```bash
   # With PyTorch (recommended)
   pip install deeplabcut-live[pytorch]

   # Or with TensorFlow
   pip install deeplabcut-live[tf]

   # Or using uv
   uv pip install deeplabcut-live[pytorch] # or [tf]
   ```

### Windows-users with GPU:
On **Windows**, the `deeplabcut-live[pytorch]` extra will not install the required CUDA-enabled wheels for PyTorch by default. Windows users with a CUDA GPU should install CUDA-enabled PyTorch first:


   ```bash
   # First install PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

   # Then install DeepLabCut-live (it will use the existing GPU-enabled PyTorch)
   pip install deeplabcut-live[pytorch]
   ```


## 2. Install from Git Repository
If you want to install from a local clone of the repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/DeepLabCut/DeepLabCut-live.git
   cd DeepLabCut-live
   ```

2. Install in editable mode:

    **Option A - with uv (recommended; fastest & dependencies are handled automatically)**
   ```bash
   # Using uv
   uv sync --extra pytorch --python 3.11 # or --extra tf for TensorFlow
   ```

   **Option B - using pip in a conda environment**
   ```
   conda create -n dlc-live python=3.11
   conda activate dlc-live
   pip install -e ".[pytorch]"  # or ".[tf]" for TensorFlow
   ```

## 3. Separate Pytorch or Tensorflow installation
If the above instructions do not work, or you want to use a specific Pytorch or Tensorflow version. You can first install Pytorch or Tensorflow separately.
   ```
   # Install Pytorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

   # Or install TensorFlow instead
   pip install tensorflow
   ```
and then install DeepLabCut-live following the instructions in option 1. or 2. above.

## How to install uv or Conda?
- **uv**: See more information on the website  [here](https://docs.astral.sh/uv/#installation). The basic installation commands are:
    - for Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - for Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **conda**: Follow the instructions [here](https://docs.anaconda.com/miniconda/miniconda-install/).




## Verify DeepLabCut-live Installation

Test your installation by running:

```bash
dlc-live-test
```

If installed properly, this script will i) create a temporary folder ii) download the
full_dog model from the [DeepLabCut Model Zoo](
http://www.mousemotorlab.org/dlc-modelzoo), iii) download a short video clip of
a dog, and iv) run inference while displaying keypoints. v) remove the temporary folder.
