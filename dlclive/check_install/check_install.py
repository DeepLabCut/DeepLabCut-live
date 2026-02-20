"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import argparse
import shutil
import warnings
import urllib.error
from pathlib import Path

from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model

from dlclive.utils import download_file
from dlclive.benchmark import benchmark_videos
from dlclive.engine import Engine
from dlclive.utils import get_available_backends

MODEL_NAME = "superanimal_quadruped"
SNAPSHOT_NAME = "snapshot-700000.pb"
TMP_DIR = Path(__file__).parent / "dlc-live-tmp"

MODELS_DIR = TMP_DIR / "test_models"
TORCH_MODEL = "resnet_50"
TORCH_CONFIG = {
    "checkpoint": MODELS_DIR / f"exported_quadruped_{TORCH_MODEL}.pt",
    "super_animal": "superanimal_quadruped",
}
TF_MODEL_DIR = MODELS_DIR / "DLC_Dog_resnet_50_iteration-0_shuffle-0"


def run_pytorch_test(video_file: str, display: bool = False):
    from dlclive.modelzoo.pytorch_model_zoo_export import export_modelzoo_model

    if Engine.PYTORCH not in get_available_backends():
        raise ImportError(
            "PyTorch backend is not available. Please ensure PyTorch is installed to run the PyTorch test."
        )
    # Download model from the DeepLabCut Model Zoo
    export_modelzoo_model(
        export_path=TORCH_CONFIG["checkpoint"],
        super_animal=TORCH_CONFIG["super_animal"],
        model_name=TORCH_MODEL,
    )
    if not TORCH_CONFIG["checkpoint"].exists():
        raise FileNotFoundError(
            f"Failed to export {TORCH_CONFIG['super_animal']} model"
        )
    if TORCH_CONFIG["checkpoint"].stat().st_size == 0:
        raise ValueError(f"Exported {TORCH_CONFIG['super_animal']} model is empty")
    benchmark_videos(
        model_path=str(TORCH_CONFIG["checkpoint"]),
        model_type="pytorch",
        video_path=video_file,
        display=display,
        pcutoff=0.25,
        pixels=1000,
    )


def run_tensorflow_test(video_file: str, display: bool = False):
    if Engine.TENSORFLOW not in get_available_backends():
        raise ImportError(
            "TensorFlow backend is not available. Please ensure TensorFlow is installed to run the TensorFlow test."
        )
    model_dir = TF_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    if Path(model_dir / SNAPSHOT_NAME).exists():
        print("Model already downloaded, using cached version")
    else:
        print(
            "Downloading superanimal_quadruped model from the DeepLabCut Model Zoo..."
        )
        download_huggingface_model(MODEL_NAME, str(model_dir))

    assert Path(model_dir / SNAPSHOT_NAME).exists(), (
        f"Missing model file {model_dir / SNAPSHOT_NAME}"
    )

    benchmark_videos(
        model_path=str(model_dir),
        model_type="base",
        video_path=video_file,
        display=display,
        pcutoff=0.25,
        pixels=1000,
    )


def main():
    backend_results = {}

    parser = argparse.ArgumentParser(
        description="Test DLC-Live installation by downloading and evaluating a demo DLC project!"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Run the test and display tracking",
    )
    parser.add_argument(
        "--nodisplay",
        action="store_false",
        dest="display",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    display = args.display

    if not display:
        print("Running without displaying video")

    # make temporary directory
    print("\nCreating temporary directory...\n")
    tmp_dir = TMP_DIR
    tmp_dir.mkdir(mode=0o775, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    video_file = str(tmp_dir / "dog_clip.avi")

    try:
        # download dog test video from github:
        # Use raw.githubusercontent.com for direct file access
        if not Path(video_file).exists():
            print(f"Downloading Video to {video_file}")
            url_link = "https://raw.githubusercontent.com/DeepLabCut/DeepLabCut-live/master/check_install/dog_clip.avi"
            try:
                download_file(url_link, video_file)
            except (urllib.error.URLError, IOError) as e:
                raise RuntimeError(f"Failed to download video file: {e}") from e
        else:
            print(f"Video file already exists at {video_file}, skipping download.")

        # assert these things exist so we can give informative error messages
        if not Path(video_file).exists():
            raise FileNotFoundError(f"Missing video file {video_file}")
        backend_failures = {}
        any_backend_succeeded = False

        for backend in get_available_backends():
            try:
                if backend == Engine.PYTORCH:
                    print("\nRunning PyTorch test...\n")
                    run_pytorch_test(video_file, display=display)
                    any_backend_succeeded = True
                    backend_results["pytorch"] = ("SUCCESS", None)
                elif backend == Engine.TENSORFLOW:
                    print("\nRunning TensorFlow test...\n")
                    run_tensorflow_test(video_file, display=display)
                    any_backend_succeeded = True
                    backend_results["tensorflow"] = ("SUCCESS", None)
                else:
                    warnings.warn(
                        f"Unrecognized backend {backend}, skipping...", UserWarning
                    )
            except Exception as e:
                backend_name = (
                    "pytorch"
                    if backend == Engine.PYTORCH
                    else "tensorflow"
                    if backend == Engine.TENSORFLOW
                    else str(backend)
                )
                backend_results[backend_name] = ("ERROR", str(e))
                backend_failures[backend] = e
                warnings.warn(
                    f"Error while running test for backend {backend}: {e}. "
                    "Continuing to test other available backends.",
                    UserWarning,
                )

        print("\n---\nBackend test summary:")
        for name in ("tensorflow", "pytorch"):
            status, _ = backend_results.get(name, ("SKIPPED", None))
            print(f"{name:<11} [{status}]")
        print("---")
        for name, (status, error) in backend_results.items():
            if status == "ERROR":
                print(f"{name.capitalize()} error:\n{error}\n")

        if not any_backend_succeeded and backend_failures:
            failure_messages = "; ".join(
                f"{b}: {exc}" for b, exc in backend_failures.items()
            )
            raise RuntimeError(f"All backend tests failed. Details: {failure_messages}")

    finally:
        # deleting temporary files
        print("\n Deleting temporary files...\n")
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
        except PermissionError:
            warnings.warn(
                f"Could not delete temporary directory {str(tmp_dir)} due to a permissions error."
            )


if __name__ == "__main__":
    # Get available backends (emits a warning if neither TensorFlow nor PyTorch is installed)
    available_backends: list[Engine] = get_available_backends()
    print(f"Available backends: {[b.value for b in available_backends]}")
    if len(available_backends) == 0:
        raise NotImplementedError(
            "Neither TensorFlow nor PyTorch is installed. Please install at least one of these frameworks to run the installation test."
        )

    main()
