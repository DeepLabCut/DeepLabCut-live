"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import os
import urllib.request
import argparse
import shutil

import urllib.request
import warnings
from pathlib import Path

from dlclive.utils import download_file
from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model

import dlclive
from dlclive.benchmark import benchmark_videos
from dlclive.engine import Engine

MODEL_NAME = "superanimal_quadruped"
SNAPSHOT_NAME = "snapshot-700000.pb"


def main():
    parser = argparse.ArgumentParser(
        description="Test DLC-Live installation by downloading and evaluating a demo DLC project!"
    )
    parser.add_argument(
        "--nodisplay",
        action="store_false",
        help="Run the test without displaying tracking",
    )
    args = parser.parse_args()
    display = args.nodisplay

    if not display:
        print("Running without displaying video")

    # make temporary directory
    print("\nCreating temporary directory...\n")
    tmp_dir = Path(dlclive.__file__).parent / "check_install" / "dlc-live-tmp"
    tmp_dir.mkdir(mode=0o775, exist_ok=True)

    video_file = str(tmp_dir / "dog_clip.avi")
    model_dir = tmp_dir / "DLC_Dog_resnet_50_iteration-0_shuffle-0"

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

    # download model from the DeepLabCut Model Zoo
    if Path(model_dir / SNAPSHOT_NAME).exists():
        print("Model already downloaded, using cached version")
    else:
        print("Downloading a test model from the DeepLabCut Model Zoo...")
        download_huggingface_model(MODEL_NAME, model_dir)

    # assert these things exist so we can give informative error messages
    assert Path(video_file).exists(), f"Missing video file {video_file}"
    assert Path(
        model_dir / SNAPSHOT_NAME
    ).exists(), f"Missing model file {model_dir / SNAPSHOT_NAME}"

    # run benchmark videos
    print("\n Running inference...\n")
    benchmark_videos(
        model_path=str(model_dir),
        model_type="base" if Engine.from_model_path(model_dir) == Engine.TENSORFLOW else "pytorch",
        video_path=video_file,
        display=display,
        resize=0.5,
        pcutoff=0.25
    )

    # deleting temporary files
    print("\n Deleting temporary files...\n")
    try:
        shutil.rmtree(tmp_dir)
    except PermissionError:
        warnings.warn(
            f"Could not delete temporary directory {str(tmp_dir)} due to a permissions error, but otherwise dlc-live seems to be working fine!"
        )

    print("\nDone!\n")


if __name__ == "__main__":
    main()
