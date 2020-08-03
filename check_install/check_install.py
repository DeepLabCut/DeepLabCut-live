"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import os
import shutil
from dlclive import benchmark_videos


def main():

    # make temporary directory in $HOME
    print("\nCreating temporary directory...\n")
    home = os.path.expanduser("~")
    tmp_dir = os.path.normpath(f"{home}/dlc-live-tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)

    # download exported dog model from DeepLabCut Model Zoo
    print("Downloading full_dog model from the DeepLabCut Model Zoo...")
    model_url = "http://deeplabcut.rowland.harvard.edu/models/DLC_Dog_resnet_50_iteration-0_shuffle-0.tar.gz"
    os.system(f"curl {model_url} | tar xvz")

    # download dog video clip from github
    print("\nDownloading dog video clip...\n")
    #video_url = "https://github.com/DeepLabCut/DeepLabCut-live/raw/master/check_install/dog_clip.avi"
    video_url = '"https://docs.google.com/uc?export=download&id=1W_5AOl1SewXR2q5QC1K5Chm71I9LAmld"'
    os.system(f"curl -L {video_url} -o dog_clip.avi")

    # run benchmark videos
    print("\n Running inference...\n")
    model_dir = "DLC_Dog_resnet_50_iteration-0_shuffle-0"
    video_file = "dog_clip.avi"
    benchmark_videos(model_dir, video_file, display=True, resize=0.5, pcutoff=0.25)

    # deleting temporary files
    print("\n Deleting temporary files...\n")
    shutil.rmtree(tmp_dir)

    print("\nDone!\n")


if __name__ == "__main__":
    main()