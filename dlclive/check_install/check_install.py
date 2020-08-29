"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import os
import shutil
from dlclive import benchmark_videos
import urllib.request


def main():

    # make temporary directory in $HOME
    print("\nCreating temporary directory...\n")
    home = os.path.expanduser("~")
    tmp_dir = os.path.normpath(f"{home}/dlc-live-tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)

    # download dog test video from github:
    url_link = "https://github.com/DeepLabCut/DeepLabCut-live/blob/master/check_install/dog_clip.avi?raw=True"
    urllib.request.urlretrieve(url_link, "dog_clip.avi")
    video_file = os.path.join(url_link, "dog_clip.avi")

    # download exported dog model from DeepLabCut Model Zoo
    print("Downloading full_dog model from the DeepLabCut Model Zoo...")
    model_url = "http://deeplabcut.rowland.harvard.edu/models/DLC_Dog_resnet_50_iteration-0_shuffle-0.tar.gz"
    os.system(f"curl {model_url} | tar xvz")

    # run benchmark videos
    print("\n Running inference...\n")
    model_dir = "DLC_Dog_resnet_50_iteration-0_shuffle-0"
    print(video_file)
    benchmark_videos(model_dir, video_file, display=True, resize=0.5, pcutoff=0.25)

    # deleting temporary files
    print("\n Deleting temporary files...\n")
    shutil.rmtree(tmp_dir)

    print("\nDone!\n")


if __name__ == "__main__":
    main()
