"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import os
import sys
import shutil
import warnings

from dlclive import benchmark_videos
import urllib.request
import argparse
from pathlib import Path
import tarfile


def urllib_pbar(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    outstr = f"{round(percent)}%"
    sys.stdout.write(outstr)
    sys.stdout.write("\b"*len(outstr))
    sys.stdout.flush()

def main(display:bool=None):
    parser = argparse.ArgumentParser(
        description="Test DLC-Live installation by downloading and evaluating a demo DLC project!")
    parser.add_argument('--nodisplay', action='store_false', help="Run the test without displaying tracking")
    args = parser.parse_args()

    if display is None:
        display = args.nodisplay

    if not display:
        print('Running without displaying video')

    # make temporary directory in $HOME
    print("\nCreating temporary directory...\n")
    tmp_dir = Path().home() / 'dlc-live-tmp'
    tmp_dir.mkdir(mode=0o775,exist_ok=True)

    video_file = str(tmp_dir / 'dog_clip.avi')
    model_tarball = tmp_dir / 'DLC_Dog_resnet_50_iteration-0_shuffle-0.tar.gz'
    model_dir = model_tarball.with_suffix('').with_suffix('') # remove two suffixes (tar.gz)


    # download dog test video from github:
    print(f"Downloading Video to {video_file}")
    url_link = "https://github.com/DeepLabCut/DeepLabCut-live/blob/master/check_install/dog_clip.avi?raw=True"
    urllib.request.urlretrieve(url_link, video_file, reporthook=urllib_pbar)

    # download exported dog model from DeepLabCut Model Zoo
    if Path(model_tarball).exists():
        print('Tarball already downloaded, using cached version')
    else:
        print("Downloading full_dog model from the DeepLabCut Model Zoo...")
        model_url = "http://deeplabcut.rowland.harvard.edu/models/DLC_Dog_resnet_50_iteration-0_shuffle-0.tar.gz"
        urllib.request.urlretrieve(model_url, str(model_tarball), reporthook=urllib_pbar)

    print('Untarring compressed model')
    model_file = tarfile.open(str(model_tarball))
    model_file.extractall(str(model_dir.parent))
    model_file.close()

    # assert these things exist so we can give informative error messages
    assert Path(video_file).exists()
    assert Path(model_dir).exists() and Path(model_dir).is_dir()

    # run benchmark videos
    print("\n Running inference...\n")
    # model_dir = "DLC_Dog_resnet_50_iteration-0_shuffle-0"
    # print(video_file)
    benchmark_videos(str(model_dir), video_file, display=display, resize=0.5, pcutoff=0.25)

    # deleting temporary files
    print("\n Deleting temporary files...\n")
    try:
        shutil.rmtree(tmp_dir)
    except PermissionError:
        warnings.warn(f'Could not delete temporary directory {str(tmp_dir)} due to a permissions error, but otherwise dlc-live seems to be working fine!')

    print("\nDone!\n")


if __name__ == "__main__":


    display = args.nodisplay


    main(display=args.nodisplay)
