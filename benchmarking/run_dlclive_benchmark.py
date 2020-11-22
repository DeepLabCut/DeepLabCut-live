"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

# Script for running the official benchmark from Kane et al, 2020.
# Please share your results at https://github.com/DeepLabCut/DLC-inferencespeed-benchmark

import os, pathlib
import glob

from dlclive import benchmark_videos, download_benchmarking_data

datafolder = os.path.join(
    pathlib.Path(__file__).parent.absolute(), "Data-DLC-live-benchmark"
)

if not os.path.isdir(datafolder):  # only download if data doesn't exist!
    # Downloading data.... this takes a while (see terminal)
    download_benchmarking_data(datafolder)

n_frames = 10000  # change to 10000 for testing on a GPU!
pixels = [2500, 10000, 40000, 160000, 320000, 640000]

dog_models = glob.glob(datafolder + "/dog/*[!avi]")
dog_video = glob.glob(datafolder + "/dog/*.avi")[0]
mouse_models = glob.glob(datafolder + "/mouse_lick/*[!avi]")
mouse_video = glob.glob(datafolder + "/mouse_lick/*.avi")[0]

this_dir = os.path.dirname(os.path.realpath(__file__))
# storing results in /benchmarking/results: (for your PR)
out_dir = os.path.normpath(this_dir + "/results")

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for m in dog_models:
    benchmark_videos(m, dog_video, output=out_dir, n_frames=n_frames, pixels=pixels)

for m in mouse_models:
    benchmark_videos(m, mouse_video, output=out_dir, n_frames=n_frames, pixels=pixels)
