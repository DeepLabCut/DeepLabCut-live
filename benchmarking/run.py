"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

# Example script for running benchmark tests in Kane et al, 2020.

import os
import glob

from dlclive import benchmark_videos

# Update the datafolder to where the data is:
datafolder = "/your/path/to/data/here"

n_frames = 1000  # change to 10000 for testing on a GPU!
pixels = [2500, 10000, 40000, 160000, 320000, 640000]

dog_models = glob.glob(datafolder + "/dog/*[!avi]")
dog_video = glob.glob(datafolder + "/dog/*.avi")[0]
mouse_models = glob.glob(datafolder + "/mouse_lick/*[!avi]")
mouse_video = glob.glob(datafolder + "/mouse_lick/*.avi")[0]

this_dir = os.path.dirname(os.path.realpath(__file__))
# storing results in /benchmarking/results: (for your PR)
out_dir = os.path.normpath(this_dir + "/results")

for ind_m, m in enumerate(dog_models):
    print("\n\nMODEL {:d} / 8".format(ind_m))
    benchmark_videos(
        m, dog_video, ind_m, output=out_dir, n_frames=n_frames, pixels=pixels
    )

offset = ind_m + 1

for ind_m, m in enumerate(mouse_models):
    print("\n\nMODEL {:d} / 8".format(ind_m))
    benchmark_videos(
        m,
        mouse_video,
        ind_m + offset,
        output=out_dir,
        n_frames=n_frames,
        pixels=pixels,
    )
