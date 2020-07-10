# Example script for running benchmark tests in Kane et al, 2020.

import os
import glob

from dlclive import benchmark_model_by_size

# Update the datafolder to where the data is:
datafolder='/media/alex/dropboxdisk/DLC-LiveBenchmarking/DLC-live-benchmarkingdata'


dog_models = glob.glob(datafolder + '/dog/*[!avi]')
dog_video = glob.glob(datafolder + '/dog/*.avi')[0]
mouse_models = glob.glob(datafolder + '/mouse_lick/*[!avi]')
mouse_video = glob.glob(datafolder + '/mouse_lick/*.avi')[0]

this_dir = os.path.dirname(os.path.realpath(__file__))
#storing results in /benchmarking/results: (for your PR)
out_dir = os.path.normpath(this_dir + '/results')

n_frames = 1000 #change to 10000 for GPU!
pixels = [2500, 10000, 40000, 160000, 320000, 640000]
ind = 1

for ind_m, m in enumerate(dog_models):
    print("\n\nMODEL {:d} / 8".format(ind_m))
    benchmark_model_by_size(m, dog_video, ind_m, out_dir=out_dir, n_frames=n_frames, pixels=pixels)

offset=ind_m+1

for ind_m, m in enumerate(mouse_models):
    print("\n\nMODEL {:d} / 8".format(ind_m))
    benchmark_model_by_size(m, mouse_video, ind_m + offset, out_dir=out_dir, n_frames=n_frames, pixels=pixels)
