# Example script for running benchmark tests in Kane et al, 2020. 

import os
import glob

from dlclive import benchmark_model_by_size

this_dir = os.path.dirname(os.path.realpath(__file__))

dog_models = glob.glob(this_dir + '/dog/*[!avi]')
dog_video = glob.glob(this_dir + '/dog/*.avi')[0]
mouse_models = glob.glob(this_dir + '/mouse_lick/*[!avi]')
mouse_video = glob.glob(this_dir + '/mouse_lick/*.avi')[0]
out_dir = os.path.normpath(this_dir + '/results')

n_frames = 100
pixels = [2500, 10000, 40000, 160000, 320000, 640000]
ind = 1

for m in dog_models:
    print("\n\nMODEL {:d} / 8".format(ind))
    benchmark_model_by_size(m, dog_video, output=out_dir, n_frames=n_frames, pixels=pixels)
    ind += 1

for m in mouse_models:
    print("\n\nMODEL {:d} / 8".format(ind))
    benchmark_model_by_size(m, mouse_video, output=out_dir, n_frames=n_frames, pixels=pixels)
    ind += 1
