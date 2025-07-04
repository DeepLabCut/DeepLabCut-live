import os
import glob
import pathlib
import pytest
from dlclive import benchmark_videos, download_benchmarking_data

@pytest.mark.functional
def test_benchmark_script_runs(tmp_path):
    datafolder = tmp_path / "Data-DLC-live-benchmark"
    download_benchmarking_data(str(datafolder))

    dog_models = glob.glob(str(datafolder / "dog" / "*[!avi]"))
    dog_video = glob.glob(str(datafolder / "dog" / "*.avi"))[0]
    mouse_models = glob.glob(str(datafolder / "mouse_lick" / "*[!avi]"))
    mouse_video = glob.glob(str(datafolder / "mouse_lick" / "*.avi"))[0]

    out_dir = tmp_path / "results"
    out_dir.mkdir(exist_ok=True)

    pixels = [100, 400] #[2500, 10000]
    n_frames = 5

    for m in dog_models:
        print(f"Running dog model: {m}")
        result = benchmark_videos(m, dog_video, output=str(out_dir), n_frames=n_frames, pixels=pixels)
        print("Dog model result:", result)

    for m in mouse_models:
        print(f"Running mouse model: {m}")
        result = benchmark_videos(m, mouse_video, output=str(out_dir), n_frames=n_frames, pixels=pixels)
        print("Mouse model result:", result)

    assert any(out_dir.iterdir())
