import os
import glob
import pathlib
import pytest
from dlclive import benchmark_videos, download_benchmarking_data
from dlclive.engine import Engine

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

    for model_path in dog_models:
        print(f"Running dog model: {model_path}")
        result = benchmark_videos(
            model_path=model_path,
            model_type="base" if Engine.from_model_path(model_path) == Engine.TENSORFLOW else "pytorch",
            video_path=dog_video,
            output=str(out_dir),
            n_frames=n_frames,
            pixels=pixels
        )
        print("Dog model result:", result)

    for model_path in mouse_models:
        print(f"Running mouse model: {model_path}")
        result = benchmark_videos(
            model_path=model_path,
            model_type="base" if Engine.from_model_path(model_path) == Engine.TENSORFLOW else "pytorch",
            video_path=mouse_video,
            output=str(out_dir),
            n_frames=n_frames,
            pixels=pixels
        )
        print("Mouse model result:", result)

    assert any(out_dir.iterdir())
