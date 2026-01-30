import glob

import pytest

from dlclive import benchmark_videos, download_benchmarking_data
from dlclive.engine import Engine


@pytest.fixture
def datafolder(tmp_path):
    datafolder = tmp_path / "Data-DLC-live-benchmark"
    download_benchmarking_data(str(datafolder))
    return datafolder


@pytest.mark.functional
@pytest.mark.slow
def test_benchmark_script_runs_tf_backend(tmp_path, datafolder):
    dog_models = glob.glob(str(datafolder / "dog" / "*[!avi]"))
    dog_video = glob.glob(str(datafolder / "dog" / "*.avi"))[0]
    mouse_models = glob.glob(str(datafolder / "mouse_lick" / "*[!avi]"))
    mouse_video = glob.glob(str(datafolder / "mouse_lick" / "*.avi"))[0]

    out_dir = tmp_path / "results"
    out_dir.mkdir(exist_ok=True)

    pixels = [100, 400]  # [2500, 10000]
    n_frames = 5

    for model_path in dog_models:
        print(f"Running dog model: {model_path}")
        benchmark_videos(
            model_path=model_path,
            model_type=("base" if Engine.from_model_path(model_path) == Engine.TENSORFLOW else "pytorch"),
            video_path=dog_video,
            output=str(out_dir),
            n_frames=n_frames,
            pixels=pixels,
        )

    for model_path in mouse_models:
        print(f"Running mouse model: {model_path}")
        benchmark_videos(
            model_path=model_path,
            model_type=("base" if Engine.from_model_path(model_path) == Engine.TENSORFLOW else "pytorch"),
            video_path=mouse_video,
            output=str(out_dir),
            n_frames=n_frames,
            pixels=pixels,
        )

    assert any(out_dir.iterdir())


@pytest.mark.parametrize("model_name", ["hrnet_w32", "resnet_50"])
@pytest.mark.functional
@pytest.mark.slow
def test_benchmark_script_with_torch_modelzoo(tmp_path, datafolder, model_name):
    from dlclive import modelzoo

    # Test configuration
    pixels = 4096  # approximately 64x64 pixels, keeping aspect ratio
    n_frames = 5
    out_dir = tmp_path / "results"
    out_dir.mkdir(exist_ok=True)

    # Export models
    model_configs = [
        {
            "checkpoint": tmp_path / f"exported_quadruped_{model_name}.pt",
            "super_animal": "superanimal_quadruped",
            "video_dir": "dog",
        },
        {
            "checkpoint": tmp_path / f"exported_topviewmouse_{model_name}.pt",
            "super_animal": "superanimal_topviewmouse",
            "video_dir": "mouse_lick",
        },
    ]

    for config in model_configs:
        modelzoo.export_modelzoo_model(
            export_path=config["checkpoint"],
            super_animal=config["super_animal"],
            model_name=model_name,
        )
        assert config["checkpoint"].exists(), f"Failed to export {config['super_animal']} model"
        assert config["checkpoint"].stat().st_size > 0, f"Exported {config['super_animal']} model is empty"

    # Get video paths and run benchmarks
    for config in model_configs:
        video_dir = datafolder / config["video_dir"]
        video_path = list(video_dir.glob("*.avi"))[0]
        print(f"Running {config['checkpoint'].stem}")
        benchmark_videos(
            model_path=config["checkpoint"],
            model_type="pytorch",
            video_path=video_path,
            output=str(out_dir),
            n_frames=n_frames,
            pixels=pixels,
        )

    # Assertions: verify output files were created
    output_files = list(out_dir.iterdir())
    assert len(output_files) > 0, "No output files were created by benchmark_videos"
    assert any(f.suffix == ".pickle" for f in output_files), "No pickle files found in output directory"
