# NOTE JR 2026-23-01: This is duplicate code, copied from the original DeepLabCut-Live codebase.

import os

import pytest
import dlclibrary
from dlclibrary.dlcmodelzoo.modelzoo_download import MODELOPTIONS

from dlclive import modelzoo


@pytest.mark.parametrize(
    "super_animal", ["superanimal_quadruped", "superanimal_topviewmouse"]
)
@pytest.mark.parametrize("model_name", ["hrnet_w32"])
@pytest.mark.parametrize("detector_name", [None, "fasterrcnn_resnet50_fpn_v2"])
def test_get_config_model_paths(super_animal, model_name, detector_name):
    model_config = modelzoo.load_super_animal_config(
        super_animal=super_animal,
        model_name=model_name,
        detector_name=detector_name,
    )

    assert isinstance(model_config, dict)
    if detector_name is None:
        assert model_config["method"].lower() == "bu"
        assert "detector" not in model_config
    else:
        assert model_config["method"].lower() == "td"
        assert "detector" in model_config


def test_download_huggingface_model(tmp_path_factory, model="full_cat"):
    folder = tmp_path_factory.mktemp("temp")
    dlclibrary.download_huggingface_model(model, str(folder))

    assert os.path.exists(folder / "pose_cfg.yaml")
    assert any(f.startswith("snapshot-") for f in os.listdir(folder))
    # Verify that the Hugging Face folder was removed
    assert not any(f.startswith("models--") for f in os.listdir(folder))


def test_download_huggingface_wrong_model():
    with pytest.raises(ValueError):
        dlclibrary.download_huggingface_model("wrong_model_name")


@pytest.mark.skip(reason="slow")
@pytest.mark.parametrize("model", MODELOPTIONS)
def test_download_all_models(tmp_path_factory, model):
    test_download_huggingface_model(tmp_path_factory, model)