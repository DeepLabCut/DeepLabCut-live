import warnings
from pathlib import Path
from collections import OrderedDict

import torch

from dlclive.modelzoo.utils import load_super_animal_config, download_super_animal_snapshot


def export_modelzoo_model(
    export_path: str | Path,
    super_animal: str,
    model_name: str,
    detector_name: str | None = None,
) -> None:
    """
    Export a DeepLabCut Model Zoo model to a single .pt file.

    Downloads the model configuration and weights from HuggingFace, bundles them
    together (optionally with a detector), and saves as a single torch archive.
    Skips export if the output file already exists.

    Args:
        export_path: Arbitrary destination path for the exported .pt file.
        super_animal: Super animal dataset name (e.g. "superanimal_quadruped").
        model_name: Pose model architecture name (e.g. "resnet_50").
        detector_name: Optional detector model name. If provided, detector
            weights are included in the export.
    """
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(export_path).exists():
        warnings.warn(f"Export path {export_path} already exists, skipping export", UserWarning)
        return

    model_cfg = load_super_animal_config(
        super_animal=super_animal,
        model_name=model_name,
        detector_name=detector_name,
    )

    def _load_model_weights(model_name: str, super_animal: str = super_animal) -> OrderedDict:
        """Download the model weights from huggingface and load them in torch state dict"""
        checkpoint: Path = download_super_animal_snapshot(dataset=super_animal, model_name=model_name)
        return torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    
    export_dict = {
        "config": model_cfg,
        "pose": _load_model_weights(model_name),
        "detector": _load_model_weights(detector_name) if detector_name is not None else None,
    }
    torch.save(export_dict, export_path)


if __name__ == "__main__":
    """Example usage"""	
    from utils import _MODELZOO_PATH
    
    model_name = "resnet_50"
    super_animal = "superanimal_quadruped"

    export_modelzoo_model(
        export_path=_MODELZOO_PATH / 'exported_models' / f'exported_{super_animal}_{model_name}.pt',
        super_animal=super_animal,
        model_name=model_name,
    )
