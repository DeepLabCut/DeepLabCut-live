from enum import Enum
from pathlib import Path

class Engine(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"

    @classmethod
    def from_model_type(cls, model_type: str) -> "Engine":
        if model_type.lower() == "pytorch":
            return cls.PYTORCH
        elif model_type.lower() in ("tensorflow", "base", "tensorrt", "lite"):
            return cls.TENSORFLOW
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> "Engine":
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if path.is_dir():
            has_cfg = (path / "pose_cfg.yaml").is_file()
            has_pb = any(p.suffix == ".pb" for p in path.glob("*.pb"))
            if has_cfg and has_pb:
                return cls.TENSORFLOW
        elif path.is_file():
            if path.suffix == ".pt":
                return cls.PYTORCH

        raise ValueError(f"Could not determine engine from model path: {model_path}")