from enum import Enum

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
