"""
Tests for the Engine class - engine detection and model type handling
"""

import pytest
from dlclive.engine import Engine


class TestEngine:
    """Test Engine enum and detection methods"""

    def test_engine_from_model_type_pytorch(self):
        """Test detecting PyTorch engine from model type"""
        assert Engine.from_model_type("pytorch") == Engine.PYTORCH
        assert Engine.from_model_type("PyTorch") == Engine.PYTORCH
        assert Engine.from_model_type("PYTORCH") == Engine.PYTORCH

    def test_engine_from_model_type_tensorflow(self):
        """Test detecting TensorFlow engine from model type"""
        assert Engine.from_model_type("tensorflow") == Engine.TENSORFLOW
        assert Engine.from_model_type("base") == Engine.TENSORFLOW
        assert Engine.from_model_type("tensorrt") == Engine.TENSORFLOW
        assert Engine.from_model_type("lite") == Engine.TENSORFLOW

    def test_engine_from_model_type_invalid(self):
        """Test that invalid model types raise ValueError"""
        with pytest.raises(ValueError, match="Unknown model type"):
            Engine.from_model_type("invalid")

    def test_engine_from_model_path_tensorflow_dir(self, tmp_path):
        """Test detecting TensorFlow engine from directory with .pb and pose_cfg.yaml"""
        model_dir = tmp_path / "tensorflow_model"
        model_dir.mkdir()
        (model_dir / "pose_cfg.yaml").write_text("test")
        (model_dir / "snapshot-100.pb").write_text("test")

        assert Engine.from_model_path(model_dir) == Engine.TENSORFLOW

    def test_engine_from_model_path_pytorch_file(self, tmp_path):
        """Test detecting PyTorch engine from .pt file"""
        model_file = tmp_path / "model.pt"
        model_file.write_text("test")

        assert Engine.from_model_path(model_file) == Engine.PYTORCH

    def test_engine_from_model_path_nonexistent(self, tmp_path):
        """Test that nonexistent paths raise FileNotFoundError"""
        nonexistent = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            Engine.from_model_path(nonexistent)

    def test_engine_from_model_path_invalid(self, tmp_path):
        """Test that invalid model paths raise ValueError"""
        # Directory without required files
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        with pytest.raises(ValueError, match="Could not determine engine"):
            Engine.from_model_path(invalid_dir)

        # File with wrong extension
        wrong_ext = tmp_path / "model.txt"
        wrong_ext.write_text("test")
        with pytest.raises(ValueError, match="Could not determine engine"):
            Engine.from_model_path(wrong_ext)
