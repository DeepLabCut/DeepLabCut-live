"""
Tests for the Factory module - runner building
"""

import pytest
from unittest.mock import Mock, patch
from dlclive import factory


class TestFactory:
    """Test factory functions for building runners"""

    def test_filter_keys(self):
        """Test filtering kwargs to only valid keys"""
        kwargs = {
            "device": "cpu",
            "precision": "FP32",
            "invalid": "value",
            "single_animal": True,
        }
        valid = {"device", "precision", "single_animal"}

        filtered = factory.filter_keys(valid, kwargs)

        assert "device" in filtered
        assert "precision" in filtered
        assert "single_animal" in filtered
        assert "invalid" not in filtered
        assert len(filtered) == 3

    @patch("dlclive.pose_estimation_pytorch.runner.PyTorchRunner")
    def test_build_runner_pytorch(self, mock_runner_class, tmp_path):
        """Test building PyTorch runner"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")

        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner

        runner = factory.build_runner(
            "pytorch", model_path, device="cpu", precision="FP32"
        )

        mock_runner_class.assert_called_once()
        assert runner == mock_runner

    def test_build_runner_tensorflow(self, tmp_path):
        """Test building TensorFlow runner"""
        model_path = tmp_path / "tf_model"
        model_path.mkdir()

        # Import the module to ensure it exists
        try:
            from dlclive.pose_estimation_tensorflow import runner
        except ImportError:
            pytest.skip("TensorFlow runner module not available")

        # Patch the TensorFlowRunner class in the runner module
        with patch.object(
            runner, "TensorFlowRunner", autospec=True
        ) as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner

            # Need to reload factory to get fresh import
            import importlib
            import sys

            if "dlclive.factory" in sys.modules:
                importlib.reload(sys.modules["dlclive.factory"])
            from dlclive import factory as factory_module

            runner_instance = factory_module.build_runner(
                "base", model_path, precision="FP32"
            )

            mock_runner_class.assert_called_once()
            assert runner_instance == mock_runner

    def test_build_runner_tensorflow_type_alias(self, tmp_path):
        """Test that 'tensorflow' model_type is converted to 'base'"""
        model_path = tmp_path / "tf_model"
        model_path.mkdir()

        # Import the module to ensure it exists
        try:
            from dlclive.pose_estimation_tensorflow import runner
        except ImportError:
            pytest.skip("TensorFlow runner module not available")

        # Patch the TensorFlowRunner class in the runner module
        with patch.object(
            runner, "TensorFlowRunner", autospec=True
        ) as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner

            # Need to reload factory to get fresh import
            import importlib
            import sys

            if "dlclive.factory" in sys.modules:
                importlib.reload(sys.modules["dlclive.factory"])
            from dlclive import factory as factory_module

            runner_instance = factory_module.build_runner("tensorflow", model_path)
            assert runner_instance is not None

            # Check that 'base' was passed to TensorFlowRunner
            call_args = mock_runner_class.call_args
            assert call_args[0][1] == "base"  # Second positional arg should be "base"

    def test_build_runner_invalid_type(self, tmp_path):
        """Test that invalid model types raise ValueError"""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with pytest.raises(ValueError, match="Unknown model type"):
            factory.build_runner("invalid", model_path)
