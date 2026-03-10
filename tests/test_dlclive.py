"""
Tests for DLCLive core functionality - frame processing, cropping, etc.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from dlclive import DLCLive
from dlclive.exceptions import DLCLiveError


class TestDLCLive:
    """Test DLCLive class core functionality"""

    @pytest.fixture
    def mock_runner(self):
        """Create a mock runner for testing"""
        runner = Mock()
        runner.cfg = {"test": "config"}
        runner.precision = "FP32"
        runner.init_inference.return_value = np.zeros((17, 3))
        runner.get_pose.return_value = np.zeros((17, 3))
        runner.close.return_value = None
        runner.read_config.return_value = {"test": "config"}
        return runner

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @patch("dlclive.factory.build_runner")
    def test_dlclive_initialization(self, mock_build_runner, mock_runner, tmp_path):
        """Test DLCLive initialization"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path, model_type="pytorch")

        assert dlc.path == model_path
        assert dlc.model_type == "pytorch"
        assert not dlc.is_initialized
        assert dlc.cropping is None
        assert dlc.dynamic == (False, 0.5, 10)
        assert dlc.processor is None

    @patch("dlclive.factory.build_runner")
    def test_dlclive_cfg_property(self, mock_build_runner, mock_runner, tmp_path):
        """Test accessing cfg property"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path)
        assert dlc.cfg == {"test": "config"}

    @patch("dlclive.factory.build_runner")
    def test_dlclive_precision_property(self, mock_build_runner, mock_runner, tmp_path):
        """Test accessing precision property"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path)
        assert dlc.precision == "FP32"

    @patch("dlclive.factory.build_runner")
    def test_dlclive_read_config(self, mock_build_runner, mock_runner, tmp_path):
        """Test reading configuration"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path)
        config = dlc.read_config()
        assert config == {"test": "config"}

    @patch("dlclive.factory.build_runner")
    def test_dlclive_parameterization(self, mock_build_runner, mock_runner, tmp_path):
        """Test parameterization property"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path, cropping=[10, 100, 20, 200])
        params = dlc.parameterization

        assert "path" in params
        assert "cfg" in params
        assert "model_type" in params
        assert params["cropping"] == [10, 100, 20, 200]

    @patch("dlclive.factory.build_runner")
    @patch("dlclive.utils.img_to_rgb")
    def test_process_frame_cropping(self, mock_img_to_rgb, mock_build_runner, mock_runner, sample_frame, tmp_path):
        """Test frame processing with cropping"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner
        mock_img_to_rgb.side_effect = lambda x: x

        dlc = DLCLive(model_path, cropping=[10, 100, 20, 200])
        result = dlc.process_frame(sample_frame)

        # Check that cropping was applied (result should be smaller)
        assert result.shape[0] == 180  # 200 - 20
        assert result.shape[1] == 90  # 100 - 10

    @patch("dlclive.factory.build_runner")
    @patch("dlclive.utils.resize_frame")
    @patch("dlclive.utils.img_to_rgb")
    def test_process_frame_resize(
        self,
        mock_img_to_rgb,
        mock_resize,
        mock_build_runner,
        mock_runner,
        sample_frame,
        tmp_path,
    ):
        """Test frame processing with resize"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner
        mock_img_to_rgb.side_effect = lambda x: x
        mock_resize.side_effect = lambda x, resize: x  # No actual resize in test

        dlc = DLCLive(model_path, resize=0.5)
        dlc.process_frame(sample_frame)

        mock_resize.assert_called_once()

    @patch("dlclive.factory.build_runner")
    def test_process_frame_dynamic_cropping(self, mock_build_runner, mock_runner, sample_frame, tmp_path):
        """Test dynamic cropping functionality"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        # Create pose with detected body parts
        pose = np.array([[100, 150, 0.8], [120, 160, 0.9], [80, 140, 0.7]])

        dlc = DLCLive(model_path, dynamic=(True, 0.5, 10))
        dlc.pose = pose

        result = dlc.process_frame(sample_frame)
        assert result is not None

        # Check that dynamic cropping was applied
        assert dlc.dynamic_cropping is not None
        assert len(dlc.dynamic_cropping) == 4

    @patch("dlclive.factory.build_runner")
    def test_init_inference_no_frame(self, mock_build_runner, mock_runner, tmp_path):
        """Test that init_inference raises error with no frame"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path)

        with pytest.raises(DLCLiveError, match="No frame provided"):
            dlc.init_inference()

    @patch("dlclive.factory.build_runner")
    def test_get_pose_no_frame(self, mock_build_runner, mock_runner, tmp_path):
        """Test that get_pose raises error with no frame"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path)

        with pytest.raises(DLCLiveError, match="No frame provided"):
            dlc.get_pose()

    @patch("dlclive.factory.build_runner")
    def test_close(self, mock_build_runner, mock_runner, tmp_path):
        """Test closing DLCLive instance"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        dlc = DLCLive(model_path)
        dlc.is_initialized = True
        dlc.close()

        assert not dlc.is_initialized
        mock_runner.close.assert_called_once()

    @patch("dlclive.factory.build_runner")
    def test_post_process_pose_with_processor(self, mock_build_runner, mock_runner, sample_frame, tmp_path):
        """Test pose post-processing with processor"""
        model_path = tmp_path / "model.pt"
        model_path.write_text("test")
        mock_build_runner.return_value = mock_runner

        # Create mock processor
        mock_processor = Mock()
        mock_processor.process.return_value = np.ones((17, 3))

        dlc = DLCLive(model_path, processor=mock_processor)
        dlc.pose = np.zeros((17, 3))

        # Manually call _post_process_pose
        result = dlc._post_process_pose(sample_frame)

        mock_processor.process.assert_called_once()
        np.testing.assert_array_equal(result, np.ones((17, 3)))
