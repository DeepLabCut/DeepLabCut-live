"""
Tests for utility functions - image processing and file operations
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dlclive import utils


class TestImageUtils:
    """Test image processing utility functions"""

    def test_convert_to_ubyte_uint8(self):
        """Test converting uint8 array (should return unchanged)"""
        frame = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        result = utils.convert_to_ubyte(frame)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, frame)

    def test_convert_to_ubyte_float(self):
        """Test converting float array to uint8"""
        frame = np.array([[0.5, 0.8], [0.2, 0.9]], dtype=np.float32)
        result = utils.convert_to_ubyte(frame)
        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_resize_frame_no_resize(self):
        """Test resize_frame with resize=None or 1 (no resize)"""
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = utils.resize_frame(frame, resize=None)
        np.testing.assert_array_equal(result, frame)

        result = utils.resize_frame(frame, resize=1)
        np.testing.assert_array_equal(result, frame)

    def test_resize_frame_downscale(self):
        """Test resizing frame down"""
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = utils.resize_frame(frame, resize=0.5)
        assert result.shape[0] == 50
        assert result.shape[1] == 100

    def test_resize_frame_upscale(self):
        """Test resizing frame up"""
        frame = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        result = utils.resize_frame(frame, resize=2.0)
        assert result.shape[0] == 100
        assert result.shape[1] == 200

    def test_img_to_rgb_grayscale(self):
        """Test converting grayscale to RGB"""
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = utils.img_to_rgb(frame)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_img_to_rgb_bgr(self):
        """Test converting BGR to RGB"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = utils.img_to_rgb(frame)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_img_to_rgb_invalid_dimensions(self):
        """Test img_to_rgb with invalid dimensions"""
        frame = np.random.randint(0, 255, (100, 100, 3, 2), dtype=np.uint8)
        with pytest.warns(utils.DLCLiveWarning):
            result = utils.img_to_rgb(frame)
            np.testing.assert_array_equal(result, frame)

    def test_gray_to_rgb(self):
        """Test gray_to_rgb conversion"""
        frame = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = utils.gray_to_rgb(frame)
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8

    def test_bgr_to_rgb(self):
        """Test bgr_to_rgb conversion"""
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = utils.bgr_to_rgb(frame)
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8

    def test_decode_fourcc(self):
        """Test decoding fourcc code"""
        # Test valid fourcc
        fourcc = 1145656920  # 'XVID' in fourcc format
        result = utils.decode_fourcc(fourcc)
        assert isinstance(result, str)
        assert len(result) == 4

    def test_decode_fourcc_invalid(self):
        """Test decoding invalid fourcc"""
        result = utils.decode_fourcc("invalid")
        assert result == ""


class TestDownloadUtils:
    """Test file download utilities"""

    @patch("urllib.request.urlopen")
    def test_download_file_success(self, mock_urlopen, tmp_path):
        """Test successful file download"""
        filepath = tmp_path / "downloaded_file.txt"

        # Mock URL response
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "12"  # Total size of 'chunk1chunk2'
        mock_response.read.side_effect = [b"chunk1", b"chunk2", b""]
        mock_urlopen.return_value.__enter__.return_value = mock_response

        utils.download_file("http://example.com/file.txt", str(filepath))

        mock_urlopen.assert_called_once()
        # Verify file was actually written
        assert filepath.exists()
        assert filepath.read_bytes() == b"chunk1chunk2"

    def test_download_file_already_exists(self, tmp_path):
        """Test that existing files are skipped"""
        filepath = tmp_path / "existing_file.txt"
        filepath.write_text("existing content")

        with patch("urllib.request.urlopen") as mock_urlopen:
            utils.download_file("http://example.com/file.txt", str(filepath))
            mock_urlopen.assert_not_called()
