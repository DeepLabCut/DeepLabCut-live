"""
Tests for configuration file reading
"""

import pytest
from dlclive.core import config


class TestConfig:
    """Test configuration file reading"""

    def test_read_yaml_success(self, tmp_path):
        """Test successfully reading a YAML config file"""
        config_file = tmp_path / "pose_cfg.yaml"
        # Write YAML content directly
        yaml_content = """num_joints: 17
all_joints:
  - head
  - neck
  - shoulder
batch_size: 1
"""
        config_file.write_text(yaml_content)

        result = config.read_yaml(config_file)

        assert result["num_joints"] == 17
        assert result["all_joints"] == ["head", "neck", "shoulder"]
        assert result["batch_size"] == 1

    def test_read_yaml_nonexistent(self, tmp_path):
        """Test that nonexistent config files raise FileNotFoundError"""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            config.read_yaml(nonexistent)

    def test_read_yaml_path_resolution(self, tmp_path):
        """Test that paths are properly resolved"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")

        # Test with relative path
        result = config.read_yaml(str(config_file))
        assert result["test"] == "value"

        # Test with Path object
        result = config.read_yaml(config_file)
        assert result["test"] == "value"
