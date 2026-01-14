"""
Tests for Processor base class
"""
import pytest
import numpy as np
from dlclive.processor import Processor


class TestProcessor:
    """Test Processor base class"""

    def test_processor_initialization(self):
        """Test processor can be initialized"""
        processor = Processor()
        assert processor is not None

    def test_processor_process_default(self):
        """Test default process method returns pose unchanged"""
        processor = Processor()
        pose = np.array([[100, 200, 0.9], [150, 250, 0.8]])
        
        result = processor.process(pose)
        
        np.testing.assert_array_equal(result, pose)

    def test_processor_process_with_kwargs(self):
        """Test process method accepts kwargs"""
        processor = Processor()
        pose = np.array([[100, 200, 0.9]])
        
        result = processor.process(pose, frame_number=1, timestamp=0.5)
        
        np.testing.assert_array_equal(result, pose)

    def test_processor_save_default(self):
        """Test default save method returns 0"""
        processor = Processor()
        result = processor.save()
        assert result == 0

    def test_processor_save_with_file(self):
        """Test save method accepts file parameter"""
        processor = Processor()
        result = processor.save(file="test.txt")
        assert result == 0


