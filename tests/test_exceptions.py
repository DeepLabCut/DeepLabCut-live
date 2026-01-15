"""
Tests for exception classes
"""
import pytest
from dlclive.exceptions import DLCLiveError, DLCLiveWarning


class TestExceptions:
    """Test exception and warning classes"""

    def test_dlclive_error(self):
        """Test DLCLiveError can be raised and caught"""
        with pytest.raises(DLCLiveError):
            raise DLCLiveError("Test error message")

    def test_dlclive_error_message(self):
        """Test DLCLiveError preserves error message"""
        error_msg = "Custom error message"
        with pytest.raises(DLCLiveError, match=error_msg):
            raise DLCLiveError(error_msg)

    def test_dlclive_error_inheritance(self):
        """Test DLCLiveError is an Exception"""
        error = DLCLiveError("test")
        assert isinstance(error, Exception)

    def test_dlclive_warning(self):
        """Test DLCLiveWarning can be issued"""
        with pytest.warns(DLCLiveWarning):
            import warnings
            warnings.warn("Test warning", DLCLiveWarning)

    def test_dlclive_warning_inheritance(self):
        """Test DLCLiveWarning is a Warning"""
        assert issubclass(DLCLiveWarning, Warning)


