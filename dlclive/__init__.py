"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

# Check which backends are installed and get available backends
# (Emits a warning if neither TensorFlow nor PyTorch is installed)
from dlclive.utils import get_available_backends

_AVAILABLE_BACKENDS = get_available_backends()

from dlclive.display import Display
from dlclive.dlclive import DLCLive
from dlclive.engine import Engine
from dlclive.processor.processor import Processor
from dlclive.version import VERSION, __version__


def benchmark_videos(*args, **kwargs):
    """Lazy import of benchmark_videos from dlclive.benchmark"""
    from dlclive.benchmark import benchmark_videos as _benchmark_videos

    return _benchmark_videos(*args, **kwargs)


def download_benchmarking_data(*args, **kwargs):
    """Lazy import of benchmark_videos from dlclive.benchmark"""
    from dlclive.benchmark import download_benchmarking_data as _download_benchmarking_data

    return _download_benchmarking_data(*args, **kwargs)


__all__ = [
    "DLCLive",
    "Display",
    "Processor",
    "Engine",
    "VERSION",
    "__version__",
]
