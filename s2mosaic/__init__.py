import logging as _logging
from typing import Union

try:
    from ._version import __version__
except ImportError:
    # Source checkout without a build step (e.g. running tests directly from
    # the repo). setuptools-scm writes _version.py at build time.
    __version__ = "0.0.0+unknown"

from .coordinator import mosaic
from .geometry import Aoi, Bbox
from .helpers import SceneFetchError
from .sources import SOURCE_AWS, SOURCE_MPC


def set_log_level(level: Union[int, str] = _logging.INFO) -> None:
    """Enable s2mosaic logging output at the given level.

    By default the library follows standard Python logging convention and emits
    no output unless the host application has configured a handler. Call this
    once to attach a stderr handler and see the pipeline's progress logs::

        import s2mosaic
        s2mosaic.set_log_level("INFO")

    Pass a string ("DEBUG", "INFO", "WARNING") or a logging level constant.
    Calling again replaces the previous handler.
    """
    pkg_logger = _logging.getLogger(__name__)
    for h in list(pkg_logger.handlers):
        pkg_logger.removeHandler(h)
    handler = _logging.StreamHandler()
    handler.setFormatter(
        _logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    pkg_logger.addHandler(handler)
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False


__all__ = [
    "mosaic",
    "Aoi",
    "Bbox",
    "set_log_level",
    "SceneFetchError",
    "SOURCE_AWS",
    "SOURCE_MPC",
    "__version__",
]
