"""Shared internal types for cross-module implementation boundaries."""

from typing import Any, NamedTuple, Protocol, Tuple

import numpy.typing as npt

SceneWindow = Tuple[int, int, int, int]  # (col_off, row_off, width, height)


class BoundsItemLike(Protocol):
    """Minimal STAC item interface used by bounds/AOI internals."""

    id: str
    assets: Any
    bbox: Any
    geometry: Any


class MaskFetch(NamedTuple):
    """Per-scene mask-fetch result, sized to the scene's footprint window.

    ``arr`` holds the read pixels at ``crop``'s pre-crop shape; the caller
    applies ``crop`` after mask compute to undo any OCM-context padding,
    yielding a block that lands at ``target_window`` in the bounds grid.
    """

    arr: npt.NDArray[Any]
    target_window: SceneWindow
    crop: Tuple[slice, slice]
