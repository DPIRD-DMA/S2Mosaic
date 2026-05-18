from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.windows import Window

from .helpers import disk_cache, with_scene_retry
from .sources import Source


def _full_band_key(href: str, source: Source, res: int = 10) -> str:
    href_parts = href.split("/")
    return f"{source.name}|{href_parts[-4]}|{href_parts[-1]}|{res / 10}|{res}"


@disk_cache("full_band", key_fn=_full_band_key)
@with_scene_retry()
def get_full_band(
    href: str, source: Source, res: int = 10
) -> Tuple[npt.NDArray[np.uint16], Dict[str, Any]]:
    spatial_ratio = res / 10

    signed_href = source.sign(href)
    is_tci = "TCI_10m" in href or "/visual/" in href or href.endswith("_TCI.tif")
    with rio.open(signed_href) as src:
        target_side = int(10980 / spatial_ratio)
        # Passing an explicit window is required for rasterio to use COG
        # overviews. Single-band reads must use a scalar index rather than
        # a 1-element list — the latter triggers a slow path that reads at
        # native resolution.
        window_cls: Any = Window
        full_window = window_cls(0, 0, src.width, src.height)
        if is_tci:
            array = src.read(
                [1, 2, 3],
                window=full_window,
                out_shape=(3, target_side, target_side),
            ).astype(np.uint16)
        else:
            array = src.read(
                1,
                window=full_window,
                out_shape=(target_side, target_side),
            ).astype(np.uint16)[None, :, :]
        return array, src.profile.copy()
