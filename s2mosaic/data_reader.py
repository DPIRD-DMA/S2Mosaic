from typing import Any, Dict, Tuple

import cv2
import numpy as np
import planetary_computer
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window

from .helpers import disk_cache, with_scene_retry


def _read_in_chunks(
    href: str,
    index: int,
    mask: np.ndarray,
    target_size: int,
    resampling: Resampling = Resampling.nearest,
    chunk_multiplier: int = 4,
):
    """Read a band masked by `mask`, resampled to (target_size, target_size).

    Chunks are defined in OUTPUT pixel coordinates and translated to native
    windows so we leverage COG overviews via rasterio's out_shape.
    """
    chunk_size = 512 * chunk_multiplier  # in output pixels
    with rio.open(href) as src:
        native_h, native_w = src.height, src.width

        if mask.shape != (target_size, target_size):
            mask_input = mask.astype(np.uint8) if mask.dtype == bool else mask
            mask = cv2.resize(
                mask_input,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST,
            )

        scale_x = native_w / target_size
        scale_y = native_h / target_size
        all_data = np.zeros((target_size, target_size), dtype=np.uint16)

        for row in range(0, target_size, chunk_size):
            for col in range(0, target_size, chunk_size):
                chunk_h = min(chunk_size, target_size - row)
                chunk_w = min(chunk_size, target_size - col)

                mask_chunk = mask[row : row + chunk_h, col : col + chunk_w]
                if not np.any(mask_chunk):
                    continue

                window = Window(
                    col_off=col * scale_x,  # type: ignore
                    row_off=row * scale_y,  # type: ignore
                    width=chunk_w * scale_x,  # type: ignore
                    height=chunk_h * scale_y,  # type: ignore
                )
                data_chunk = src.read(
                    index,
                    window=window,
                    out_shape=(chunk_h, chunk_w),
                    resampling=resampling,
                )
                all_data[row : row + chunk_h, col : col + chunk_w] = (
                    data_chunk * mask_chunk
                )

        return all_data


def _band_with_mask_key(
    href_and_index: tuple[str, int],
    mask: np.ndarray,
    target_size: int,
    resampling: Resampling = Resampling.nearest,
    mosaic_method: str = "",
) -> str:
    href, index = href_and_index
    href_parts = href.split("/")
    return (
        f"{href_parts[-4]}|{href_parts[-1]}|{index}|{mosaic_method}"
        f"|{target_size}|{resampling.name}|masked"
    )


@disk_cache("band", key_fn=_band_with_mask_key)
@with_scene_retry()
def get_band_with_mask(
    href_and_index: tuple[str, int],
    mask: np.ndarray,
    target_size: int,
    resampling: Resampling = Resampling.nearest,
    mosaic_method: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Download a S2 band at `target_size` resolution, in chunks intersecting `mask`."""
    href = href_and_index[0]
    index = href_and_index[1]

    singed_href = planetary_computer.sign(href)
    with rio.open(singed_href) as src:
        array = _read_in_chunks(
            href=singed_href,
            index=index,
            mask=mask,
            target_size=target_size,
            resampling=resampling,
        )
        profile = src.profile.copy()
        scale_x = src.width / target_size
        scale_y = src.height / target_size
        profile["transform"] = src.transform * rio.Affine.scale(scale_x, scale_y)
        profile["width"] = target_size
        profile["height"] = target_size
        return array, profile


def _full_band_key(href: str, res: int = 10) -> str:
    href_parts = href.split("/")
    return f"{href_parts[-4]}|{href_parts[-1]}|{res / 10}|{res}"


@disk_cache("full_band", key_fn=_full_band_key)
@with_scene_retry()
def get_full_band(href: str, res: int = 10) -> Tuple[np.ndarray, Dict[str, Any]]:
    spatial_ratio = res / 10

    singed_href = planetary_computer.sign(href)
    is_tci = "TCI_10m" in href
    with rio.open(singed_href) as src:
        target_side = int(10980 / spatial_ratio)
        # Passing an explicit window is required for rasterio to use COG
        # overviews. Single-band reads must use a scalar index rather than
        # a 1-element list — the latter triggers a slow path that reads at
        # native resolution.
        full_window = Window(0, 0, src.width, src.height)  # type: ignore
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
