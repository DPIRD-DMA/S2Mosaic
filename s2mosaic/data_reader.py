import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import planetary_computer
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def read_in_chunks(
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


def get_band_with_mask(
    href_and_index: tuple[str, int],
    mask: np.ndarray,
    target_size: int,
    resampling: Resampling = Resampling.nearest,
    attempt: int = 0,
    debug_cache: bool = False,
    mosaic_method: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Download a S2 band at `target_size` resolution, in chunks intersecting `mask`."""
    href = href_and_index[0]
    index = href_and_index[1]
    if debug_cache:
        href_parts = href.split("/")
        cache_name = (
            f"{href_parts[-4]}_{href_parts[-1]}_{index}_{mosaic_method}"
            f"_{target_size}_{resampling.name}_masked.pkl"
        )
        cache_path = Path("cache") / cache_name
        cache_path.parent.mkdir(exist_ok=True)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            return result
    try:
        singed_href = planetary_computer.sign(href)
        with rio.open(singed_href) as src:
            array = read_in_chunks(
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
            result = array, profile
            if debug_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)

            return result

    except Exception as e:
        logger.error(f"Failed to open {href}: {e}")
        if attempt < 3:
            logger.info(f"Retrying attempt {attempt + 1}/3")
            if debug_cache:
                logger.info("Debug cache is enabled, skipping cache for retry")
            return get_band_with_mask(
                href_and_index=href_and_index,
                mask=mask,
                target_size=target_size,
                resampling=resampling,
                attempt=attempt + 1,
                debug_cache=False,
                mosaic_method=mosaic_method,
            )
        else:
            logger.error(f"All retry attempts failed for {href}")
            raise Exception(
                f"Failed to open {href} after {attempt + 1} attempts"
            ) from None


def get_full_band(
    href: str, attempt: int = 0, res: int = 10, debug_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        singed_href = planetary_computer.sign(href)
        spatial_ratio = res / 10

        if debug_cache:
            href_parts = href.split("/")
            cache_path = (
                Path("cache")
                / f"{href_parts[-4]}_{href_parts[-1]}_{spatial_ratio}_{res}.pkl"
            )
            cache_path.parent.mkdir(exist_ok=True)
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                return result

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
            result = array, src.profile.copy()
            if debug_cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
            return result

    except Exception as e:
        logger.error(f"Failed to open {href}: {e}")
        if attempt < 3:
            logger.info(f"Retrying attempt {attempt + 1}/3")
            if debug_cache:
                logger.info("Debug cache is enabled, skipping cache for retry")
            return get_full_band(
                href=href, attempt=attempt + 1, res=res, debug_cache=False
            )
        else:
            logger.error(f"All retry attempts failed for {href}")
            raise Exception(
                f"Failed to open {href} after {attempt + 1} attempts"
            ) from None
