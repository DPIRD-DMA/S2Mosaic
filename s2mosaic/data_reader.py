import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import planetary_computer
import rasterio as rio
import scipy
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def read_in_chunks(
    href: str,
    index: int,
    mask: np.ndarray,
    chunk_multiplier: int = 4,
):
    chunk_size = 512 * chunk_multiplier
    with rio.open(href) as src:
        height, width = src.height, src.width

        mask = scipy.ndimage.zoom(
            mask, (height / mask.shape[0], width / mask.shape[1]), order=0
        )

        all_data = np.zeros((height, width), dtype=np.uint16)
        for row in range(0, height, chunk_size):
            for col in range(0, width, chunk_size):
                chunk_height = min(chunk_size, height - row)
                chunk_width = min(chunk_size, width - col)

                mask_chunk = mask[row : row + chunk_height, col : col + chunk_width]

                if np.any(mask_chunk):
                    window = Window(col, row, chunk_width, chunk_height)  # type: ignore
                    data_chunk = src.read(index, window=window)

                    masked_data = data_chunk * mask_chunk
                    all_data[row : row + chunk_height, col : col + chunk_width] = (
                        masked_data
                    )

        return all_data


def get_band_with_mask(
    href_and_index: tuple[str, int],
    mask: np.ndarray,
    attempt: int = 0,
    debug_cache: bool = False,
    mosaic_method: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Download a S2 band in chunks that intersect with the mask"""
    href = href_and_index[0]
    index = href_and_index[1]
    if debug_cache:
        href_parts = href.split("/")
        cache_path = (
            Path("cache")
            / f"{href_parts[-4]}_{href_parts[-1]}_{index}_{mosaic_method}_10_masked.pkl"
        )
        cache_path.parent.mkdir(exist_ok=True)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            return result
    try:
        singed_href = planetary_computer.sign(href)
        with rio.open(singed_href) as src:
            array = read_in_chunks(
                href=singed_href, index=index, mask=mask, chunk_multiplier=4
            )
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
            return get_band_with_mask(
                href_and_index=href_and_index,
                mask=mask,
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

        if "TCI_10m" in href:
            band_indexes = [1, 2, 3]
        else:
            band_indexes = [1]
        with rio.open(singed_href) as src:
            array = src.read(
                band_indexes,
                out_shape=(
                    len(band_indexes),
                    int(10980 / spatial_ratio),
                    int(10980 / spatial_ratio),
                ),
            ).astype(np.uint16)
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
