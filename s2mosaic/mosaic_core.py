import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from .data_reader import get_band_with_mask
from .helpers import (
    CLOUD_MASK_OCM,
    CLOUD_MASK_SCL,
    MOSAIC_FIRST,
    MOSAIC_MEAN,
    MOSAIC_PERCENTILE,
    get_rasterio_resampling,
    pick_ocm_resolution,
)
from .masking import get_masks, get_scl_masks
from .mosaic_utils import calculate_percentile_mosaic
from .stac_utils import ITEM_COL

logger = logging.getLogger(__name__)


def download_bands_pool(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: np.ndarray,
    no_data_threshold: Union[float, None],
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    max_dl_workers: int = 4,
    percentile_value: float | None = 50.0,
    s2_scene_size: int = 10980,
    resampling_method: str = "nearest",
    resolution: int = 10,
    cloud_mask: str = CLOUD_MASK_OCM,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rio_resampling = get_rasterio_resampling(resampling_method)
    ocm_resolution = pick_ocm_resolution(resolution)
    logger.info(f"OCM resolution: {ocm_resolution}m")
    possible_pixel_count = coverage_mask.sum()
    logger.info(f"Possible pixel count: {possible_pixel_count}")

    is_visual = "visual" in required_bands
    if is_visual:
        band_count = 3
        # Visual is one TCI asset; read bands 1/2/3 from it.
        hrefs_template = [("visual", 1), ("visual", 2), ("visual", 3)]
    else:
        band_count = len(required_bands)
        hrefs_template = [(band, 1) for band in required_bands]

    mosaic: np.ndarray
    all_scene_data: List[np.ndarray] = []
    if mosaic_method == MOSAIC_PERCENTILE:
        mosaic = np.empty(0, dtype=np.float32)  # filled in after the loop
    else:
        mosaic = np.zeros((band_count, s2_scene_size, s2_scene_size), dtype=np.float32)

    good_pixel_tracker = np.zeros((s2_scene_size, s2_scene_size), dtype=np.uint16)
    last_profile: Dict[str, Any] = {}

    for index, item in enumerate(sorted_scenes[ITEM_COL].tolist()):
        if cloud_mask == CLOUD_MASK_SCL:
            non_cloud_pixels, valid_pixels = get_scl_masks(
                item=item,
                user_resolution=resolution,
            )
        else:
            non_cloud_pixels, valid_pixels = get_masks(
                item=item,
                batch_size=ocm_batch_size,
                inference_dtype=ocm_inference_dtype,
                max_dl_workers=max_dl_workers,
                target_size=s2_scene_size,
                ocm_resolution=ocm_resolution,
            )

        combo_mask = (non_cloud_pixels * valid_pixels).astype(bool)

        # if method is first, only download valid,
        # non cloudy pixels that have not been filled,
        # else download all valid non cloudy pixels
        if mosaic_method == MOSAIC_FIRST:
            combo_mask = (good_pixel_tracker == 0) & combo_mask

        good_pixel_tracker += combo_mask

        hrefs_and_indexes = [
            (item.assets[asset].href, band_index)
            for asset, band_index in hrefs_template
        ]

        get_band_with_mask_partial = partial(
            get_band_with_mask,
            mask=combo_mask,
            target_size=s2_scene_size,
            resampling=rio_resampling,
            mosaic_method=mosaic_method,
        )

        with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
            bands_and_profiles = list(
                executor.map(get_band_with_mask_partial, hrefs_and_indexes)
            )

        bands = []
        for band, profile in bands_and_profiles:
            if band.shape != (s2_scene_size, s2_scene_size):
                band = cv2.resize(
                    band,
                    (s2_scene_size, s2_scene_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            bands.append(band)
            last_profile = profile

        scene_data = np.array(bands)

        if mosaic_method == MOSAIC_PERCENTILE:
            scene_data = np.where(combo_mask, scene_data, np.nan)
            all_scene_data.append(scene_data)
        else:
            mosaic += scene_data

        completed_of_possible = coverage_mask * (good_pixel_tracker != 0)
        no_data_sum = coverage_mask.sum() - completed_of_possible.sum()
        no_data_pct = (1 - (completed_of_possible.sum() / possible_pixel_count)) * 100
        logger.info(
            f"Scene {index + 1}/{len(sorted_scenes)} processed; "
            f"no-data {no_data_pct:.2f}% ({no_data_sum} px)"
        )

        if mosaic_method == MOSAIC_FIRST and no_data_sum == 0:
            break
        if no_data_threshold is not None and no_data_sum < (
            possible_pixel_count * no_data_threshold
        ):
            break

    if mosaic_method == MOSAIC_PERCENTILE:
        if percentile_value is None:
            raise ValueError("Percentile must be provided for percentile mosaic method")
        max_workers = multiprocessing.cpu_count() // 2
        mosaic = calculate_percentile_mosaic(
            all_scene_data=all_scene_data,
            s2_scene_size=s2_scene_size,
            max_workers=max_workers,
            percentile_value=float(percentile_value),
        )
    elif mosaic_method == MOSAIC_MEAN:
        mosaic = np.divide(
            mosaic,
            good_pixel_tracker,
            out=np.zeros_like(mosaic),
            where=good_pixel_tracker != 0,
        )

    if is_visual:
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        mosaic = np.clip(mosaic, 0, 65535).astype(np.uint16)

    return mosaic, last_profile
