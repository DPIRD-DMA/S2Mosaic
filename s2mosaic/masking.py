from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple

import cv2
import numpy as np
import pystac
from multiclean import clean_array
from omnicloudmask import predict_from_array

from .data_reader import get_full_band


def get_valid_mask(bands: np.ndarray, dilation_count: int = 4) -> np.ndarray:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    # erode mask to remove edge pixels
    if dilation_count > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        no_data = cv2.dilate(no_data, kernel, iterations=dilation_count)
    return no_data == 0


def compute_masks_from_array(
    rgb_nir: np.ndarray,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run cloud + valid masking on an in-memory (3, H, W) R+G+NIR uint16 array.

    Returns (clear_mask, valid_mask) at the same resolution as the input."""
    cloud_class = predict_from_array(
        input_array=rgb_nir,
        batch_size=batch_size,
        inference_dtype=inference_dtype,
    )[0]
    clear = (cloud_class == 0).astype(np.uint8)
    clear = clean_array(
        clear, min_island_size=8, smooth_edge_size=3, connectivity=4
    ).astype(bool)
    valid = get_valid_mask(rgb_nir)
    return clear, valid


def get_masks(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 4,
    target_size: int = 10980,
    ocm_resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    # download RG+NIR bands at OCM resolution for cloud masking
    required_bands = ["B04", "B03", "B8A"]
    get_band_at_ocm_res = partial(
        get_full_band, res=ocm_resolution, debug_cache=debug_cache
    )

    hrefs = [item.assets[band].href for band in required_bands]

    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_at_ocm_res, hrefs))

    bands, _ = zip(*bands_and_profiles, strict=False)
    ocm_input = np.vstack(bands)

    clear, valid = compute_masks_from_array(
        ocm_input, batch_size=batch_size, inference_dtype=inference_dtype
    )
    # Resample masks from OCM resolution (20m) to the target output size
    if clear.shape != (target_size, target_size):
        clear = cv2.resize(
            clear.astype(np.uint8),
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid = cv2.resize(
            valid.astype(np.uint8),
            (target_size, target_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return clear, valid
