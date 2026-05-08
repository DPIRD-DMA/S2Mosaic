import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple

import cv2
import numpy as np
import pystac
from multiclean import clean_array
from omnicloudmask import predict_from_array

from .data_reader import get_full_band

# Sentinel-2 SCL band class values:
#   0  no_data       6  water
#   1  saturated     7  unclassified
#   2  dark/shadow   8  cloud_medium_probability
#   3  cloud_shadow  9  cloud_high_probability
#   4  vegetation   10  thin_cirrus
#   5  bare_soil    11  snow
# Treated as cloudy (excluded from "clear"): saturated, cloud shadow, both cloud
# probabilities, thin cirrus.
SCL_CLOUDY_CLASSES: Tuple[int, ...] = (1, 3, 8, 9, 10)
SCL_NO_DATA: int = 0


def _dilate_no_data(no_data: np.ndarray, dilation_count: int) -> np.ndarray:
    """Dilate a no-data mask (1=no_data) by `dilation_count` cross-3x3 iterations."""
    if dilation_count <= 0:
        return no_data
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.dilate(no_data, kernel, iterations=dilation_count)


def get_valid_mask(bands: np.ndarray, dilation_count: int = 4) -> np.ndarray:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    no_data = _dilate_no_data(no_data, dilation_count)
    return no_data == 0


def compute_masks_from_scl(
    scl: np.ndarray, dilation_count: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (clear, valid) masks from an SCL band.

    Mirrors :func:`compute_masks_from_array` so OCM and SCL providers are
    interchangeable. ``clear`` is True where the pixel's SCL class isn't in
    :data:`SCL_CLOUDY_CLASSES`; ``valid`` is True where SCL != 0, dilated to
    erode scene-edge no-data the same way the OCM path does.
    """
    if scl.ndim == 3 and scl.shape[0] == 1:
        scl = scl[0]
    no_data = (scl == SCL_NO_DATA).astype(np.uint8)
    no_data = _dilate_no_data(no_data, dilation_count)
    valid = no_data == 0
    clear = ~np.isin(scl, SCL_CLOUDY_CLASSES)
    return clear, valid


def compute_masks_from_array(
    rgb_nir: np.ndarray,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run cloud + valid masking on an in-memory (3, H, W) R+G+NIR uint16 array.

    Returns (clear_mask, valid_mask) at the same resolution as the input."""
    # Silence OCM's chatty patch-size adjustment notices — they fire on every
    # small-AOI scene and obscure the s2mosaic pipeline logs without telling
    # the user anything actionable.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"omnicloudmask\..*"
        )
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


def get_scl_masks(
    item: pystac.Item,
    user_resolution: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """SCL-based clear+valid masks at the user's output resolution.

    Cheaper than :func:`get_masks` (one COG read, no DL inference) but less
    accurate — relies on the L2A processor's published Scene Classification
    Layer rather than re-running cloud detection.
    """
    href = item.assets["SCL"].href
    arr, _ = get_full_band(href=href, res=user_resolution)
    return compute_masks_from_scl(arr)


def get_masks(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
    max_dl_workers: int = 4,
    target_size: int = 10980,
    ocm_resolution: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    # download RG+NIR bands at OCM resolution for cloud masking
    required_bands = ["B04", "B03", "B8A"]
    get_band_at_ocm_res = partial(get_full_band, res=ocm_resolution)

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
