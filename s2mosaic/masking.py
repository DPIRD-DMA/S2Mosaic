from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple

import numpy as np
import pystac
import scipy
from omnicloudmask import predict_from_array

from .data_reader import get_full_band


def get_valid_mask(bands: np.ndarray, dilation_count: int = 4) -> np.ndarray:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    # erode mask to remove edge pixels
    if dilation_count > 0:
        no_data = scipy.ndimage.binary_dilation(no_data, iterations=dilation_count)
    return ~no_data


def get_masks(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    # download RG+NIR bands at 20m resolution for cloud masking
    required_bands = ["B04", "B03", "B8A"]
    get_band_20m = partial(get_full_band, res=20, debug_cache=debug_cache)

    hrefs = [item.assets[band].href for band in required_bands]

    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_20m, hrefs))

    # Separate bands and profiles
    bands, _ = zip(*bands_and_profiles, strict=False)
    ocm_input = np.vstack(bands)

    mask = (
        predict_from_array(
            input_array=ocm_input,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
        )[0]
        == 0
    )
    # interpolate mask back to 10m
    mask = mask.repeat(2, axis=0).repeat(2, axis=1)
    valid_mask = get_valid_mask(ocm_input)
    valid_mask = valid_mask.repeat(2, axis=0).repeat(2, axis=1)
    return mask, valid_mask
