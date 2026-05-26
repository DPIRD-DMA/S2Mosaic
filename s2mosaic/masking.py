import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
import pystac
from multiclean import clean_array
from omnicloudmask import predict_from_array

from .data_reader import get_full_band
from .sources import Source

# Sentinel-2 SCL band class values:
#   0  no_data       6  water
#   1  saturated     7  unclassified
#   2  dark/shadow   8  cloud_medium_probability
#   3  cloud_shadow  9  cloud_high_probability
#   4  vegetation   10  thin_cirrus
#   5  bare_soil    11  snow
# Treated as unsafe for "clear": saturated, dark/shadow, cloud shadow,
# unclassified, both cloud probabilities, thin cirrus.
SCL_CLOUDY_CLASSES: Tuple[int, ...] = (1, 2, 3, 7, 8, 9, 10)
SCL_NO_DATA: int = 0


def _dilate_no_data(no_data: npt.NDArray[Any], dilation_count: int) -> npt.NDArray[Any]:
    """Dilate a no-data mask (1=no_data) by `dilation_count` cross-3x3 iterations."""
    if dilation_count <= 0:
        return no_data
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.dilate(no_data, kernel, iterations=dilation_count)


def get_valid_mask(
    bands: npt.NDArray[Any], dilation_count: int = 4
) -> npt.NDArray[Any]:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    no_data = _dilate_no_data(no_data, dilation_count)
    return no_data == 0  # type: ignore[no-any-return, unused-ignore]


def compute_masks_from_scl(
    scl: npt.NDArray[Any], dilation_count: int = 4
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Build (clear, valid) masks from an SCL band.

    Mirrors :func:`compute_masks_from_array` so OCM and SCL providers are
    interchangeable. ``clear`` is True where the pixel's SCL class is safe for
    compositing; ``valid`` is True where SCL != 0, dilated to erode scene-edge
    no-data the same way the OCM path does.
    """
    if scl.ndim == 3 and scl.shape[0] == 1:
        scl = scl[0]
    no_data = (scl == SCL_NO_DATA).astype(np.uint8)
    no_data = _dilate_no_data(no_data, dilation_count)
    valid = no_data == 0
    clear = ~np.isin(scl, SCL_CLOUDY_CLASSES)
    return clear, valid


def compute_masks_from_array(
    rgb_nir: npt.NDArray[Any],
    batch_size: int = 6,
    inference_dtype: str = "fp16",
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Run cloud + valid masking on an in-memory (3, H, W) R+G+NIR uint16 array.

    Returns (clear_mask, valid_mask) at the same resolution as the input.

    Suppresses omnicloudmask's "Significant no-data areas detected" warning —
    it fires on every cross-UTM-zone edge scene where the swath polygon is
    tilted relative to the read rectangle (triangular nodata wedge). OCM
    auto-shrinks the patch size and produces correct masks; the warning is
    just noise.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Significant no-data areas detected",
            category=UserWarning,
        )
        patch_size = min(*rgb_nir.shape[1:], 1000)
        patch_overlap = min(patch_size // 2, 50)
        cloud_class = predict_from_array(
            input_array=rgb_nir,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )[0]
    clear = (cloud_class == 0).astype(np.uint8)
    clear = clean_array(
        clear, min_island_size=8, smooth_edge_size=3, connectivity=4
    ).astype(bool)
    valid = get_valid_mask(rgb_nir)
    return clear, valid


def get_scl_masks(
    item: pystac.Item,
    source: Source,
    user_resolution: int = 10,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """SCL-based clear+valid masks at the user's output resolution.

    Cheaper than :func:`get_masks` (one COG read, no DL inference) but less
    accurate — relies on the L2A processor's published Scene Classification
    Layer rather than re-running cloud detection.
    """
    href = item.assets[source.asset_name("SCL")].href
    arr, _ = get_full_band(
        href=href, source=source, res=user_resolution, asset_name="SCL"
    )
    return compute_masks_from_scl(arr)


def get_masks(
    item: pystac.Item,
    source: Source,
    batch_size: int = 6,
    inference_dtype: str = "fp16",
    max_dl_workers: int = 4,
    target_size: Union[int, Tuple[int, int]] = 10980,
    ocm_resolution: int = 20,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # download RG+NIR bands at OCM resolution for cloud masking
    ocm_bands = ["B04", "B03", "B8A"]

    def get_band_at_ocm_res(
        band: str,
    ) -> Tuple[npt.NDArray[np.uint16], dict[str, Any]]:
        return get_full_band(
            href=item.assets[source.asset_name(band)].href,
            source=source,
            res=ocm_resolution,
            asset_name=band,
        )

    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_at_ocm_res, ocm_bands))

    band_arrays, _ = zip(*bands_and_profiles, strict=False)
    ocm_input = np.vstack(band_arrays)

    clear, valid = compute_masks_from_array(
        ocm_input, batch_size=batch_size, inference_dtype=inference_dtype
    )
    # Resample masks from OCM resolution (20m) to the target output shape.
    target_height, target_width = (
        (target_size, target_size) if isinstance(target_size, int) else target_size
    )
    if clear.shape != (target_height, target_width):
        clear = cv2.resize(
            clear.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        valid = cv2.resize(
            valid.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return clear, valid
