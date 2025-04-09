import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pkg_resources
import planetary_computer
import pystac
import pystac_client
import rasterio as rio
import scipy
import shapely
from dateutil.relativedelta import relativedelta
from omnicloudmask import predict_from_array
from pandas import DataFrame
from pystac.item_collection import ItemCollection
from rasterio.windows import Window
from tqdm.auto import tqdm


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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Download a S2 band in chunks that intersect with the mask"""
    href = href_and_index[0]
    index = href_and_index[1]
    if debug_cache:
        cache_path = Path("cache") / f"{href.split('/')[-1]}_10_masked.pkl"
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
        print(e)
        print(f"Failed to open {href}")
        if attempt < 3:
            print(f"Trying again {attempt+1}")
            return get_band_with_mask(href_and_index, mask, attempt + 1)
        else:
            raise Exception(f"Failed to open {href}")


def get_full_band(
    href: str, attempt: int = 0, res: int = 10, debug_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        singed_href = planetary_computer.sign(href)
        spatial_ratio = res / 10

        if debug_cache:
            cache_path = Path("cache") / f"{href.split('/')[-1]}_{spatial_ratio}.pkl"
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
        print(e)
        print(f"Failed to open {href}")
        if attempt < 3:
            print(f"Trying again {attempt+1}")
            return get_full_band(href, attempt + 1)
        else:
            raise Exception(f"Failed to open {href}")


def download_bands_pool(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: np.ndarray,
    no_data_threshold: Union[float, None],
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    s2_scene_size = 10980
    possible_pixel_count = coverage_mask.sum()

    logging.info(f"Possible pixel count: {possible_pixel_count}")

    if "visual" in required_bands:
        mosaic = np.zeros((3, s2_scene_size, s2_scene_size)).astype(np.float32)
        band_indexes = [1, 2, 3]
        required_bands = required_bands * 3

    else:
        mosaic = np.zeros((len(required_bands), s2_scene_size, s2_scene_size)).astype(
            np.float32
        )
        band_indexes = [1] * len(required_bands)

    good_pixel_tracker = np.zeros((s2_scene_size, s2_scene_size)).astype(np.uint16)

    pbar = tqdm(
        total=len(sorted_scenes),
        desc=format_progress(0, len(sorted_scenes), 100.0),
        leave=False,
        bar_format="{desc}",
    )

    for index, item in enumerate(sorted_scenes["item"].tolist()):
        clear_pixels, good_pixels = ocm_cloud_mask(
            item=item,
            batch_size=ocm_batch_size,
            inference_dtype=ocm_inference_dtype,
            debug_cache=debug_cache,
            max_dl_workers=max_dl_workers,
        )
        combo_mask = (clear_pixels * good_pixels).astype(bool)

        # if method is first, only download valid, non cloudy pixels that have not been filled, else download all valid non cloudy pixels
        if mosaic_method == "first":
            combo_mask = (good_pixel_tracker == 0) & combo_mask

        good_pixel_tracker += combo_mask

        hrefs_and_indexes = [
            (item.assets[band].href, band_index)
            for band, band_index in zip(required_bands, band_indexes)
        ]

        get_band_with_mask_partial = partial(
            get_band_with_mask, mask=combo_mask, debug_cache=debug_cache
        )

        with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
            bands_and_profiles = list(
                executor.map(get_band_with_mask_partial, hrefs_and_indexes)
            )

        bands = []

        for band, profile in bands_and_profiles:
            bands.append(
                scipy.ndimage.zoom(
                    band,
                    (s2_scene_size / band.shape[0], s2_scene_size / band.shape[1]),
                    order=0,
                )
            )

        mosaic += np.array(bands)

        compleated_of_possible = coverage_mask * (good_pixel_tracker != 0)
        no_data_sum = coverage_mask.sum() - compleated_of_possible.sum()
        logging.info(f"No data sum: {no_data_sum}")

        no_data_pct = (1 - (compleated_of_possible.sum() / possible_pixel_count)) * 100
        logging.info(f"No data pct: {no_data_pct}")

        pbar.set_description(
            format_progress(index + 1, len(sorted_scenes), no_data_pct)
        )

        if mosaic_method == "first":
            if no_data_sum == 0:
                break
        # if no_data_threshold is set, stop if threshold is reached
        if no_data_threshold is not None:
            if no_data_sum < (possible_pixel_count * no_data_threshold):
                break
        pbar.update(1)

    remaining_scenes = pbar.total - pbar.n
    pbar.update(remaining_scenes)
    pbar.refresh()
    pbar.close()

    mosaic = normalise_for_output(
        mosaic_method=mosaic_method,
        mosaic=mosaic,
        good_pixel_tracker=good_pixel_tracker,
        required_bands=required_bands,
    )

    return mosaic, profile


def get_valid_mask(bands: np.ndarray, dilation_count: int = 4) -> np.ndarray:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    # erode mask to remove edge pixels
    if dilation_count > 0:
        no_data = scipy.ndimage.binary_dilation(no_data, iterations=dilation_count)
    return ~no_data


def save_bands_and_mask_as_geotiffs(
    input_bands: List[np.ndarray],
    cloud_mask: np.ndarray,
    profiles: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    scene_id: str = ""
) -> List[Path]:
    """
    Save the input bands and cloud mask as GeoTIFF files.
    
    Args:
        input_bands (List[np.ndarray]): List of band arrays
        cloud_mask (np.ndarray): Cloud mask array
        profiles (List[Dict[str, Any]]): Rasterio profiles for the bands
        output_dir (Union[str, Path]): Directory to save the GeoTIFFs
        scene_id (str): Optional scene identifier for filename
        
    Returns:
        List[Path]: Paths to the saved GeoTIFF files
    """
    import os
    from pathlib import Path
    import rasterio as rio
    
    # Create output directory if it doesn't exist
    output_dir = str(output_dir)  # Convert Path to string if necessary
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    # Band names for the filenames
    band_names = ["B04_red", "B03_green", "B08_nir"]
    
    # Save each band as a GeoTIFF
    for i, (band, profile, band_name) in enumerate(zip(input_bands, profiles, band_names)):
        # Create a filename
        if scene_id:
            filename = f"{scene_id}_{band_name}_20m.tif"
        else:
            filename = f"{band_name}_20m.tif"
        
        output_path = os.path.join(output_dir, filename)
        
        # Prepare the profile for writing
        write_profile = profile.copy()
        write_profile.update(
            dtype=band.dtype,
            count=1,
            compress="lzw"
        )
        
        # Write the band to a GeoTIFF
        with rio.open(output_path, 'w', **write_profile) as dst:
            dst.write(band.squeeze(), 1)
        
        saved_paths.append(Path(output_path))
    
    # Save the cloud mask using the profile from the first band
    if scene_id:
        mask_filename = f"{scene_id}_cloud_mask_20m.tif"
    else:
        mask_filename = "cloud_mask_20m.tif"
    
    mask_path = os.path.join(output_dir, mask_filename)
    
    # Prepare the profile for the mask
    mask_profile = profiles[0].copy()
    mask_profile.update(
        dtype='uint8',
        count=1,
        compress="lzw",
        nodata=255  # Using 255 as nodata value
    )
    
    # Convert boolean mask to uint8 (0=cloudy, 1=clear)
    mask_uint8 = cloud_mask.astype('uint8')
    
    # Write the mask to a GeoTIFF
    with rio.open(mask_path, 'w', **mask_profile) as dst:
        dst.write(mask_uint8, 1)
    
    saved_paths.append(Path(mask_path))
    
    return saved_paths


def ocm_cloud_mask(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 4,
    save_geotiffs: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cloud masks for a Sentinel-2 scene using the OmniCloudMask model.
    Optionally visualizes the input data and resulting mask and saves as GeoTIFFs.
    
    Args:
        item (pystac.Item): STAC item representing a Sentinel-2 scene
        batch_size (int): Batch size for model inference
        inference_dtype (str): Data type for inference ("bf16" or "fp32")
        debug_cache (bool): Whether to cache downloads
        max_dl_workers (int): Maximum number of parallel download threads
        save_geotiffs (bool): Whether to save bands and mask as GeoTIFFs
        output_dir (Optional[Union[str, Path]]): Directory to save GeoTIFFs and plots
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Cloud mask (True = clear, False = cloudy)
            - Valid data mask (True = valid, False = no data)
    """
    # Extract scene ID from item for filenames
    scene_id = item.id if hasattr(item, 'id') else ""
    
    # Use the specified output directory or fall back to default
    output_dir = "C:/Users/Franz/Downloads/Neuer_test"
    
    
    # Download Red, Green and NIR bands at 20m resolution for cloud masking
    required_bands = ["B04", "B03", "B08"]  # Red, Green, NIR
    get_band_20m = partial(get_full_band, res=20, debug_cache=debug_cache)

    # Get URLs for each band
    hrefs = [item.assets[band].href for band in required_bands]

    # Download bands in parallel
    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_20m, hrefs))

    # Separate bands and profiles
    bands, profiles = zip(*bands_and_profiles)
    
    # Stack bands for model input (omnicloudmask expects RGB+NIR stack)
    ocm_input = np.vstack(bands)

    # Run cloud mask prediction
    # Output value 0 = clear pixel, 1 = cloud
    cloud_mask = (
        predict_from_array(
            input_array=ocm_input,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
        )[0]
        == 0  # Convert to boolean mask where True = clear
    )
    
    # Get mask of valid (non-zero) pixels
    valid_mask = get_valid_mask(ocm_input)
    
    # Save bands and mask as GeoTIFFs if requested
    if save_geotiffs:
        saved_paths = save_bands_and_mask_as_geotiffs(
            input_bands=bands,
            cloud_mask=cloud_mask,
            profiles=profiles,
            output_dir=output_dir,
            scene_id=scene_id
        )
        print(f"Saved GeoTIFFs to: {output_dir}")
        for path in saved_paths:
            print(f"  - {path.name}")
    
    # Upsample masks from 20m to 10m resolution for final output
    cloud_mask_10m = cloud_mask.repeat(2, axis=0).repeat(2, axis=1)
    valid_mask_10m = valid_mask.repeat(2, axis=0).repeat(2, axis=1)
    
    return cloud_mask_10m, valid_mask_10m



def format_progress(current, total, no_data_pct):
    return f"Scenes: {current}/{total} | Mosaic currently contains {no_data_pct:.2f}% no data pixels"


def normalise_for_output(
    mosaic_method: str,
    mosaic: np.ndarray,
    good_pixel_tracker: np.ndarray,
    required_bands: List[str],
):
    if mosaic_method == "mean":
        mosaic = np.divide(
            mosaic,
            good_pixel_tracker,
            out=np.zeros_like(mosaic),
            where=good_pixel_tracker != 0,
        )
    if "visual" in required_bands:
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        mosaic = np.clip(mosaic, 0, 65535).astype(np.int16)
    return mosaic


def add_item_info(items: ItemCollection, max_cloud_percentage: float = 20.0) -> DataFrame:
    """
    Split items by orbit and sort by no_data
    
    Args:
        items (ItemCollection): Collection of STAC items
        max_cloud_percentage (float): Maximum cloud percentage to include (default: 20.0)
        
    Returns:
        DataFrame: DataFrame with filtered items and their metadata
    """
    items_list = []
    filtered_count = 0
    total_count = 0
    
    for item in items:
        total_count += 1
        nodata = item.properties["s2:nodata_pixel_percentage"]
        data_pct = 100 - nodata

        # Get cloud percentage
        cloud = item.properties["s2:high_proba_clouds_percentage"]
        shadow = item.properties["s2:cloud_shadow_percentage"]
        cloud_total = cloud + shadow
        
        # Skip scenes with too many clouds
        if cloud_total > max_cloud_percentage:
            filtered_count += 1
            continue
            
        good_data_pct = data_pct * (1 - cloud_total / 100)
        capture_date = item.datetime

        items_list.append(
            {
                "item": item,
                "orbit": item.properties["sat:relative_orbit"],
                "good_data_pct": good_data_pct,
                "datetime": capture_date,
                "cloud_percentage": cloud_total,
            }
        )

    items_df = pd.DataFrame(items_list)
    logging.info(f"Filtered out {filtered_count} of {total_count} scenes due to cloud cover > {max_cloud_percentage}%")
    return items_df


def export_tif(
    array: np.ndarray,
    profile: Dict[str, Any],
    export_path: Path,
    required_bands: List[str],
    nodata_value: Union[int, None] = 0,
) -> None:
    profile.update(
        count=array.shape[0], dtype=array.dtype, nodata=nodata_value, compress="lzw"
    )
    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = required_bands


def search_for_items(
    bounds, grid_id: str, start_date: date, end_date: date
) -> ItemCollection:
    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": shapely.to_geojson(bounds),
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        "query": {"s2:mgrs_tile": {"eq": grid_id}},
    }

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )
    return catalog.search(**query).item_collection()


def sort_items(items: DataFrame, sort_method: str) -> DataFrame:
    # Sort the dataframe by selected method then by orbit
    if sort_method == "valid_data":
        items_sorted = items.sort_values("good_data_pct", ascending=False)
        orbits = items_sorted["orbit"].unique()
        orbit_groups = {
            orbit: items_sorted[items_sorted["orbit"] == orbit] for orbit in orbits
        }

        result = []

        while any(len(group) > 0 for group in orbit_groups.values()):
            for orbit in orbits:
                if len(orbit_groups[orbit]) > 0:
                    result.append(orbit_groups[orbit].iloc[0])
                    orbit_groups[orbit] = orbit_groups[orbit].iloc[1:]

        items_sorted = pd.DataFrame(result).reset_index(drop=True)

    elif sort_method == "oldest":
        items_sorted = items.sort_values("datetime", ascending=True).reset_index(
            drop=True
        )
    elif sort_method == "newest":
        items_sorted = items.sort_values("datetime", ascending=False).reset_index(
            drop=True
        )
    else:
        raise Exception("Invalid sort method, must be valid_data, oldest or newest")

    return items_sorted


def get_extent_from_grid_id(
    grid_id: str, use_cache: bool = True
) -> shapely.geometry.polygon.Polygon:
    S2_grid_file = Path(
        pkg_resources.resource_filename("s2mosaic", "S2_grid/sentinel_2_index.gpkg")
    )
    S2_grid_file_cache = S2_grid_file.with_suffix(".pkl")

    if S2_grid_file_cache.exists() and use_cache:
        with open(S2_grid_file_cache, "rb") as f:
            S2_grid_gdf = pickle.load(f)
    else:
        assert S2_grid_file.exists()
        S2_grid_gdf = gpd.read_file(S2_grid_file)
        with open(S2_grid_file_cache, "wb") as f:
            pickle.dump(S2_grid_gdf, f)

    S2_grid_gdf = S2_grid_gdf[S2_grid_gdf["Name"] == grid_id]
    if len(S2_grid_gdf) != 1:
        raise ValueError(
            f"Grid {grid_id} not found. It should be in the format '50HMH'. "
            "For more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products"
        )
    return S2_grid_gdf.iloc[0].geometry


def define_dates(
    start_year: int,
    start_month: int,
    start_day: int,
    duration_years: int,
    duration_months: int,
    duration_days: int,
) -> Tuple[date, date]:
    start_date = datetime(start_year, start_month, start_day)
    end_date = start_date + relativedelta(
        years=duration_years, months=duration_months, days=duration_days
    )
    return start_date, end_date


SORT_VALID_DATA = "valid_data"
SORT_OLDEST = "oldest"
SORT_NEWEST = "newest"
SORT_CUSTOM = "custom"
MOSAIC_MEAN = "mean"
MOSAIC_FIRST = "first"

VALID_SORT_METHODS = {SORT_VALID_DATA, SORT_OLDEST, SORT_NEWEST, SORT_CUSTOM}
VALID_MOSAIC_METHODS = {MOSAIC_MEAN, MOSAIC_FIRST}


def validate_inputs(
    sort_method: str,
    mosaic_method: str,
    no_data_threshold: Union[float, None],
    required_bands: List[str],
    grid_id: str,
    max_cloud_percentage: float = 20.0,
) -> None:
    if not grid_id.isalnum() or not grid_id.isupper():
        raise ValueError(
            f"Grid {grid_id} is invalid. It should be in the format '50HMH'. "
            "For more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products"
        )
    if sort_method not in VALID_SORT_METHODS:
        raise ValueError(
            f"Invalid sort method: {sort_method}. Must be one of {VALID_SORT_METHODS}"
        )
    if mosaic_method not in VALID_MOSAIC_METHODS:
        raise ValueError(
            f"Invalid mosaic method: {mosaic_method}. Must be one of {VALID_MOSAIC_METHODS}"
        )
    if no_data_threshold is not None:
        if not (0.0 <= no_data_threshold <= 1.0):
            raise ValueError(
                f"No data threshold must be between 0 and 1 or None, got {no_data_threshold}"
            )
    if not (0.0 <= max_cloud_percentage <= 100.0):
        raise ValueError(
            f"Maximum cloud percentage must be between 0 and 100, got {max_cloud_percentage}"
        )
    valid_bands = [
        "AOT",
        "SCL",
        "WVP",
        "visual",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    for band in required_bands:
        if band not in valid_bands:
            raise ValueError(f"Invalid band: {band}, must be one of {valid_bands}")
    if "visual" in required_bands and len(required_bands) > 1:
        raise ValueError("Cannot use visual band with other bands, must be used alone")


def get_output_path(
    output_dir: Union[Path, str],
    grid_id: str,
    start_date: date,
    end_date: date,
    sort_method: str,
    mosaic_method: str,
    required_bands: List[str],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    bands_str = "_".join(required_bands)
    export_path = output_dir / (
        f"{grid_id}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_{sort_method}_{mosaic_method}_{bands_str}.tif"
    )
    return export_path
