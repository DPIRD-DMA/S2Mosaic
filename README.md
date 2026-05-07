## S2Mosaic

S2Mosaic is a Python package for creating cloud-free mosaics from Sentinel-2 satellite imagery. It allows users to generate composite images for specified grid areas and time ranges, with various options for scene selection and mosaic creation.

[S2Mosaic blog post here](https://dpird-dma.github.io/blog/S2Mosaic-Creating-Cloud-Free-Sentinel-2-Mosaics/)


## Features

- Mosaic by MGRS grid tile (`grid_id`) **or** an arbitrary lon/lat bounding box (`bounds`) — bounds can cross MGRS tile boundaries and are reprojected onto a common UTM grid in one step.
- Flexible scene selection methods: by valid data percentage, oldest, or newest scenes.
- Multiple mosaic creation methods: mean, arbitrary percentile, median or first valid pixel.
- Support for different spectral bands, including visual (RGB) composites.
- State-of-the-art cloud masking using the OmniCloudMask library.
- Export mosaics as GeoTIFF files or return as NumPy arrays.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Note

We use OmniCloudMask (OCM) for state-of-the-art cloud and cloud shadow masking. OCM will run significantly faster if an available NVIDIA GPU or MPS accelerator is present.

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/drive/1-vdAAnpzp_VCotTV07cbSC9iQFiD7DcH?usp=sharing 'Try S2Mosaic In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab



## Installation

You can install S2Mosaic using pip:
```
pip install s2mosaic
```
Or with uv:
```
uv add s2mosaic
```
## Usage Example 1

Here's a basic example of how to use S2Mosaic:

```python
from s2mosaic import mosaic
from pathlib import Path

# Create a mosaic for a specific grid area and time range
result = mosaic(
    grid_id="50HMH", # Sentinel-2 scene grid ID
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2, # Duration to collect data from
    output_dir=Path("output"), # Output directory for mosaic TIFF files
    sort_method="valid_data", # Method to sort potential scenes before download
    mosaic_method="mean", # Approach used to combine scenes
    required_bands=['visual'], # Required Sentinel-2 bands
    no_data_threshold=0.001 # Threshold for early stopping
)

print(f"Mosaic saved to: {result}")
```

This example creates a mosaic for the grid area "50HMH" for the first two months of 2022, using the visual (TCI) product. The scenes are sorted by valid data percentage, and the mosaic is created using the mean of valid pixels. The process stops iterating through scenes once the no_data_threshold is reached.

## Usage Example 2

Here's another example of how to use S2Mosaic:

```python
from s2mosaic import mosaic

# Create a mosaic for a specific grid area and time range
array, rio_profile = mosaic(
    grid_id="50HMH",
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2,
    sort_method="valid_data",
    mosaic_method="mean",
    required_bands=["B04", "B03", "B02", "B08"],
    no_data_threshold=0.001
)

print(f"Mosaic array shape: {array.shape}")
```

Similar to the example above but with 16-bit red, green, blue, and NIR bands returned as a NumPy array and rasterio profile.

## Usage Example 3 — arbitrary bounding box

Pass `bounds=(minx, miny, maxx, maxy)` instead of `grid_id` to mosaic any rectangular AOI, including ones that cross MGRS tile boundaries. Scenes from the intersecting tiles are pulled and reprojected onto a common UTM grid via stackstac.

```python
from s2mosaic import mosaic

# ~5km AOI near Perth, WA, in EPSG:4326 (default)
bounds = (115.83, -31.97, 115.91, -31.94)

array, profile = mosaic(
    bounds=bounds,
    start_year=2023,
    start_month=6,
    duration_months=2,
    required_bands=["B04", "B03", "B02"],
    mosaic_method="percentile",
    percentile_value=50,
)

print(f"Shape: {array.shape}")
print(f"CRS:   {profile['crs']}")
print(f"Pixel: {profile['transform'].a}m")
```

`bounds_crs` (default `4326`) controls the input CRS; `target_crs` controls the output CRS (auto-picked from the AOI centroid if omitted). Use `resolution` (in metres) and `resampling_method` (`nearest`, `bilinear`, ...) to control the output grid. See [Example use - bounds.ipynb](Example%20use%20-%20bounds.ipynb) for cross-tile and lower-resolution examples.

## Advanced Usage

S2Mosaic provides several options for customizing the mosaic creation process:

- `sort_method`: Choose between `"valid_data"`, `"oldest"`, or `"newest"` to determine scene selection priority.
- `mosaic_method`: Use `"mean"` for an average of valid pixels, `"percentile"` with `percentile_value` for more particular merging, `"median"` (shortcut for the 50th percentile), or `"first"` to use the first valid pixel.
- `required_bands`: Specify which spectral bands to include in the mosaic. Defaults to `["B04", "B03", "B02", "B08"]`. Use `["visual"]` for the 3-band uint8 TCI RGB composite.
- `no_data_threshold`: Set the threshold for considering a pixel as no-data. Set to `None` to process all scenes (default `0.01`).
- `ocm_batch_size`: Set the batch size for OmniCloudMask inference (default: 1).
- `ocm_inference_dtype`: Set the data type for OmniCloudMask inference (default: `"bf16"`).
- `additional_query`: Set additional STAC query filters, e.g. `{"eo:cloud_cover": {"lt": 80}}`.
- `coverage_threshold_pct`: Drop scene-edge pixels covered by only a small minority of overlapping scenes (default `0.1`; set to `None` to keep everything).

Bounds-mode-specific options:

- `bounds`: `(minx, miny, maxx, maxy)` rectangle. Mutually exclusive with `grid_id`.
- `bounds_crs`: EPSG of `bounds` (default `4326`).
- `target_crs`: EPSG of the output. Auto-picked from the AOI centroid (UTM zone) if omitted.
- `resolution`: Output pixel size in metres (default `10`). At lower resolutions stackstac reads from COG overviews — much less data over the wire.
- `resampling_method`: `"nearest"` (default), `"bilinear"`, etc. — how the source is resampled to the output grid.

For more detailed information on these options and additional functionality, please refer to the function docstring in the source code.

## Logging

S2Mosaic emits progress logs at each pipeline stage (search, sort, fetch, cloud-mask, aggregate, export) and shows tqdm progress bars during the slow per-scene loops. Following standard Python logging convention, no output is produced unless logging is configured. The simplest way to enable it:

```python
import s2mosaic

s2mosaic.set_log_level("INFO")  # or "DEBUG" for more detail
```

If your application already configures the `logging` module, the package logger (`s2mosaic`) will respect that — no need to call `set_log_level()`.

## Performance Tips
- `ocm_batch_size`: If using a GPU, setting this above the default value (1) will speed up cloud masking. In most cases, a value of 4 works well. If you encounter CUDA errors, try using a lower number.
- `ocm_inference_dtype`: if the device supports it 'bf16' tends to be the fastest option, failing this try 'fp16' then 'fp32'.
- `sort_method`: Using "valid_data" as the sort method tends to be the fastest option if no_data_threshold is not None.
- `mosaic_method`: Using 'first' can be a lot faster than 'mean' as only valid, non cloudy, new pixels are downloaded.

## Contributing

Contributions to S2Mosaic are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License

This project is licensed under the MIT License.

## Acknowledgments

This package uses the Planetary Computer STAC API and the OmniCloudMask library for cloud masking.