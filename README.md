## S2Mosaic

[![PyPI version](https://img.shields.io/pypi/v/s2mosaic.svg)](https://pypi.org/project/s2mosaic/)
[![Python versions](https://img.shields.io/pypi/pyversions/s2mosaic.svg)](https://pypi.org/project/s2mosaic/)
[![CI](https://github.com/DPIRD-DMA/S2Mosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/DPIRD-DMA/S2Mosaic/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/s2mosaic)](https://pepy.tech/project/s2mosaic)
[![License](https://img.shields.io/pypi/l/s2mosaic.svg)](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/LICENSE)

S2Mosaic is a Python package for creating cloud-free mosaics from Sentinel-2 satellite imagery. It allows users to generate composite images for specified grid areas and time ranges, with various options for scene selection and mosaic creation.

## Features

- Mosaic by MGRS grid tile (`grid_id`), rectangular bounds (`bounds`), or a single polygon (`aoi`) — bounds/AOIs can cross MGRS tile boundaries and are reprojected onto a common UTM grid in one step.
- Flexible scene selection methods: by valid data percentage, oldest, or newest scenes.
- Multiple mosaic creation methods: mean, arbitrary percentile, median or first valid pixel.
- Support for different spectral bands, including visual (RGB) composites.
- State-of-the-art cloud masking using the OmniCloudMask library.
- Tile-streaming pipeline keeps peak memory low even for full-MGRS percentile mosaics over many scenes — the aggregation is parallelised across ~2048-pixel tiles, so only a handful of tile-sized buffers live in RAM at a time.
- Resilient to transient COG read failures — per-scene fetches retry with exponential backoff, and a scene that still fails is logged and skipped so one bad asset doesn't abort the whole mosaic.
- Export mosaics as GeoTIFF files or return as NumPy arrays.

## Changelog

See [CHANGELOG.md](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/CHANGELOG.md) for the full release history.

## Note

S2Mosaic uses OmniCloudMask (OCM) for state-of-the-art cloud and cloud shadow masking. OCM will run significantly faster if an available NVIDIA GPU or MPS accelerator is present. When you have limited compute and need to generate large images, pass `cloud_mask="SCL"` to skip the deep-learning model and use the L2A Scene Classification Layer instead — much cheaper, at the cost of some accuracy.

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/github/DPIRD-DMA/S2Mosaic/blob/main/examples/Quick%20start.ipynb 'Try S2Mosaic In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab

The Colab badge opens [`examples/Quick start.ipynb`](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Quick%20start.ipynb) — a minimal end-to-end demo.



## Installation

You can install S2Mosaic using pip:
```
pip install s2mosaic
```
Or with uv:
```
uv add s2mosaic
```
## Quick start — MGRS grid tile

Mosaic an entire Sentinel-2 MGRS tile by ID, returning a NumPy array and rasterio profile. Find the tile ID for your area of interest with the [Sentinel-2 grid explorer](https://dpird-dma.github.io/Sentinel-2-grid-explorer/).

```python
from s2mosaic import mosaic

array, rio_profile = mosaic(
    grid_id="50HMH",            # Sentinel-2 MGRS tile ID
    start_year=2022,
    start_month=1,
    duration_months=2,          # collect scenes over this window
    sort_method="valid_data",   # prioritise scenes with the most valid pixels
    mosaic_method="mean",       # combine valid pixels by mean
    required_bands=["B04", "B03", "B02", "B08"],
)

print(f"Mosaic array shape: {array.shape}")
```

To save a GeoTIFF instead of returning the array, pass `output_dir=Path("output")`
for an auto-generated filename, or `output_path=Path("output/custom.tif")` to
choose the exact filename. The function then returns the file path.

Auto-generated `output_dir` filenames include a readable summary of the target,
date range, bands, method, resolution, cloud-mask provider, source, and a short
deterministic hash of the output-affecting request fields. A matching `.json`
sidecar is written next to the GeoTIFF with the normalized request metadata, so
similar requests do not silently collide while the filename stays scan-friendly.

## Quick start — arbitrary bounding box

Pass `bounds=(minx, miny, maxx, maxy)` instead of `grid_id` to mosaic any rectangular AOI, including ones that cross MGRS tile boundaries. Each intersecting scene is streamed through a rasterio `WarpedVRT` and aggregated onto a common UTM grid.

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
    percentile=50,
)

print(f"Shape: {array.shape}")
print(f"CRS:   {profile['crs']}")
print(f"Pixel: {profile['transform'].a}m")
```

`input_crs` (default `4326`) controls the bounds/AOI CRS; `output_crs` controls the output CRS (auto-picked from the AOI centroid if omitted). Use `resolution` (in metres) and `resampling_method` (`nearest`, `bilinear`, ...) to control the output grid. See [Example use - bounds.ipynb](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Example%20use%20-%20bounds.ipynb) for cross-tile and lower-resolution examples, and [Example use - aoi.ipynb](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Example%20use%20-%20aoi.ipynb) for polygon AOIs.

## Advanced Usage

S2Mosaic provides several options for customizing the mosaic creation process. Defaults shown in parentheses.

**Area (pass exactly one)**

- `grid_id` (`None`): Sentinel-2 MGRS tile ID, e.g. `"50HMH"`. Mosaics the entire tile.
- `bounds` (`None`): `(minx, miny, maxx, maxy)` rectangle. Mosaics an arbitrary AOI, including ones that cross MGRS tile boundaries. See the bounds-mode-specific options below.
- `aoi` (`None`): single shapely `Polygon`. Mosaics the polygon bounds while skipping and masking pixels outside the polygon. Mutually exclusive with `grid_id` and `bounds`.

**Time window**

- `start_year` (required), `start_month` (`1`), `start_day` (`1`): start of the scene-search window.
- `duration_years` (`0`), `duration_months` (`0`), `duration_days` (`0`): length of the search window. Inclusive of `start_*`, exclusive of the end.

**Output content**

- `required_bands` (`["B04", "B03", "B02", "B08"]`): bands to include. Use `["visual"]` for the 3-band uint8 TCI RGB composite (mutually exclusive with other bands).
- `mosaic_method` (`"mean"`): `"mean"`, `"first"`, `"percentile"` (with `percentile`), or `"median"` (shortcut for percentile 50).
- `percentile` (`None`): percentile to compute when `mosaic_method="percentile"` (0–100).
- `observation_target` (`None`): minimum valid observations per pixel for `"mean"` and `"percentile"`. When set, aggregation stops reading later scenes for a tile once every coverable pixel has reached the target. This is not an output quality filter; pixels that cannot reach the target use whatever observations are available.
- `tile_workers` (`min(4, os.cpu_count() or 1)`): number of output tiles to aggregate concurrently. Higher values can improve throughput for network-bound reads, but increase memory use and simultaneous source reads.
- `adaptive_tiling` (`True`): split sparse output tiles based on the actual cloud-valid contribution masks. This reduces wasted reads for irregular AOIs, sparse coverage, and heavily masked scenes. Set to `False` to use fixed-size output tiles.

**Output destination**

- `output_dir` (`None`): if set, writes a GeoTIFF to this directory using an auto-generated filename and returns the file path. The filename includes a readable request summary plus a short deterministic hash, and a matching `.json` sidecar records the normalized request metadata. Mutually exclusive with `output_path`.
- `output_path` (`None`): if set, writes a GeoTIFF to this exact `.tif`/`.tiff` path and returns it. Mutually exclusive with `output_dir`.
- `overwrite` (`True`): when exporting and the target path exists, controls whether to overwrite it.

**Output grid**

- `output_crs` (`None`): EPSG of the output. In bounds mode, auto-picked from the AOI centroid (UTM zone) if omitted. Ignored in grid mode.
- `resolution` (`10`): output pixel size in metres. At lower resolutions rasterio reads from COG overviews — much less data over the wire.
- `resampling_method` (`"nearest"`): how the source is resampled to the output grid. Also accepts `"bilinear"`, `"cubic"`, `"average"`, `"lanczos"`.

**Scene selection**

- `source` (`"MPC"`): STAC provider. `"MPC"` (default) uses Microsoft Planetary Computer with SAS-signed URLs. `"AWS"` uses Element 84's Earth Search on AWS Open Data — same Sentinel-2 L2A scenes, public S3 (no auth, no SAS rotation). Useful as a fallback when Planetary Computer is having an outage, or for bulk runs that benefit from no rate-limited token issuance.
- `additional_query` (`{"eo:cloud_cover": {"lt": 100}}`): extra STAC query filters, e.g. `{"eo:cloud_cover": {"lt": 80}}`.
- `no_data_tolerance` (`0.0`): scene-selection early stop once the AOI no-data fraction drops below this. The default `0.0` (or `None`) examines every scene. Ignored for `percentile` mosaic methods.
- `min_coverage_fraction` (`0.1`): drop scene-edge pixels covered by fewer than this fraction of the maximum overlap count in the AOI. Set to `None` to keep everything.
- `ignore_duplicate_items` (`True`): drop duplicate acquisitions, keeping the latest processing baseline.

**Scene ordering**

- `sort_method` (`"valid_data"`): scene ordering — `"valid_data"`, `"oldest"`, or `"newest"`.
- `sort_function` (`None`): custom callable `fn(items: pd.DataFrame) -> pd.DataFrame`. Overrides `sort_method` when set.

**Cloud masking**

- `cloud_mask` (`"OCM"`): provider — `"OCM"` runs the OmniCloudMask deep-learning model on R+G+NIR bands (most accurate); `"SCL"` reads the L2A Scene Classification Layer (much cheaper, lower accuracy).
- `ocm_batch_size` (`1`): OCM inference batch size. Only used with `cloud_mask="OCM"`.
- `ocm_inference_dtype` (`"bf16"`): OCM inference dtype. Only used with `cloud_mask="OCM"`.

**Bounds/AOI-mode-specific options**

- `input_crs` (`4326`): EPSG of `bounds` or `aoi`.

For more detailed information on these options and additional functionality, please refer to the function docstring in the source code.

## Logging

S2Mosaic emits progress logs at each pipeline stage (search, sort, fetch, cloud-mask, aggregate, export). Following standard Python logging convention, no output is produced unless logging is configured. The simplest way to enable it:

```python
import s2mosaic

s2mosaic.set_log_level("INFO")  # or "DEBUG" for more detail
```

If your application already configures the `logging` module, the package logger (`s2mosaic`) will respect that — no need to call `set_log_level()`.

## Performance Tips
- `cloud_mask`: Default `"OCM"` runs the OmniCloudMask deep-learning model — most accurate but needs reasonable compute (GPU/MPS recommended). Switch to `"SCL"` on CPU-only machines or for bulk processing — it skips inference entirely and just reads the L2A Scene Classification Layer.
- `ocm_batch_size`: If using a GPU, setting this above the default value (1) will speed up cloud masking. In most cases, a value of 4 works well. If you encounter CUDA errors, try using a lower number.
- `ocm_inference_dtype`: if the device supports it 'bf16' tends to be the fastest option, failing this try 'fp16' then 'fp32'.
- `sort_method`: Using `"valid_data"` tends to work well with early stopping because clear scenes are considered first.
- `observation_target`: For large `"mean"` or `"percentile"` jobs, set this to the number of observations per pixel you actually need to avoid reading later scenes for already-satisfied tiles.
- `mosaic_method`: Using `"first"` can be a lot faster than `"mean"` as only valid, non-cloudy, new pixels are downloaded.

## Known limitations

- **SCL is less accurate than OCM.** The L2A Scene Classification Layer is fast (one COG read per scene, no inference) but is consistently less accurate than OCM at identifying clouds and cloud shadow. Use SCL when compute is the bottleneck (CPU-only machines, bulk processing); use OCM when accuracy matters.

## Contributing

Contributions to S2Mosaic are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

For maintainers: the release flow (cut a tag, GitHub Actions builds + publishes to PyPI) is documented in [RELEASING.md](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/RELEASING.md).

### Debug caching

When iterating on the code or running the slow test suite, set the `S2MOSAIC_DEBUG_CACHE` environment variable to skip repeated STAC and COG fetches:

```bash
export S2MOSAIC_DEBUG_CACHE=1
```

When set (to `1`, `true`, or `yes`), STAC search results, per-scene cloud masks, and per-(scene, band) tile sources are written to a `cache/` directory next to the working directory and reused on subsequent runs. Tile sources are stored as tiled GeoTIFFs on the target grid, so warm-cache reads bypass both the PC download and (for bounds mode) the WarpedVRT reprojection.

## License

This project is licensed under the MIT License.

## Acknowledgments

S2Mosaic is built on top of:

- **[Sentinel-2](https://sentiwiki.copernicus.eu/web/s2-products)** — ESA's Copernicus Earth-observation mission, the imagery source.
- **[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)** — STAC catalog and signed access to the Sentinel-2 L2A archive.
- **[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)** — the deep-learning cloud and cloud-shadow mask used by the default `cloud_mask="OCM"` provider.
- **L2A Scene Classification Layer (SCL)** — the published per-scene classification used by the optional `cloud_mask="SCL"` provider.
- **[rasterio](https://rasterio.readthedocs.io/)**, **[GeoPandas](https://geopandas.org/)**, **[pystac-client](https://pystac-client.readthedocs.io/)**, **[OpenCV](https://opencv.org/)**, **[Numba](https://numba.pydata.org/)**, and **[multiclean](https://github.com/DPIRD-DMA/multiclean)** — supporting libraries for I/O (including per-scene `WarpedVRT` reprojection), geometry, search, image ops, percentile aggregation, and mask post-processing.
