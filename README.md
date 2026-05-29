<p align="center">
  <img src="https://raw.githubusercontent.com/DPIRD-DMA/S2Mosaic/main/assets/s2mosaic-heading.svg" alt="S2Mosaic" width="600">
</p>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/s2mosaic.svg)](https://pypi.org/project/s2mosaic/) [![Python versions](https://img.shields.io/pypi/pyversions/s2mosaic.svg)](https://pypi.org/project/s2mosaic/) [![CI](https://github.com/DPIRD-DMA/S2Mosaic/actions/workflows/ci.yml/badge.svg)](https://github.com/DPIRD-DMA/S2Mosaic/actions/workflows/ci.yml) [![Downloads](https://static.pepy.tech/badge/s2mosaic)](https://pepy.tech/project/s2mosaic) [![License](https://img.shields.io/pypi/l/s2mosaic.svg)](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/LICENSE)

</div>

S2Mosaic is a Python package for creating cloud-free mosaics from Sentinel-2 satellite imagery. It can mosaic full MGRS tiles, rectangular bounds, or polygon AOIs over a chosen time window, with configurable scene ordering, masking, aggregation, and export options.

## Features

- Mosaic by MGRS grid tile (`grid_id`), rectangular bounds (`bounds`), or a single polygon (`aoi`) — bounds/AOIs can cross MGRS tile boundaries and are reprojected onto a common UTM grid in one step.
- Flexible scene ordering: by valid data percentage, oldest, newest, or a custom sort function.
- Multiple mosaic creation methods: mean, arbitrary percentile, median, medoid (scene closest to per-band median — preserves real observed spectra) or first valid pixel.
- Support for different spectral bands, including visual (RGB) composites.
- State-of-the-art cloud masking using OmniCloudMask, with an SCL option for computationally cheaper bulk processing.
- STAC source selection: Microsoft Planetary Computer by default, or Element 84 Earth Search on AWS Open Data.
- Tile-streaming pipeline keeps peak memory low even for full-MGRS percentile mosaics over many scenes — the aggregation is parallelised across ~2048-pixel tiles, so only a handful of tile-sized buffers live in RAM at a time.
- Resilient to transient COG read failures — per-scene fetches retry with exponential backoff, and a scene that still fails is logged and skipped so one bad asset doesn't abort the whole mosaic.
- Export mosaics as GeoTIFF files or return as NumPy arrays.

## Changelog

See [CHANGELOG.md](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/CHANGELOG.md) for the full release history.

## Note

S2Mosaic uses OmniCloudMask (OCM) by default for cloud and cloud-shadow masking. OCM runs much faster when an NVIDIA GPU or MPS accelerator is available. When compute is limited or throughput matters more than mask quality, pass `cloud_mask="SCL"` to skip the deep-learning model and use the Sentinel-2 L2A Scene Classification Layer instead.

## Try in Colab

[![Colab_Button]][Link]

[Link]: https://colab.research.google.com/github/DPIRD-DMA/S2Mosaic/blob/main/examples/Quick%20start.ipynb 'Try S2Mosaic In Colab'

[Colab_Button]: https://img.shields.io/badge/Try%20in%20Colab-grey?style=for-the-badge&logo=google-colab

The Colab badge opens [`examples/Quick start.ipynb`](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Quick%20start.ipynb) — a minimal end-to-end demo.



## Installation

S2Mosaic 2.0 is currently in beta. Install with pip:
```
pip install --pre s2mosaic
```
Or with uv:
```
uv add --prerelease=allow s2mosaic
```
Drop `--pre` / `--prerelease=allow` once `2.0.0` ships stable.
## Quick start — MGRS grid tile

Mosaic an entire Sentinel-2 MGRS tile by ID, returning a NumPy array and rasterio profile. Find the tile ID for your area of interest with the [Sentinel-2 grid explorer](https://dpird-dma.github.io/Sentinel-2-grid-explorer/).

```python
from s2mosaic import mosaic

array, profile = mosaic(
    grid_id="50HMH",            # Sentinel-2 MGRS tile ID
    start_year=2022,
    start_month=1,
    duration_months=2,          # collect scenes over this window
    scene_order="valid_data",   # prioritise scenes with the most valid pixels
    mosaic_method="mean",       # combine valid pixels by mean
    bands=["B04", "B03", "B02", "B08"],
)

print(f"Mosaic array shape: {array.shape}")
print(f"CRS: {profile['crs']}")
```

To save a GeoTIFF instead of returning the array, pass `output_dir=Path("output")`
for an auto-generated filename, or `output_path=Path("output/custom.tif")` to
choose the exact filename. The function then returns the file path.

Auto-generated `output_dir` filenames include a readable summary of the target,
date range, bands, method, scene order, resolution, cloud-mask provider, source, and a short
deterministic hash of the output-affecting request fields. A matching `.json`
sidecar is written next to the GeoTIFF with the normalized request metadata, so
similar requests do not silently collide while the filename stays scan-friendly.

## Quick start — arbitrary bounding box

Pass `bounds=(minx, miny, maxx, maxy)` instead of `grid_id` to mosaic any rectangular AOI, including ones that cross MGRS tile boundaries. Each intersecting scene is streamed through a rasterio `WarpedVRT` and aggregated onto a common UTM grid.

```python
from s2mosaic import mosaic

# ~7.6km x 3.4km AOI near Perth, WA, in UTM zone 50S (EPSG:32750)
bounds = (389410, 6462290, 397010, 6465700)

array, profile = mosaic(
    bounds=bounds,
    input_crs=32750,
    snap_to_source_grid=True,
    start_year=2023,
    start_month=6,
    duration_months=2,
    bands=["B04", "B03", "B02"],
    mosaic_method="median",
)

print(f"Shape: {array.shape}")
print(f"CRS:   {profile['crs']}")
print(f"Pixel: {profile['transform'].a}m")
```

`bounds=` always fills the requested rectangle: same-CRS uses it directly, cross-CRS (e.g. lon/lat input → UTM output) uses its reprojected axis-aligned envelope. There is no implicit polygon mask, so cross-CRS bounds don't produce nodata wedges at the corners — the envelope is just slightly larger than the original lon/lat region. The recommended pattern for single-zone AOIs is to pass bounds in the local UTM zone so `input_crs == output_crs` (no envelope inflation) with `snap_to_source_grid=True` (zero resampling against the source grid). Use lon/lat input for AOIs that genuinely span multiple UTM zones, or pass `aoi=shapely.geometry.box(*bounds)` instead if you want the lat/lon rectangle clipped after reprojection. `output_crs` defaults to the UTM zone containing the AOI centroid if omitted. Use `resolution` (in metres) and `resampling_method` (`nearest`, `bilinear`, ...) to control the output grid. See [Example use - bounds.ipynb](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Example%20use%20-%20bounds.ipynb) for cross-tile and lower-resolution examples, [Example use - aoi.ipynb](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Example%20use%20-%20aoi.ipynb) for polygon AOIs, and [Advanced - wide area visual export.ipynb](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Advanced%20-%20wide%20area%20visual%20export.ipynb) for a large, wide visual-only GeoTIFF export that streams tiles directly to disk.

## Mosaic method comparison

Different mosaic methods trade speed, smoothness, and spectral consistency. In this cloudy Singapore example, `medoid` keeps the sharper real-scene look of an observed pixel while avoiding the most scene-order-dependent artefacts from `first` and the blended look of `mean` / `median`.

![S2Mosaic method comparison showing first valid, mean, median, and medoid mosaics over a cloudy Singapore AOI](https://raw.githubusercontent.com/DPIRD-DMA/S2Mosaic/main/assets/mosaic-method-comparison.jpg)

The image was generated by [Mosaic method comparison.ipynb](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/examples/Mosaic%20method%20comparison.ipynb).

## Advanced Usage

S2Mosaic provides several options for customizing the mosaic creation process. Defaults shown in parentheses.

**Area (pass exactly one)**

- `grid_id` (`None`): Sentinel-2 MGRS tile ID, e.g. `"50HMH"`. Mosaics the entire tile.
- `bounds` (`None`): `(minx, miny, maxx, maxy)` rectangle. Mosaics an arbitrary AOI, including ones that cross MGRS tile boundaries. See the bounds/AOI-mode-specific options below.
- `aoi` (`None`): single shapely `Polygon`. Mosaics the polygon bounds while skipping and masking pixels outside the polygon. Mutually exclusive with `grid_id` and `bounds`.

**Time window**

- `start_year` (required), `start_month` (`1`), `start_day` (`1`): start of the scene-search window.
- `duration_years` (`0`), `duration_months` (`0`), `duration_days` (`0`): length of the search window. Inclusive of `start_*`, exclusive of the end.

**Output content**

- `bands` (`["B04", "B03", "B02", "B08"]`): bands to include. Leave as `None` to use the default RGB+NIR set. Use `["visual"]` for the 3-band uint8 TCI RGB composite (mutually exclusive with other bands).
- `mosaic_method` (`"mean"`): how per-pixel scene stacks are reduced to one output value.
  - `"mean"`: per-band arithmetic mean of all valid scenes. Streams scenes incrementally so peak memory stays small.
  - `"first"`: first valid pixel in `scene_order`. Cheapest method — reads only what's needed to fill each tile and stops as soon as it can.
  - `"percentile"`: per-band percentile across all valid scenes. Requires the `percentile` parameter (0–100).
  - `"median"`: shortcut for `"percentile"` with `percentile=50`.
  - `"medoid"`: picks the scene whose multi-band spectrum is closest (squared Euclidean) to the per-band median across all valid scenes for the pixel. Preserves real observed spectra, so band relationships stay coherent for indices and classifiers. This is the approximate-medoid formulation popularised by the gee-community / Open-MRV tutorials, not the strict Flood 2013 pairwise-distance medoid; the two often agree, but can differ.
- `percentile` (`None`): percentile to compute when `mosaic_method="percentile"` (0–100).
- `min_observations` (`None`): minimum valid observations to read per pixel for `"mean"`, `"percentile"`, and `"medoid"`. When set, tile aggregation stops reading later scenes once every coverable pixel has reached the target. This is not an output quality guarantee; pixels that cannot reach the target use whatever observations are available.
- `max_observations` (`None`): per-pixel cap on valid observations for `"mean"`, `"percentile"`, and `"medoid"`. Each pixel accepts at most this many scenes (in `scene_order`); later valid scenes are dropped for that pixel. Combined with `scene_order="oldest"` or `"newest"` this biases the mosaic toward early or late dates. Must be `>= min_observations` when both are set; ignored by `"first"` (effectively N=1).
- `include_observation_count` (`False`): append a final `Observation count` band with the number of valid source scenes that contributed to each output pixel. Works for both returned arrays and exported GeoTIFFs. For `bands=["visual"]`, enabling this writes/returns `uint16` output so the count band is not limited to 255; the RGB values remain in their usual 0–255 range.
- `tile_workers` (`8`): number of output tiles to aggregate concurrently. Tuned higher than CPU count because the work is I/O-bound on remote COG reads. Raise for faster networks; lower if memory or simultaneous-connection limits matter more than throughput.
- `adaptive_tiling` (`True`): split sparse output tiles based on the actual cloud-valid contribution masks. This reduces wasted reads for irregular AOIs, sparse coverage, and heavily masked scenes. Set to `False` to use fixed-size output tiles.

**Output destination**

- `output_dir` (`None`): if set, writes a GeoTIFF to this directory using an auto-generated filename and returns the file path. The filename includes a readable request summary plus a short deterministic hash, and a matching `.json` sidecar records the normalized request metadata. Mutually exclusive with `output_path`.
- `output_path` (`None`): if set, writes a GeoTIFF to this exact `.tif`/`.tiff` path and returns it. Mutually exclusive with `output_dir`.
- `overwrite` (`True`): when exporting and the target path exists, controls whether to overwrite it.

**Output grid**

- `output_crs` (`None`): EPSG of the output. Must be a projected CRS — geographic CRSes (e.g. 4326) are rejected at validation, because `resolution` is metres in the target CRS and a geographic output would produce a degenerate grid. If you need a lat/lon raster, reproject the mosaic afterwards with `gdalwarp` / `rio warp`. In bounds/AOI mode, auto-picked as the UTM zone containing the AOI centroid if omitted. For AOIs wider than ~6° of longitude (one UTM zone), pass an explicit equal-area projection instead (e.g. `output_crs=3577` for Australia, `5070` for the contiguous US) — the auto-picked centroid UTM has growing scale distortion and a larger envelope overshoot far from its central meridian. Ignored in grid mode (the tile's native UTM zone is used).
- `resolution` (`10`): output pixel size in metres. At lower resolutions rasterio reads from COG overviews — much less data over the wire.
- `resampling_method` (`"nearest"`): how the source is resampled to the output grid. Also accepts `"bilinear"`, `"cubic"`, `"average"`, `"lanczos"`.
- `snap_to_source_grid` (`False`): bounds/AOI mode only. When `True`, expand the output extent outward to whole multiples of `resolution` in the target CRS. This makes repeat runs over the same area produce identical grids, and at `resolution=10` aligns the output to the native Sentinel-2 pixel grid — source COG reads become zero-cost copies rather than sub-pixel resamples. The output may grow by up to one pixel on each side; pixels outside an `aoi` polygon are still written as nodata.

**Scene selection**

- `source` (`"MPC"`): STAC provider. `"MPC"` (default) uses Microsoft Planetary Computer with SAS-signed URLs. `"AWS"` uses Element 84's Earth Search on AWS Open Data — Sentinel-2 L2A scenes, public COGs, no auth, no SAS rotation.
- `additional_query` (`{"eo:cloud_cover": {"lt": 100}}`): extra STAC query filters, e.g. `{"eo:cloud_cover": {"lt": 80}}`.
- `min_coverage_fraction` (`None`): optional scene-edge trimming. When set, drops pixels covered by fewer than this fraction of the maximum scene-overlap count in the requested area. The default keeps the full requested coverage.
- `ignore_duplicate_items` (`True`): drop duplicate acquisitions, keeping the latest processing baseline.

**Scene ordering**

- `scene_order` (`"valid_data"`): scene ordering — `"valid_data"`, `"oldest"`, or `"newest"`.
- `scene_sort_fn` (`None`): custom callable `fn(items: pd.DataFrame) -> pd.DataFrame`. Overrides `scene_order` when set.

**Cloud masking**

- `cloud_mask` (`"OCM"`): provider — `"OCM"` runs the OmniCloudMask deep-learning model on R+G+NIR bands (most accurate); `"SCL"` reads the L2A Scene Classification Layer (much cheaper, lower accuracy).
- `ocm_batch_size` (`1`): OCM inference batch size. Only used with `cloud_mask="OCM"`.
- `ocm_inference_dtype` (`"fp32"`): OCM inference dtype. Defaults to `"fp32"` — runs everywhere and is the fastest option on CPU. On GPU, use `"fp16"` for ~2× speedup and lower VRAM, or `"bf16"` on hardware that supports it. Only used with `cloud_mask="OCM"`.

**Diagnostics**

- `show_progress` (`False`): show tqdm progress bars for the cloud-mask streaming and tile-aggregation phases. Useful in notebooks; leave off for headless/batch runs.

Example:

```python
array, profile = mosaic(
    bounds=(389410, 6462290, 397010, 6465700),
    input_crs=32750,
    snap_to_source_grid=True,
    start_year=2023,
    duration_months=2,
    bands=["visual"],
    mosaic_method="mean",
    include_observation_count=True,
)

rgb = array[:3]
observation_count = array[3]
print(array.shape)  # (4, height, width): Red, Green, Blue, Observation count
```

**Bounds/AOI-mode-specific options**

- `input_crs` (`4326`): EPSG of `bounds` or `aoi`.

For the exact function signature and return types, see the `mosaic()` docstring in the source code.

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
- `ocm_inference_dtype`: defaults to `'fp32'` — runs on every backend and is the fastest option on CPU (most CPUs don't have efficient fp16/bf16 paths). On GPU, switch to `'fp16'` for ~2× faster inference and lower VRAM use, or `'bf16'` on hardware that supports it (Ampere+ NVIDIA, Apple Silicon).
- `scene_order`: Using `"valid_data"` tends to work well with early stopping because clear scenes are considered first.
- `min_observations`: For large `"mean"`, `"percentile"`, or `"medoid"` jobs, set this to the number of observations per pixel you actually need to avoid reading later scenes for already-satisfied tiles.
- `max_observations`: Caps each pixel at N valid scenes. Combine with `scene_order="oldest"` (or `"newest"`) to bias the mosaic toward early/late dates over a long search window without paying for the extra reads.
- `mosaic_method`: Roughly ordered fastest to slowest, `"first"` < `"mean"` < `"percentile"`/`"median"` ≈ `"medoid"`. `"first"` only reads pixels needed to fill each tile and stops as soon as it can, so cloud-free scenes can finish a tile in one pass. `"mean"` streams every contributing scene but accumulates incrementally, so its memory stays small. `"percentile"`/`"median"` and `"medoid"` hold each tile's per-scene stack in memory to compute the result, so they use more RAM and — without `min_observations`/`max_observations` — read every contributing scene. `"medoid"` keeps its scene stack as `uint16` plus a validity mask and stripe-blocks scratch arrays, so peak per-tile memory is lower than percentile/median's `float32+NaN` stack. Set `min_observations` (and/or `max_observations`) to cap reads once every coverable pixel has enough samples.

## Known limitations

- **SCL is less accurate than OCM.** The L2A Scene Classification Layer is fast (one COG read per scene, no inference) but is consistently less accurate than OCM at identifying clouds and cloud shadow. Use SCL when compute is the bottleneck (CPU-only machines, bulk processing); use OCM when accuracy matters.

## Contributing

Contributions to S2Mosaic are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

### Running the tests

Tests use `pytest`. The fast suite (unit tests + mocked pipelines) runs in under 15s and is what CI runs by default:

```bash
uv run pytest                       # full fast suite
uv run pytest tests/test_readers.py # one file
uv run pytest -k requeue            # match by name
```

End-to-end tests that hit the network and run a real mosaic are marked `slow` and excluded by default (see `addopts` in `pyproject.toml`). To run them explicitly:

```bash
uv run pytest -m slow               # only slow tests
uv run pytest -m ""                 # everything, including slow
```

Lint with ruff:

```bash
uv run ruff check s2mosaic/ tests/
```

For maintainers: the release flow (cut a tag, GitHub Actions builds + publishes to PyPI) is documented in [RELEASING.md](https://github.com/DPIRD-DMA/S2Mosaic/blob/main/RELEASING.md).

## License

This project is licensed under the MIT License.

## Acknowledgments

S2Mosaic is built on top of:

- **[Sentinel-2](https://sentiwiki.copernicus.eu/web/s2-products)** — ESA's Copernicus Earth-observation mission, the imagery source.
- **[Element 84 Earth Search](https://earth-search.aws.element84.com/)** — optional public AWS Open Data access to Sentinel-2 L2A COGs.
- **[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)** — the default STAC catalog and signed access to the Sentinel-2 L2A archive.
- **[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)** — the deep-learning cloud and cloud-shadow mask used by the default `cloud_mask="OCM"` provider.
- **L2A Scene Classification Layer (SCL)** — the published per-scene classification used by the optional `cloud_mask="SCL"` provider.
- **[rasterio](https://rasterio.readthedocs.io/)**, **[GeoPandas](https://geopandas.org/)**, **[pystac-client](https://pystac-client.readthedocs.io/)**, **[OpenCV](https://opencv.org/)**, **[Numba](https://numba.pydata.org/)**, and **[multiclean](https://github.com/DPIRD-DMA/multiclean)** — supporting libraries for I/O (including per-scene `WarpedVRT` reprojection), geometry, search, image ops, percentile aggregation, and mask post-processing.
