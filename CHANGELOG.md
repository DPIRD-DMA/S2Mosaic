# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - Unreleased

### Added
- `cloud_mask` parameter on `mosaic()` selects the cloud-mask provider: `"OCM"` (default, OmniCloudMask deep-learning model) or `"SCL"` (the L2A Scene Classification Layer that ships with each scene). SCL is much cheaper — useful on CPU-only machines and for bulk processing — at the cost of accuracy.
- Bounds mode: pass `bounds=(minx, miny, maxx, maxy)` (with optional `bounds_crs` / `target_crs`) to mosaic an arbitrary rectangle, including AOIs that intersect Sentinel-2 tiles in different UTM zone projections — scenes from intersecting tiles are reprojected onto a common UTM grid via stackstac.
- Bounds validation: rejects longitudes outside ±180 or latitudes outside ±90 when `bounds_crs=4326` (catches most lat/lon axis swaps), and rejects bboxes whose width or height falls outside the 10m–200km range.
- Pre-commit hooks (`ruff-check`, `ruff-format`, `mypy`) and a pre-push `pytest` hook.
- GitHub Actions CI running lint, type-check, and tests on push and PR.
- `@overload`s on `mosaic()` so the return type narrows to `Tuple[ndarray, dict]` when `output_dir` is omitted and to `Path` when it is set.
- Unit tests covering masking, frequent-coverage, bounds reprojection, bounds validation, and percentile aggregation.
- Transient per-scene fetch failures (network blips, expired SAS tokens, 5xx from MPC) are now retried with exponential backoff (3 attempts, 1/2/4s). If retries are exhausted, the scene is logged at WARNING and skipped instead of aborting the whole mosaic; a summary line reports `N/M scenes failed`. Cloud-mask inference errors are *not* swallowed — they still hard-stop so they can be diagnosed.

### Changed
- Replaced `scipy.ndimage` with OpenCV (`cv2.dilate` / `cv2.resize`) for no-data mask dilation and band resampling — drops the SciPy dependency.
- Mask post-processing now uses `multiclean`.
- "No scenes found" and "no usable scenes" conditions now raise `ValueError` and `RuntimeError` respectively instead of bare `Exception`, so callers can catch them specifically.
- Bounds mode now streams per-scene fetches and accumulates the mosaic in place (mirroring grid mode's `download_bands_pool`). Peak memory is dropped from O(N scenes) to O(one scene + accumulator) for `mosaic_method="mean"` and `"first"`; benchmarked on a 22 × 22km Perth AOI × 28 scenes × 4 bands, peak RSS fell from 15 GB → 2.1 GB (~7×). `"percentile"` / `"median"` still buffer N scenes (exact percentiles need all values per pixel) but avoid stackstac's transient float32 doubling.
- Bounds-mode output grid is now derived from `_target_grid(bounds, resolution, target_crs)` for the user bands (matching the cloud-mask, coverage-mask, and TCI paths) instead of stackstac's `snap_bounds=True` global-aligned grid. The output may differ by ~1 pixel in width/height from 1.x for the same bounds, and pixel centres can shift by a sub-pixel fraction — the math is unchanged (verified bit-exact against `aggregate_stack`), only the bbox-to-grid rounding moved.
- `uv.lock` is no longer tracked in the repo.

## [1.1.0] - 2026-01-22

### Changed
- Updated OmniCloudMask to v1.7.

## [1.0.1] - 2025-07-10

### Changed
- Bumped OmniCloudMask to v1.3.
- Relocated the Sentinel-2 index file inside the package.

### Fixed
- Replaced mutable default arguments with `None` sentinels.

## [1.0.0] - 2025-06-23

### Added
- `percentile` and `median` mosaic methods, with a `percentile_value` parameter for the percentile method.
- `ignore_duplicate_items` option to dedupe scenes by ID.
- Vectorised rasterisation in the coverage pipeline.
- `uv` packaging support and a test suite.

### Changed
- Renamed the `percentile` parameter to `percentile_value` for clarity.
- Improved download caching logic.
- Internal refactor of the percentile aggregation path.

## [0.3.2] - 2025-04-10

### Added
- `additional_query` parameter for STAC search filters (e.g. `{"eo:cloud_cover": {"lt": 80}}`).
- GeoPackage (`.gpkg`) support for grid lookups.

### Changed
- Improved `get_extent_from_grid_id`.

## [0.3.1] - 2025-03-17

### Fixed
- Handle the `proj:epsg` → `proj:code` rename in the Planetary Computer STAC v2.0 catalog.

## [0.3.0] - 2024-12-30

### Added
- Custom `sort_function` option for user-defined scene ordering.

## [0.2.1] - 2024-12-22

### Added
- Frequent-coverage filter: drops scene-edge pixels covered by only a small minority of overlapping scenes.

## [0.1.9] - 2024-11-07

### Fixed
- Slight dilation of the no-data mask removes diagonal no-data pixels at scene edges.

## [0.1.8] - 2024-08-09

### Added
- Support for non-10m bands (auto-resampled to the target resolution).

## [0.1.7] - 2024-08-07

### Added
- `mosaic_method="first"` for first-valid-pixel composites, with optimised partial downloads (only valid, non-cloudy, unfilled pixels are fetched).
- Chunked band reading via rasterio windows + COG overviews.

## [0.1.5] - 2024-08-05

### Changed
- Internal restructure: split download/coordinator/helpers modules.

## [0.1.4] - 2024-08-01

### Fixed
- No-data value handling bug.

## [0.1.3] - 2024-08-01

### Added
- Initial release.
- Mosaic creation by MGRS grid ID and time range.
- Sort methods: `valid_data`, `oldest`, `newest`.
- Mosaic method: `mean`.
- OmniCloudMask integration for cloud and cloud-shadow masking.
- Visual (TCI) and arbitrary-band output.
- GeoTIFF export and NumPy array return.
