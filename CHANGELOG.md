# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0b1] - 2026-05-15

### Added
- `source` parameter on `mosaic()` selects the STAC imagery source: `"MPC"` (default, Microsoft Planetary Computer with SAS-signed URLs) or `"AWS"` (Element 84 Earth Search on AWS Open Data — public S3, no auth). AWS is useful as a fallback during MPC outages and avoids SAS token rotation for bulk workloads. The `sentinel-2-l2a` collection is used on both providers (L1C, C1, and pre-C1 collections are not searched). Asset-name differences (`B04` vs `red`, `SCL` vs `scl`) are handled internally by the `s2mosaic.sources.Source` abstraction; on AWS, `sat:relative_orbit` is recovered from `s2:product_uri`'s `R\\d+` token, and `s2:mgrs_tile` from `grid:code` (`MGRS-50HMH` → `50HMH`). Element 84 returns 0 items when STAC `query` is combined with `intersects`, so grid-mode precision is restored by client-side post-filtering on `grid:code` after the search.
- STAC `datetime` range strings are now produced via `strftime("%Y-%m-%dT00:00:00Z")` rather than `isoformat() + "Z"`, so both `date` and `datetime` inputs round-trip to valid RFC 3339. `define_dates()` returns `datetime`, so the previous `isoformat()` code happened to produce valid output in production — but only for that input shape. Element 84's pystac-client validation is stricter than MPC's and would reject the date-only form if it ever appeared.
- `cloud_mask` parameter on `mosaic()` selects the cloud-mask provider: `"OCM"` (default, OmniCloudMask deep-learning model) or `"SCL"` (the L2A Scene Classification Layer that ships with each scene). SCL is much cheaper — useful on CPU-only machines and for bulk processing — at the cost of accuracy.
- Bounds mode: pass `bounds=(minx, miny, maxx, maxy)` (with optional `bounds_crs` / `target_crs`) to mosaic an arbitrary rectangle, including AOIs that intersect Sentinel-2 tiles in different UTM zone projections — each scene is streamed through a rasterio `WarpedVRT` onto a common UTM grid.
- Bounds validation: rejects longitudes outside ±180 or latitudes outside ±90 when `bounds_crs=4326` (catches most lat/lon axis swaps), rejects bboxes smaller than 100 square metres or with either side shorter than 10m, and logs a warning for AOIs larger than 200km × 200km without blocking them.
- Small bounds-mode AOIs using `cloud_mask="OCM"` are internally padded to at least 100×100 OCM pixels (20m+ resolution) before inference, then clipped back to the requested bounds so OmniCloudMask has enough spatial context without changing the output extent.
- Pre-commit hooks (`ruff-check`, `ruff-format`, `mypy`) and a pre-push `pytest` hook.
- GitHub Actions CI running lint, type-check, and tests on push and PR.
- `@overload`s on `mosaic()` so the return type narrows to `Tuple[ndarray, dict]` when `output_dir` is omitted and to `Path` when it is set.
- Unit tests covering masking, frequent-coverage, bounds reprojection, bounds validation, and percentile aggregation.
- Transient per-scene fetch failures (network blips, expired SAS tokens, 5xx from MPC) are now retried with exponential backoff (3 attempts, 1/2/4s). If retries are exhausted, the scene is logged at WARNING and skipped instead of aborting the whole mosaic; a summary line reports `N/M scenes failed`. Cloud-mask inference errors are *not* swallowed — they still hard-stop so they can be diagnosed.
- Per-tile `no_data_tolerance` short-circuit: each tile's per-scene time series walks scenes in priority order and stops once the tile's coverage of `coverage_mask` exceeds `1 - tolerance`. Clear tiles finish after one scene; cloudy tiles process more. Replaces the old global short-circuit, which stopped the whole pipeline for every method when one region of the mosaic was sufficiently covered.
- STAC search results are now cached when `S2MOSAIC_DEBUG_CACHE` is set, so dev iteration survives transient PC outages.
- `output_dir` exports now write a JSON sidecar beside the GeoTIFF with the normalized request metadata, source, date window, resolved CRS where relevant, and filename hash inputs.

### Changed
- **Breaking:** `start_year` is now a required keyword-only argument on `mosaic()`. Callers passing it positionally (`mosaic("50HMH", 2023)`) must switch to keyword form (`mosaic("50HMH", start_year=2023)`). The type is now `int` (was `Optional[int]` with a runtime check). All README/notebook examples already use the keyword form.
- **Breaking:** `percentile_value` renamed to `percentile` on `mosaic()`. Callers using `mosaic_method="percentile", percentile_value=N` must switch to `percentile=N`. Reverts the 1.0.0 rename — `percentile_value` was redundant alongside `mosaic_method="percentile"`, and the shorter name matches `numpy.percentile` (0–100 scale).
- **Breaking:** `no_data_threshold` renamed to `no_data_tolerance` on `mosaic()`, and the default changed from `0.01` to `0.0`. The old name suggested a pixel-value cutoff; the parameter is actually an early-stop tolerance — stop walking scenes once the AOI no-data fraction drops below this. Default `0.0` (zero tolerance) examines every scene; pass an explicit `0.01` to restore the previous behaviour.
- **Breaking:** `coverage_threshold` renamed to `min_coverage_fraction` on `mosaic()`. Same semantics (drop pixels covered by less than this fraction of the AOI's max overlap count), clearer name — it's a minimum, expressed as a fraction.
- `SceneFetchError` is now exported from the top-level `s2mosaic` package so callers can `except s2mosaic.SceneFetchError` to distinguish fetch failures from other exceptions.
- Replaced `scipy.ndimage` with OpenCV (`cv2.dilate` / `cv2.resize`) for no-data mask dilation and band resampling — drops the SciPy dependency.
- Mask post-processing now uses `multiclean`.
- "No scenes found" and "no usable scenes" conditions now raise `ValueError` and `RuntimeError` respectively instead of bare `Exception`, so callers can catch them specifically.
- **Both modes now use a unified 2D tile-streaming pipeline for aggregation.** The mosaic is partitioned into ~2048×2048 tiles; each tile is processed in parallel by a `ThreadPoolExecutor`, with the per-(scene, band) tile windows read directly from the COG (grid_id) or via a `WarpedVRT` (bounds). Peak memory for `percentile` / `median` is now bounded by `n_workers × tile_size² × bands × n_scenes × 4 B` (a few GB at typical defaults) rather than the previous `n_scenes × full_output × bands × 4 B` (~65 GB for a full-MGRS percentile over 34 scenes). `mean` / `first` get the same architecture for consistency; their memory was already low. End-to-end on a full-MGRS warm-cache percentile (34 scenes × 10980² × 4 bands): unchanged in wall time vs the old in-memory path, but cold-cache wall drops ~40% and peak RAM drops from ~65 GB to a few GB.
- `download_bands_pool` (grid_id) and the in-place per-scene aggregation loop in `run_bounds_pipeline` have been replaced by the shared streaming pipeline. The bounds-mode helpers `_fetch_one_user_scene` and `_fetch_one_tci` are now unused and have been removed; their `@disk_cache("user_scene", ...)` / `@disk_cache("tci_one", ...)` cache files in `cache/` are safe to delete.
- `S2MOSAIC_DEBUG_CACHE` now caches the per-(scene, band) tile-source as a tiled GeoTIFF on the target grid instead of a pickled full-band ndarray. First debug run materialises the cache; subsequent runs open the local tiled files directly, skipping both PC download and (for bounds mode) WarpedVRT reprojection.
- Bounds mode no longer depends on `stackstac`. Each scene is read directly through a rasterio `WarpedVRT` snapped to the output grid, which removes the eager-stack step and lets the pipeline skip scenes that won't contribute (full coverage already filled, or all-cloud). The output grid is derived from `_target_grid(bounds, resolution, target_crs)`; pixel size and CRS are unchanged from 1.x, but the bbox-to-grid rounding may shift the output by ~1 pixel in width/height or by a sub-pixel fraction in pixel-centre location.
- **Breaking:** auto-generated `output_dir` filenames now use a v2 format with a readable request summary plus a short deterministic hash of output-affecting fields. This avoids collisions between requests that previously shared the same name despite differing in source, resolution, cloud mask, query filters, CRS, or AOI geometry. Use `output_path` when an exact filename is required.
- `stackstac` removed from dependencies.
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
