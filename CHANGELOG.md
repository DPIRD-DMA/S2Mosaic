# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]


## [2.0.0b1] - 2026-05-28

### Added
- `mosaic_method="medoid"` selects a per-pixel medoid composite. For each pixel the kernel picks the scene whose multi-band spectrum is closest (squared Euclidean) to the per-band median across all valid scenes for that pixel. Unlike per-band `"median"` / `"percentile"` the result is always an actually-observed spectrum — band relationships are preserved, which matters for spectral indices and downstream classifiers. This is the *approximate* medoid that the gee-community / Open-MRV tutorials popularised (O(S·B) per pixel), not the strict Flood 2013 medoid (`arg min_s Σᵢ d(s,i)`, O(S²·B)); the two often agree, but can differ. The kernel keeps its per-tile stack as `uint16` plus a separate `(scene, h, w) bool` validity mask rather than `float32+NaN`, uses exact doubled integer median targets so even-count half-integer medians are not rounded, and stripe-blocks scratch arrays to lower peak per-tile memory. `nogil=True` lets the existing tile-worker pool actually run the kernel in parallel rather than serialising on the GIL. Works for both reflectance bands (uint16) and the `"visual"` uint8 RGB mode through the same code path.
- `nogil=True` and a hoisted per-pixel `values` allocation on `_nanquantile_axis0` (the percentile/median kernel). Tile workers now actually compute in parallel inside the kernel instead of serialising on the GIL, and Numba's allocator no longer takes a per-pixel lock. Around 17% faster single-thread plus near-perfect parallel scaling at 8 tile workers. No API change; the existing `"percentile"` and `"median"` mosaics just get faster.
- `source` parameter on `mosaic()` selects the STAC imagery source: `"MPC"` (default, Microsoft Planetary Computer with SAS-signed URLs) or `"AWS"` (Element 84 Earth Search on AWS Open Data — public S3, no auth). The `sentinel-2-l2a` collection is used on both providers (L1C, C1, and pre-C1 collections are not searched). Asset-name differences (`B04` vs `red`, `SCL` vs `scl`) are handled internally by the `s2mosaic.sources.Source` abstraction; on AWS, `sat:relative_orbit` is recovered from `s2:product_uri`'s `R\\d+` token, and `s2:mgrs_tile` from `grid:code` (`MGRS-50HMH` → `50HMH`). Element 84 returns 0 items when STAC `query` is combined with `intersects`, so grid-mode precision is restored by client-side post-filtering on `grid:code` after the search.
- STAC `datetime` range strings are now produced via `strftime("%Y-%m-%dT00:00:00Z")` rather than `isoformat() + "Z"`, so both `date` and `datetime` inputs round-trip to valid RFC 3339. `define_dates()` returns `datetime`, so the previous `isoformat()` code happened to produce valid output in production — but only for that input shape. Element 84's pystac-client validation is stricter than MPC's and would reject the date-only form if it ever appeared.
- `cloud_mask` parameter on `mosaic()` selects the cloud-mask provider: `"OCM"` (default, OmniCloudMask deep-learning model) or `"SCL"` (the L2A Scene Classification Layer that ships with each scene). SCL is much cheaper — useful on CPU-only machines and for bulk processing — at the cost of accuracy.
- Bounds mode: pass `bounds=(minx, miny, maxx, maxy)` (with optional `bounds_crs` / `target_crs`) to mosaic an arbitrary rectangle, including AOIs that intersect Sentinel-2 tiles in different UTM zone projections — each scene is streamed through a rasterio `WarpedVRT` onto a common UTM grid.
- AOI mode: pass `aoi=<shapely Polygon>` (with optional `input_crs`) to mosaic a single-polygon AOI alongside the existing `grid_id` / `bounds` modes. The output raster uses the polygon's bounding box; pixels outside the polygon are written as nodata. Shares the bounds-mode streaming pipeline (cross-zone-safe `WarpedVRT` reads on a common UTM grid).
- Bounds validation: rejects longitudes outside ±180 or latitudes outside ±90 when `bounds_crs=4326` (catches most lat/lon axis swaps), rejects bboxes smaller than 100 square metres or with either side shorter than 10m, and logs a warning for AOIs larger than 200km × 200km without blocking them.
- `output_crs` validation: reject geographic CRSes (e.g. `4326`) with a clear error. `resolution` is interpreted as metres in the target CRS, so a geographic output would silently produce a degenerate grid (a `resolution=10` request would land ~10 *degrees* per pixel). Users who need a lat/lon raster are pointed at `gdalwarp` / `rio warp` to reproject the UTM output. The error fires in all modes, even grid mode where `output_crs` is otherwise ignored, to keep the message consistent.
- `bounds=` now has a single, predictable behaviour: it always fills the requested rectangle (or its reprojected axis-aligned envelope for cross-CRS input), regardless of whether `input_crs == output_crs`. Previously, cross-CRS `bounds=` silently synthesised a densified polygon from the input rectangle and used it as an AOI mask, producing nodata wedges at the envelope corners — a different behaviour from same-CRS `bounds=` (which fills the rectangle) that was easy to mistake for source-data gaps. Callers who want the lat/lon-rectangle clip after reprojection now use `aoi=shapely.geometry.box(*bounds)` explicitly. STAC scene search also follows: cross-CRS `bounds=` uses bbox search instead of the previous polygon search.
- Small bounds-mode AOIs using `cloud_mask="OCM"` are internally padded to at least 100×100 OCM pixels (20m+ resolution) before inference, then clipped back to the requested bounds so OmniCloudMask has enough spatial context without changing the output extent.
- Pre-commit hooks (`ruff-check`, `ruff-format`, `mypy`) and a pre-push `pytest` hook.
- GitHub Actions CI running lint, type-check, and tests on push and PR.
- `@overload`s on `mosaic()` so the return type narrows to `Tuple[ndarray, dict]` when `output_dir` is omitted and to `Path` when it is set.
- Unit tests covering masking, frequent-coverage, bounds reprojection, bounds validation, and percentile aggregation.
- Transient per-scene fetch failures (network blips, expired SAS tokens, 5xx from MPC) are now retried with exponential backoff (3 attempts, 1/2/4s). If retries are exhausted, the scene is logged at WARNING and skipped instead of aborting the whole mosaic; a summary line reports `N/M scenes failed`. Cloud-mask inference errors are *not* swallowed — they still hard-stop so they can be diagnosed.
- Per-tile early stop: each tile's per-scene time series walks scenes in priority order and stops when the tile is "done" — for `first`, once every pixel is filled; for `mean`/`percentile`, once every coverable pixel has at least `min_observations` valid observations (when set). Clear tiles finish after one scene; cloudy tiles process more. Replaces the old global short-circuit, which stopped the whole pipeline for every method when one region of the mosaic was sufficiently covered.
- STAC search results are now cached when `S2MOSAIC_DEBUG_CACHE` is set, so dev iteration survives transient PC outages.
- GeoTIFF exports (`output_dir` or `output_path`) now write a JSON sidecar beside the raster with the normalized request metadata, source, date window, resolved CRS where relevant, and filename hash inputs.
- `max_observations` parameter on `mosaic()` for `"mean"` and `"percentile"` modes. Per-pixel cap: each pixel accepts at most N valid scenes (in `scene_order`), with later valid scenes dropped for that pixel. Pairs with `scene_order="oldest"`/`"newest"` to bias the mosaic toward early or late dates. Tile-streamed: `tile_mean` masks the accumulator on `count < max_observations`; `tile_percentile` writes NaN into the per-scene stack past the per-pixel cap, and the existing NaN-skipping quantile kernel handles the rest. Stop condition in `_contributing_scene_indices` uses `max(min_observations, max_observations)`, so reads halt once every coverable pixel saturates. Validated as a positive integer with `max_observations >= min_observations` when both are set; ignored by `"first"` (effectively N=1).
- `show_progress` parameter on `mosaic()` (off by default). When enabled, renders two tqdm bars: Phase 1 ticks per-scene cloud-mask compute, Phase 2 ticks per (scene, tile) × bands during the streaming aggregation. Phase 2's total is an upper bound and fast-forwards on early-stop modes (`first`, per-tile `min_observations`).
- `adaptive_tiling` parameter on `mosaic()` (on by default). Splits sparse output tiles based on the actual cloud-valid contribution masks, so irregular AOIs and sparse scene coverage stop paying full-tile read cost for mostly-empty tiles.
- `tile_workers` parameter on `mosaic()` for explicit control over the streaming-aggregation pool size. Defaults are tuned higher than CPU count (the work is I/O-bound on remote COG reads); raise for faster networks, lower if memory-constrained.
- `apply_gdal_network_defaults()` is invoked at the top of `mosaic()` so HTTP/2, byte-range caching, and connect/read timeouts are applied to every worker thread (previously these were set via `rasterio.Env`, which is thread-local and never reached the pool workers used in the hot path).
- `snap_to_source_grid` parameter on `mosaic()` (default `False`). When `True`, the bounds/AOI output extent is expanded outward to whole multiples of `resolution` in the target CRS. Repeat runs over the same area produce identical grids (useful for compositing and change detection), and at `resolution=10` the output aligns with the native Sentinel-2 10m grid so reads become true zero-cost copies instead of sub-pixel resamples. The output may grow by up to one pixel on each side; AOI polygons are still respected (pixels outside the polygon write nodata).
- `include_observation_count` parameter on `mosaic()` (default `False`). When enabled, outputs append a final `Observation count` band containing the number of valid source scenes that contributed to each pixel. Works for both returned arrays and GeoTIFF exports. Visual RGB outputs are promoted to `uint16` when this flag is enabled so the count band is not limited to 255; RGB values remain in the usual 0–255 range.


### Changed
- **Breaking:** `start_year` is now a required keyword-only argument on `mosaic()`. Callers passing it positionally (`mosaic("50HMH", 2023)`) must switch to keyword form (`mosaic("50HMH", start_year=2023)`). The type is now `int` (was `Optional[int]` with a runtime check). All README/notebook examples already use the keyword form.
- **Breaking:** `percentile_value` renamed to `percentile` on `mosaic()`. Callers using `mosaic_method="percentile", percentile_value=N` must switch to `percentile=N`. Reverts the 1.0.0 rename — `percentile_value` was redundant alongside `mosaic_method="percentile"`, and the shorter name matches `numpy.percentile` (0–100 scale).
- **Breaking:** `required_bands` renamed to `bands`, `observation_target` renamed to `min_observations`, `sort_method` renamed to `scene_order`, and `sort_function` renamed to `scene_sort_fn`.
- `source` defaults to `"MPC"`.
- `ocm_inference_dtype` now defaults to `"fp32"` instead of `"fp16"` so OCM works out-of-the-box on any backend. fp32 is also the fastest option on CPU (most CPUs lack efficient fp16/bf16 paths), which is the most common "no GPU configured" scenario. GPU users can opt in to `"fp16"` for ~2× faster inference and lower VRAM, or `"bf16"` on hardware that supports it (Ampere+ NVIDIA, Apple Silicon).
- **Breaking:** `min_coverage_fraction` now defaults to `None` instead of `0.1`, so scene-edge coverage trimming is opt-in.
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
- Bounds- and AOI-mode per-scene mask reads (OCM and SCL) are now clipped to the scene's polygon-intersected bounds window and stored as a sparse windowed mask, instead of being sized to the full AOI extent. Reads pick the right COG overview level for both same-CRS (direct read with `out_shape`) and cross-CRS (`WarpedVRT` on an overview-opened source) paths. For wide AOIs this drops per-scene mask cost from ~1 GB allocated mostly outside the scene's footprint to a window sized to the scene itself.
- SCL is read at its native 20m in bounds/AOI mode instead of being upsampled to the user resolution. Coverage-mask materialisation is also deferred via lazy array-like wrappers so per-tile mask reads stay window-sized.
- Per-(scene, tile) band reads are now issued concurrently and the default tile-worker count is raised to 8 — remote-COG reads are I/O-bound and benefit from more workers than CPU count would suggest (~400 Mbps sweet spot).
- Tiled GeoTIFF writes set `BIGTIFF=IF_SAFER`, and the expected-reads scan short-circuits on very large mosaics.
- `stackstac` removed from dependencies.
- `uv.lock` is no longer tracked in the repo.
- **Breaking:** the v1.x `no_data_threshold` parameter on `mosaic()` is gone. The whole-pipeline early-stop heuristic it controlled is replaced by the per-tile early stop described above (driven by `min_observations` for `mean`/`percentile`, by fill state for `first`); scene selection now examines every candidate scene that could still contribute. Callers passing `no_data_threshold=...` must drop the argument — the v1.x default `0.01` had no exact equivalent, but in practice the per-tile stop is both stricter and faster.


### Fixed
- Streaming-pipeline robustness fixes around partial scene failures, mask/scene alignment, and propagation of non-fetch errors out of the grid pipeline (so programming bugs surface as real tracebacks instead of being swallowed as "All scenes failed to fetch masks").
- Bounds/AOI visual mosaics now treat all-zero multi-band source reads as source nodata during tile aggregation. This prevents one-pixel black strips at some Sentinel-2 overlap edges where the SCL/footprint mask can mark a pixel valid but the warped TCI read falls just outside the real source data. The fix also applies when `max_observations` is set, so zero edge pixels are not counted toward the per-pixel observation cap.
- Bounds/AOI per-scene mask reads now use `WarpedVRT` for both same-CRS and cross-CRS sources so out-of-source pixels return nodata consistently with the band reader. The previous same-CRS fast path (`src.read(window, out_shape, boundless=True)`) returned in-data values for output pixels whose centres fell west of the source extent when the target grid origin was fractionally misaligned to the source pixel grid — making SCL/OCM masks claim "valid" for pixels the visual band correctly reported as nodata. With `max_observations` set, this mismatch could exhaust the per-pixel observation budget on zero-data scenes and starve later valid scenes, leaving 1-pixel dark vertical stripes at MGRS column boundaries in wide-area mosaics.


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
- Custom `scene_sort_fn` option for user-defined scene ordering.

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
