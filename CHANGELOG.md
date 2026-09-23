# Changelog

All notable changes to this project will be documented in this file.


## [2.0.0b4] - 2026-09-22

### Changed
- **`percentile` and `median` run on a new kernel, `_quantile_axis0_u16`.** The old one held each tile as `float32` with NaN marking invalid observations, and sorted every pixel with an insertion sort. Both choices cost throughput. An insertion sort branches on data, so it ran one pixel at a time, and the NaN test sat in the innermost loop. The replacement keeps the stack as `uint16` and carries validity in a separate `bool` plane. It sorts with the same pruned selection network the medoid uses, generalised here from the median to any quantile. Hoisting the comparator loop above a block of 128 pixels puts pixels in the innermost loop, which compiles to vector min/max. Timed on its own against fully-valid 2048px tiles, the kernel is 4.6-9.2x faster. A whole `median` tile gains less, roughly 1.2-4x, because the per-scene reads and bookkeeping outside the kernel then dominate. Dropping `float32` also halves the largest working buffer: a 34-scene 4-band 2048px percentile tile peaks at ~1.5 GB, so ~12 GB across the default 8 tile workers.
- **`percentile` and `median` output can differ by 1 from 2.0.0b3.** The new kernel is bit-identical to `np.nanquantile`'s default linear interpolation, pinned by `TestQuantileAxis0U16._assert_matches_numpy`. The old one was not. It computed the read position and the blend in `float32`, and always anchored the interpolation at the lower of the two bracketing values, where NumPy switches to the upper anchor once the fraction reaches 0.5. Only pixels whose percentile falls between two observations can move, since one landing exactly on an observation involves no interpolation. `_finalise_tile` truncates toward zero on the cast back to `uint16`, so a value sitting near an integer boundary can fall either side of it. Across random stacks of 2 to 40 scenes at 25% cloud, 0.7% of output pixels shift, always by exactly 1, and the new value is the one matching an exact float64 quantile in 93% of those cases.
- **The medoid's median pass is vectorised the same way.** Building the per-band median target was 85-94% of `_medoid_axis0_u16`'s runtime, rather than the distance scoring the complexity notation describes. It was an insertion sort per pixel per band. It now runs through the selection network across a block of pixels. Measured on 6 bands and a 224x224 tile, single-threaded: 2.78x at 5 scenes, 4.50x at 20, 5.72x at 60. The gain survives cloud, roughly 3x at 25% cover and 1.5x at 75%. Output is unchanged.

### Fixed
- **Single-band requests now filter NODATA.** `_source_valid_from_bands` returned `None` for a one-band read, so those mosaics applied no filtering at all and wrote scene-edge `0` out as if it were a reading. With one band, "every requested band is zero" and "this band is zero" are the same test, so the existing all-bands-zero rule now applies unchanged.
- **`min_observations` and `max_observations` now count observations that survive the band read.** `_contributing_scene_indices` counts from cloud masks alone, so it can truncate the scene list before `_source_valid_from_bands` has had a say. The tile functions already compensated by taking every scene and driving the early stop from the post-filter count, but that handover was gated on `bands_count > 1`. One-band requests fell through to the mask-driven count, so a scene that turned out to be all-zero still consumed the target: `min_observations=2` could return a single observation, and `min_observations=1` could return an empty pixel while later scenes held data. The gate is gone; the band count never made a difference to whether a scene survives the read.
- **Deduplication keys on the granule, not on the item's timestamp, so a scene is neither duplicated nor dropped.** `filter_latest_processing_baselines` grouped items by `datetime` truncated to the second plus the tile, and the `datetime` property does not identify an acquisition. The two providers stamp it differently and each broke the grouping in an opposite direction. Element 84 stores the granule sensing time, which the Collection-1 reprocessing restamps: the 2019-03-23 acquisition of 50HMH is `02:31:32` under baseline 02.11 and `02:27:06` under 05.00, so the function whose job is to keep one baseline kept both and the same scene entered the stack twice. Microsoft stores the *datatake* start, shared by every granule of that datatake, so where one tile is covered by two granules they collided; with their baselines tied, `max` discarded one by STAC response order alone, and on 2019-08-17 that meant choosing between a granule at 2.2% nodata and one at 83.5% on a coin flip.

  The key is now the datastrip sensing start parsed from `s2:datastrip_id` (`..._S20191123T022659_N02.12`), plus the tile. Only the generation time and the baseline move when ESA reprocesses, so the sensing start names the granule itself, and both providers publish it and agree on it. Measured over six tiles across four years, the change recovers 56 granules MPC was dropping and collapses 85 duplicates AWS was keeping; the providers then agree to within 0.2% on how many distinct granules exist, against 5% before. Items lacking a parseable `s2:datastrip_id` fall back to the old key with a warning -- none of the 6,652 items sampled lacked one, so that path is defensive.

  Mosaics change where this bit. A `mean`, `median` or `percentile` over an affected date was weighting a duplicated scene twice on AWS; on MPC it was missing half the observations available for that tile-date, and `min_observations` counted the survivors only.
- **Searches no longer emit `DoesNotConformTo: Server does not conform to QUERY` on every call.** Microsoft Planetary Computer's landing page declares four conformance classes and omits `item-search#query`, so pystac_client warned on every search this package makes -- a warning users could neither act on nor silence without suppressing unrelated ones. MPC honours the extension regardless, verified against both things `query` carries: an `s2:mgrs_tile` filter returns only the requested tile, and `eo:cloud_cover` `lt` 20 returns nothing above 19.5. `Source.open_catalog` now asserts `QUERY` on the client, which records what was checked rather than hiding a category of warning, and is a no-op on Element 84, which declares it.

  The newer CQL2 Filter extension is not an alternative here. Neither provider declares `item-search#filter`, so moving to it would warn on both instead of one, and Element 84 answers a CQL2 search with HTTP 200 and an *unfiltered* item list -- a grid mosaic would silently draw scenes from the wrong MGRS tiles, caught only by the client-side backstop discarding nearly everything. The new slow test `TestServerSideQueryFiltering` pins that the server, not the backstop, is doing the filtering, for both providers.
- **`AOT` and `WVP` were unusable with `source="AWS"`.** Both pass `validate_inputs`, but neither had an entry in the Element 84 asset map, and `asset_name` returns the canonical name unchanged for anything unmapped. Earth Search publishes the two as `aot` and `wvp`, so the request reached `item.assets["AOT"]` and raised `KeyError` rather than anything a caller could act on. MPC was unaffected, because it names its assets the way s2mosaic does. `test_aws_maps_every_requestable_band` now checks the whole of `VALID_BANDS` against the map, so a band added later cannot repeat this quietly.
- `tile_workers` was documented as defaulting to `min(4, os.cpu_count() or 1)` in both `mosaic()` and the README for some time after it became 8. `tests/test_docs.py` now checks that the README lists every `mosaic()` parameter and that every default it quotes is the one a caller actually gets, including those filled in downstream from a `None` sentinel.
- The medoid documentation credited the gee-community libraries for the "closest to per-band median" formulation. Both generations of geetools implement the strict Flood 2013 medoid instead. The kernel implements LandTrendr's `medoidMosaic`, described in Kennedy et al. 2018 (doi:10.3390/rs10050691), which the docstring and README now cite. Documentation only.
- The publish workflow ran three actions on the Node 20 runtime, which GitHub has deprecated and now force-runs on Node 24: `upload-artifact` v4 to v7, `download-artifact` v4 to v8, and `action-gh-release` v2 to v3. Only the first was annotated on the v2.0.0b3 build, because the other two sit behind the approval gate and had not executed yet.
- The percentile kernel's equivalence test compared against `np.nanquantile` on a float32 stack, which made the reference depend on the NumPy version rather than on the kernel. NumPy computed that interpolation in float64 through 1.26 and switched to float32 in 2.0, where it lands an ulp off for any quantile falling between two observations. `_quantile_axis0_u16` does the position and blend in float64 and casts once at the end, so the reference drifted and the kernel's correct output began failing on every CI leg while still passing against the older NumPy pinned locally. `_assert_matches_numpy` now builds the reference as float64, which is the quantile the kernel actually targets; across 60 random stacks at seven quantiles it agrees exactly on both 1.26.4 and 2.5.3, where the float32 reference disagreed 84 times on 2.5.3. Tests only, and the kernel is unchanged: no released version ever carried the float32 reference.
- `mypy --strict` failed against numba 0.66. Whether the `untyped-decorator` ignore on `@njit` is needed depends on the installed numba, and 0.63 requires it while 0.66 makes it redundant, which mypy reports as an error in its own right. The `@njit` ignores now list `unused-ignore` alongside it, matching the `no-any-return` ignores elsewhere in the package. Both numpy 1.26.4 with numba 0.63 and numpy 2.4.6 with numba 0.66 pass. This is the breakage the uncapped dependencies make more likely, and the weekly CI run added in 2.0.0b3 is what should catch the next one.
- The README's Python blocks are now formatted with ruff, so the documented calls match the style the package is linted to.

### Notes
- **Isolated single-band zeros in Microsoft Planetary Computer's product are a known limitation.** At roughly 0.01-0.04% of valid pixels, in dark water, a single band reads `0` against neighbours near DN 1000 in the same band. 0 is Sentinel-2 L2A's NODATA and never a measurable reflectance -- since baseline 04.00 the `BOA_ADD_OFFSET` of -1000 puts zero reflectance at DN 1000, so DN 0 decodes to -0.1 -- so that band is wrong by about 1000 DN where it lands, which distorts band ratios and spectral indices at those pixels. Element 84's copy of the same acquisitions is unaffected, so `source="AWS"` avoids it entirely.

  This release briefly carried a stricter rule that rejected any pixel with a zero in *any* band, and it was reverted before release because it made the same pixels worse. It was also far broader than the 0.01-0.04% those measurements suggested: that figure came from spectral bands and was never checked against `visual`. TCI is a quantised 8-bit render, so near-zero reflectance rounds a channel to 0 -- over the Perth AOI used in `examples/`, 13679 of 143375 valid pixels (9.5%) carry a zero channel, almost all of it dark water where blue rounds down. Rejecting those would have blanked `visual` mosaics over water. Scene selection happens in phase 1 from cloud masks alone, before a single band is read: `first` narrows each scene's mask by the coverage tracker and stops fetching once coverage is claimed, and `_contributing_scene_indices` truncates the scene list against observation targets. These artifacts sit below a 20m SCL cell and below the OCM path's band-sum test, so no mask can see them. Rejecting a pixel at read time therefore stranded it, with no other scene left to try, and `first` wrote `0` across every band rather than the one that dropped out -- three bands destroyed to avoid one on a three-band request, and eleven on a twelve-band one.

  One limit is inherent to TCI rather than to any rule: ESA reserves 0 for NODATA, but quantisation drops genuinely black targets there too, so a `[0, 0, 0]` TCI pixel over very dark water cannot be told apart from an unobserved one and is dropped. That is 329 of 143375 valid pixels (0.23%) on the same AOI, inside water that renders near black regardless. Requests needing dark water should use the spectral bands, where DN 0 really is NODATA.

  Source validity is deliberately held to the same question the mask providers ask -- `get_valid_mask` marks no-data as `bands.sum(axis=0) == 0`, `compute_masks_from_scl` marks `SCL == NO_DATA` -- so that a pixel phase 1 calls usable is one phase 3 can fill. `TestSourceValidityMatchesMaskValidity` pins that correspondence. Catching sub-mask-cell artifacts needs the validity signal to reach scene selection, not a stricter test downstream of it.

## [2.0.0b3] - 2026-07-30

### Changed
- **`opencv-python` replaced by `opencv-python-headless`, and the `multiclean` floor raised to `>=0.4.0`.** multiclean 0.4.0 moved to the headless build; while s2mosaic still declared the non-headless one, a clean install pulled *both* distributions. Each installs a `cv2/` package into site-packages and they overwrite each other's files (37 shared paths, including `cv2/__init__.py`), so whichever was unpacked last silently won — `import cv2` could resolve to a different major than the resolver was asked for. s2mosaic only uses `cv2.dilate` and `cv2.resize`, so the headless build is sufficient. Raising the multiclean floor matters as much as the switch: multiclean 0.3.x still requires the non-headless build and would recreate the same collision from the other side. If your own code relied on s2mosaic to pull full `opencv-python` for GUI functions (`cv2.imshow` and friends), depend on `opencv-python` explicitly.
- **Dependency upper bounds removed** from `geopandas`, `numpy`, `opencv-python`, `pandas`, `pyproj`, `pystac`, and `shapely`. These came from a blanket "cap at the next major" sweep rather than any observed incompatibility, and two had since turned into hard install failures: `pandas<3` (pandas 3.0.0 shipped 2026-01-21) and `opencv-python<5` (5.0.0 shipped 2026-07-02) made `pip install s2mosaic` fail outright for anyone already holding those majors. An upper bound in a library propagates into every downstream resolution, so a stale one can make an otherwise-valid environment unsolvable. The `numpy` cap was additionally redundant — numba always pins numpy tighter (0.66 requires `numpy<2.5`), so numba, not this list, has always set the real ceiling.
- `rasterio>=1.3,<2` is kept deliberately, and is now the only upper bound. It is the deepest integration in the codebase (`WarpedVRT`, windowed reads, profile plumbing) and the place a silent geometry or resampling change would hide; it has also not bumped major since 2018, so the cap costs nothing in practice.

### Added
- Weekly scheduled CI run (Sundays 18:00 UTC) plus manual `workflow_dispatch`. With upper bounds gone, an upstream release can break the build without any commit to this repo, and `uv sync` resolves fresh on every run — so the scheduled job is what surfaces an incompatible dependency before users install into it.

### Fixed
- `test_streaming.py`'s parallel-fetch test no longer relies on a `sleep` to make two concurrent fetches overlap. It uses a `threading.Barrier`, so the "both fetches were in flight simultaneously" assertion is structural rather than timing-dependent and cannot flake on a loaded CI runner.
- The first cell of `examples/Mosaic method comparison.ipynb` was missing the `id` field that nbformat 4.5 requires. It only warned so far, but nbformat has announced this will become a hard error — which would have broken the notebook smoke-test step in CI.


## [2.0.0b2] - 2026-05-29

### Fixed
- Bounds/AOI per-scene mask reads no longer pass `OVERVIEW_LEVEL` through Rasterio `WarpedVRT.warp_extras`. Newer Rasterio/GDAL builds (for example Colab's Rasterio 1.5.0 with GDAL 3.12.1) reject that as an unsupported warp option and log `CPLE_NotSupported ... WARP_EXTRAS`; GDAL's default warp overview selection is used instead.


## [2.0.0b1] - 2026-05-29

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
- Cross-CRS `bounds=` no longer leaves nodata wedges at the corners of the output. The STAC scene search now uses the *target-CRS output envelope reprojected back to 4326* instead of the original lat/lng rectangle. Parallels and meridians curve in UTM, so the axis-aligned UTM envelope of a lat/lng box extends a few km beyond the box at the corners; a search keyed off the original lat/lng box missed scenes whose footprint only touched those corner pixels, leaving the output with empty wedges that looked like source-data gaps. The filename hash and sidecar metadata still use the original user-supplied bounds, so the same input keeps producing the same identifying hash.
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
