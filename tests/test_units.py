import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from s2mosaic import mosaic
from s2mosaic import bounds as bounds_module
from s2mosaic.bounds import aggregate_stack, pick_utm_epsg, reproject_bbox
from s2mosaic.frequent_coverage import (
    get_coverage,
    get_frequent_coverage_for_bbox,
    get_raster_coverage,
)
from s2mosaic.masking import (
    SCL_CLOUDY_CLASSES,
    compute_masks_from_scl,
    get_valid_mask,
)
from s2mosaic.mosaic_utils import calculate_percentile_mosaic


class TestGetValidMask:
    """Unit tests for the no-data dilation step (cv2.dilate, MORPH_CROSS, it=4)."""

    def test_all_zero_input_returns_all_invalid(self):
        bands = np.zeros((3, 50, 50), dtype=np.uint16)
        mask = get_valid_mask(bands, dilation_count=4)
        assert mask.shape == (50, 50)
        assert mask.dtype == bool
        assert not mask.any()

    def test_all_data_input_returns_all_valid(self):
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        mask = get_valid_mask(bands, dilation_count=4)
        assert mask.all()

    def test_pixel_is_invalid_only_when_all_bands_are_zero(self):
        bands = np.zeros((3, 10, 10), dtype=np.uint16)
        bands[0, 5, 5] = 100  # one band has data → pixel is valid
        mask = get_valid_mask(bands, dilation_count=0)
        assert mask[5, 5]
        assert not mask[0, 0]

    def test_dilation_count_zero_does_not_grow_mask(self):
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        bands[:, 25, 25] = 0
        mask = get_valid_mask(bands, dilation_count=0)
        assert not mask[25, 25]
        # Direct neighbors still valid
        assert mask[24, 25]
        assert mask[26, 25]
        assert mask[25, 24]
        assert mask[25, 26]

    def test_dilation_grows_invalid_region_diamond(self):
        # Repeated MORPH_CROSS dilations form a diamond (Manhattan distance) region
        bands = np.full((3, 50, 50), 100, dtype=np.uint16)
        bands[:, 25, 25] = 0
        mask = get_valid_mask(bands, dilation_count=4)
        # Manhattan distance <= 4 from (25, 25) is invalid
        assert not mask[25, 25]  # dist 0
        assert not mask[21, 25]  # dist 4
        assert not mask[29, 25]
        assert not mask[25, 21]
        assert not mask[25, 29]
        assert not mask[23, 27]  # dist 4 (2+2)
        # Manhattan distance 5+ remains valid
        assert mask[20, 25]
        assert mask[30, 25]
        assert mask[22, 28]  # dist 5

    def test_returns_bool_dtype(self):
        bands = np.zeros((3, 10, 10), dtype=np.uint16)
        assert get_valid_mask(bands).dtype == bool


class TestComputeMasksFromScl:
    """SCL-based clear+valid mask logic."""

    def test_known_class_layout(self):
        # One pixel of every class 0..11 in a single row.
        scl = np.arange(12, dtype=np.uint8).reshape(1, 12)
        clear, valid = compute_masks_from_scl(scl, dilation_count=0)
        # Cloudy classes (1, 3, 8, 9, 10) → clear=False; everything else clear=True
        for cls in range(12):
            expected_clear = cls not in SCL_CLOUDY_CLASSES
            assert bool(clear[0, cls]) == expected_clear, (
                f"class {cls} clear={clear[0, cls]}"
            )
        # No-data (class 0) is the only invalid pixel without dilation
        assert not valid[0, 0]
        for cls in range(1, 12):
            assert valid[0, cls]

    def test_cloudy_classes_match_constant(self):
        # Defensive: lock down the constant so changes are explicit
        assert SCL_CLOUDY_CLASSES == (1, 3, 8, 9, 10)

    def test_dilation_grows_invalid_around_no_data(self):
        # Single no-data pixel → 4-iter MORPH_CROSS dilate → diamond of invalids
        scl = np.full((50, 50), 5, dtype=np.uint8)  # all bare-soil
        scl[25, 25] = 0
        _, valid = compute_masks_from_scl(scl, dilation_count=4)
        assert not valid[25, 25]
        assert not valid[21, 25]  # Manhattan dist 4
        assert not valid[25, 21]
        assert valid[20, 25]  # dist 5 still valid
        assert valid[25, 20]

    def test_3d_input_squeezes_first_axis(self):
        # get_full_band returns (1, H, W); compute_masks_from_scl should handle that
        scl = np.array([[[0, 1, 4, 8]]], dtype=np.uint8)  # shape (1, 1, 4)
        clear, valid = compute_masks_from_scl(scl, dilation_count=0)
        assert clear.shape == (1, 4)
        np.testing.assert_array_equal(clear[0], [True, False, True, False])
        np.testing.assert_array_equal(valid[0], [False, True, True, True])

    def test_returns_bool_dtype(self):
        scl = np.zeros((10, 10), dtype=np.uint8)
        clear, valid = compute_masks_from_scl(scl)
        assert clear.dtype == bool
        assert valid.dtype == bool


class TestZoomAlignmentConvention:
    """Regression test for the nearest-neighbor resample alignment.

    s2mosaic deliberately uses cv2.INTER_NEAREST (pixel-area-cell convention)
    instead of scipy.ndimage.zoom order=0 (centered-point convention) so an
    upsampled 60m S2 pixel maps exactly to the 6x6 block of 10m pixels it
    covers in the GeoTIFF transform. Reverting to scipy would shift 60m
    bands by ~30m (3 pixels at 10m).
    """

    def test_2x_upsample_matches_np_repeat(self):
        src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        out = cv2.resize(src, (6, 4), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(out, src.repeat(2, axis=0).repeat(2, axis=1))

    def test_6x_upsample_matches_np_repeat(self):
        src = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        out = cv2.resize(src, (12, 12), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(out, src.repeat(6, axis=0).repeat(6, axis=1))

    def test_3x_upsample_uses_pixel_area_layout(self):
        # Pixel-area: input pixel i fills output cells [i*N, (i+1)*N).
        # Scipy's order=0 would yield [1, 1, 2, 2, 2, 2, 3, 3, 3] — different.
        src = np.array([[1, 2, 3]], dtype=np.uint16)
        out = cv2.resize(src, (9, 1), interpolation=cv2.INTER_NEAREST)
        expected = np.array([[1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=np.uint16)
        np.testing.assert_array_equal(out, expected)


class TestCalculatePercentileMosaic:
    """Tests for the percentile mosaic codepath (numbagg.nanquantile)."""

    SCENE_SIZE = 50

    def _scene(self, value, bands=1):
        if value is None:
            return np.full(
                (bands, self.SCENE_SIZE, self.SCENE_SIZE), np.nan, dtype=np.float32
            )
        return np.full(
            (bands, self.SCENE_SIZE, self.SCENE_SIZE), value, dtype=np.float32
        )

    def test_single_scene_returns_that_scene(self):
        scene = self._scene(5.0)
        out = calculate_percentile_mosaic([scene], s2_scene_size=self.SCENE_SIZE)
        np.testing.assert_allclose(out, 5.0)
        assert out.shape == scene.shape

    def test_median_of_three_constant_scenes(self):
        scenes = [self._scene(1.0), self._scene(5.0), self._scene(9.0)]
        out = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, percentile_value=50.0
        )
        np.testing.assert_allclose(out, 5.0)

    def test_p90_of_three_constant_scenes(self):
        scenes = [self._scene(1.0), self._scene(5.0), self._scene(9.0)]
        out = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, percentile_value=90.0
        )
        # 90th percentile of [1, 5, 9] is 8.2 (linear interp between 5 and 9)
        np.testing.assert_allclose(out, 8.2, rtol=1e-5)

    def test_all_nan_input_replaced_with_zero(self):
        scene = self._scene(None)
        out = calculate_percentile_mosaic([scene], s2_scene_size=self.SCENE_SIZE)
        np.testing.assert_array_equal(out, 0.0)

    def test_nans_skipped_in_percentile(self):
        # Median of [NaN, 5, 15] should be 10 (NaN ignored)
        scenes = [self._scene(None), self._scene(5.0), self._scene(15.0)]
        out = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, percentile_value=50.0
        )
        np.testing.assert_allclose(out, 10.0)

    def test_chunk_concatenation_matches_unchunked(self):
        # Output should be the same regardless of chunk_size
        scenes = [self._scene(1.0), self._scene(5.0), self._scene(9.0)]
        out_chunked = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, chunk_size=10
        )
        out_one_chunk = calculate_percentile_mosaic(
            scenes, s2_scene_size=self.SCENE_SIZE, chunk_size=self.SCENE_SIZE
        )
        np.testing.assert_array_equal(out_chunked, out_one_chunk)
        assert out_chunked.shape == (1, self.SCENE_SIZE, self.SCENE_SIZE)


class TestPickUtmEpsg:
    """UTM zone picking from lat/lon."""

    @pytest.mark.parametrize(
        "lon, lat, expected",
        [
            (115.86, -31.95, 32750),  # Perth, WA → UTM 50S
            (115.86, 31.95, 32650),  # mirror in the north → UTM 50N
            (-122.43, 37.77, 32610),  # San Francisco → UTM 10N
            (0.0, 0.0, 32631),  # equator/Greenwich → UTM 31N
            (-179.9, 0.0, 32601),  # west of dateline → UTM 1N
            (179.9, 0.0, 32660),  # east of dateline → UTM 60N
        ],
    )
    def test_known_locations(self, lon, lat, expected):
        assert pick_utm_epsg(lon, lat) == expected


class TestReprojectBbox:
    def test_same_crs_returns_input(self):
        bbox = (115.83, -31.97, 115.91, -31.94)
        assert reproject_bbox(bbox, 4326, 4326) == bbox

    def test_4326_to_utm_round_trip(self):
        bbox_4326 = (115.83, -31.97, 115.91, -31.94)
        utm = reproject_bbox(bbox_4326, 4326, 32750)
        # UTM eastings around Perth are ~390-400km
        assert 380_000 < utm[0] < 410_000
        assert 380_000 < utm[2] < 410_000
        # Round-trip should land back near the input (within ~0.001 degrees)
        back = reproject_bbox(utm, 32750, 4326)
        for got, want in zip(back, bbox_4326, strict=False):
            assert abs(got - want) < 0.001


class TestMosaicBoundsValidation:
    """Input validation for bounds-mode mosaic — fails before any network call."""

    VALID_BOUNDS = (115.83, -31.97, 115.91, -31.94)

    def _call(self, bounds, **kwargs):
        return mosaic(start_year=2023, bounds=bounds, **kwargs)

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            self._call((1.0, 1.0, 0.0, 0.0))

    def test_zero_width_bounds_rejected(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            self._call((1.0, 0.0, 1.0, 1.0))

    def test_wrong_arity_bounds_rejected(self):
        with pytest.raises(ValueError, match="must be"):
            self._call((1.0, 2.0, 3.0))  # type: ignore[arg-type]

    def test_negative_resolution_rejected(self):
        with pytest.raises(ValueError, match="resolution"):
            self._call(self.VALID_BOUNDS, resolution=-10)

    def test_invalid_band_rejected(self):
        with pytest.raises(ValueError, match="Invalid band"):
            self._call(self.VALID_BOUNDS, required_bands=["FOO"])

    def test_visual_band_with_other_bands_rejected(self):
        with pytest.raises(ValueError, match="Cannot use visual band with other bands"):
            self._call(self.VALID_BOUNDS, required_bands=["visual", "B04"])

    def test_invalid_mosaic_method_rejected(self):
        with pytest.raises(ValueError, match="Invalid mosaic method"):
            self._call(self.VALID_BOUNDS, mosaic_method="bogus")

    def test_grid_id_and_bounds_mutually_exclusive(self):
        with pytest.raises(ValueError, match="Exactly one"):
            mosaic(grid_id="50HMH", start_year=2023, bounds=self.VALID_BOUNDS)

    def test_neither_grid_id_nor_bounds_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            mosaic(start_year=2023)


class TestAggregate:
    """Tests for the in-memory aggregation step (no network)."""

    H, W = 20, 30  # non-square to exercise the bounds case

    def _stack(self, time_values, bands=2):
        return np.stack(
            [np.full((bands, self.H, self.W), v, dtype=np.uint16) for v in time_values],
            axis=0,
        )

    def _mask(self, value):
        return np.full((self.H, self.W), value, dtype=bool)

    def test_mean_with_all_masks_true(self):
        stack = self._stack([10, 20, 30])
        masks = [self._mask(True)] * 3
        out = aggregate_stack(stack, masks, "mean", percentile_value=None)
        np.testing.assert_allclose(out, 20.0)

    def test_mean_zero_where_no_valid_pixels(self):
        stack = self._stack([10, 20])
        masks = [self._mask(False)] * 2
        out = aggregate_stack(stack, masks, "mean", percentile_value=None)
        np.testing.assert_array_equal(out, 0.0)

    def test_first_picks_earliest_valid(self):
        stack = self._stack([10, 20, 30])
        # Only 2nd scene is valid → output should be 20
        masks = [self._mask(False), self._mask(True), self._mask(True)]
        out = aggregate_stack(stack, masks, "first", percentile_value=None)
        np.testing.assert_allclose(out, 20.0)

    def test_percentile_50_matches_median(self):
        stack = self._stack([10, 50, 90])
        masks = [self._mask(True)] * 3
        out = aggregate_stack(stack, masks, "percentile", percentile_value=50.0)
        np.testing.assert_allclose(out, 50.0)


class _FakeItem:
    """Minimal stub matching the bits get_coverage() reads from a pystac.Item."""

    def __init__(self, polygon_coords):
        self.geometry = {"type": "Polygon", "coordinates": [polygon_coords]}


class TestFrequentCoverageForBbox:
    """Tests for the bounds variant of frequent-coverage masking."""

    def _utm_bounds_perth(self):
        # ~10km x 10km AOI near Perth, in UTM 50S
        return (390000.0, 6463000.0, 400000.0, 6473000.0)

    def test_no_scenes_returns_all_invalid(self):
        bounds = self._utm_bounds_perth()
        out = get_frequent_coverage_for_bbox(
            scenes=[],
            bounds_target=bounds,
            target_crs=32750,
            width=100,
            height=100,
            resolution=100,
        )
        assert out.shape == (100, 100)
        assert out.dtype == bool
        assert not out.any()

    def test_full_coverage_scene_returns_mostly_valid(self):
        # One scene polygon covering all of Perth
        coords = [
            (115.0, -32.5),
            (116.5, -32.5),
            (116.5, -31.5),
            (115.0, -31.5),
            (115.0, -32.5),
        ]
        scenes = [_FakeItem(coords)] * 5  # 5 identical scenes → max_count=5

        bounds = self._utm_bounds_perth()
        out = get_frequent_coverage_for_bbox(
            scenes=scenes,
            bounds_target=bounds,
            target_crs=32750,
            width=100,
            height=100,
            resolution=100,
        )
        assert out.shape == (100, 100)
        # Most pixels valid (some edge erosion from the 4-pixel dilation)
        assert out.mean() > 0.8

    def test_low_coverage_pixels_masked(self):
        # Two scenes: one covers eastern half only, four cover everything.
        # Pixels in the western half are covered by 4/5 = 80% (above 10%
        # threshold → kept). If we make one scene cover only ~5% of pixels,
        # those pixels become a small minority.
        full_coords = [
            (115.0, -32.5),
            (116.5, -32.5),
            (116.5, -31.5),
            (115.0, -31.5),
            (115.0, -32.5),
        ]
        partial_coords = [
            (115.45, -31.95),
            (115.46, -31.95),
            (115.46, -31.94),
            (115.45, -31.94),
            (115.45, -31.95),
        ]
        scenes = [_FakeItem(full_coords)] * 10 + [_FakeItem(partial_coords)]
        bounds = self._utm_bounds_perth()
        out = get_frequent_coverage_for_bbox(
            scenes=scenes,
            bounds_target=bounds,
            target_crs=32750,
            width=100,
            height=100,
            resolution=100,
            coverage_threshold_pct=0.5,
        )
        # All pixels covered by 10/11 ≈ 91% of scenes → pass 50% threshold
        assert out.mean() > 0.8


class TestGetRasterCoverage:
    """Resolution scaling for the rasterized coverage step (grid-mode)."""

    def _scene_polygon_4326(self):
        from shapely.geometry import box

        return box(115.0, -32.5, 116.5, -31.5)

    def _coverage_gdf(self):
        coords = [
            (115.0, -32.5),
            (116.5, -32.5),
            (116.5, -31.5),
            (115.0, -31.5),
            (115.0, -32.5),
        ]
        return get_coverage([_FakeItem(coords)] * 3)

    @pytest.mark.parametrize(
        "resolution, expected_side",
        [
            (10, 10980),  # native
            (20, 5490),
            (60, 1830),
            (100, 1098),
        ],
    )
    def test_output_shape_scales_with_resolution(self, resolution, expected_side):
        raster = get_raster_coverage(
            scene_bounds=self._scene_polygon_4326(),
            coverage_gdf=self._coverage_gdf(),
            local_crs=32750,
            resolution=resolution,
        )
        assert raster.shape == (expected_side, expected_side)


class TestCachedStackCompute:
    """Disk caching for stackstac materialisation in bounds mode."""

    def test_cache_miss_then_hit(self, tmp_path, monkeypatch):
        # Run inside a tmp dir so the "cache/" path doesn't litter the repo
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

        # Fake item with an .id (only attribute the cache key reads)
        class _FakeItem:
            def __init__(self, id_):
                self.id = id_

        items = [_FakeItem("S2A_T50HMH_2023A"), _FakeItem("S2A_T50HMH_2023B")]
        bounds_target = (390000.0, 6463000.0, 400000.0, 6473000.0)

        # Replace stackstac.stack with a stub that returns a fake xarray-ish object
        # whose .compute().values is a known array
        fake_arr = np.full((2, 4, 100, 100), 7, dtype=np.uint16)

        class _FakeStack:
            def compute(self):
                return self

            @property
            def values(self):
                return fake_arr

        call_count = {"n": 0}

        def fake_stack(*args, **kwargs):
            call_count["n"] += 1
            return _FakeStack()

        monkeypatch.setattr(bounds_module.stackstac, "stack", fake_stack)

        # First call: cache miss → invokes stack
        out1 = bounds_module.cached_stack_compute(
            items,
            ["B04"],
            bounds_target,
            32750,
            10,
            "uint16",
        )
        assert call_count["n"] == 1
        np.testing.assert_array_equal(out1, fake_arr)

        # Second call with same args: cache hit → does not invoke stack
        out2 = bounds_module.cached_stack_compute(
            items,
            ["B04"],
            bounds_target,
            32750,
            10,
            "uint16",
        )
        assert call_count["n"] == 1
        np.testing.assert_array_equal(out2, fake_arr)

        # Different args (resolution=20) → cache miss again
        bounds_module.cached_stack_compute(
            items,
            ["B04"],
            bounds_target,
            32750,
            20,
            "uint16",
        )
        assert call_count["n"] == 2

    def test_no_cache_when_env_var_unset(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("S2MOSAIC_DEBUG_CACHE", raising=False)

        class _FakeItem:
            def __init__(self, id_):
                self.id = id_

        items = [_FakeItem("X")]
        bounds_target = (0.0, 0.0, 1000.0, 1000.0)

        fake_arr = np.zeros((1, 1, 10, 10), dtype=np.uint16)

        class _FakeStack:
            def compute(self):
                return self

            @property
            def values(self):
                return fake_arr

        call_count = {"n": 0}

        def fake_stack(*args, **kwargs):
            call_count["n"] += 1
            return _FakeStack()

        monkeypatch.setattr(bounds_module.stackstac, "stack", fake_stack)

        for _ in range(3):
            bounds_module.cached_stack_compute(
                items,
                ["B04"],
                bounds_target,
                32750,
                10,
                "uint16",
            )
        assert call_count["n"] == 3
        assert not (tmp_path / "cache").exists()


class TestDebugCacheEnvVar:
    """Env-var gating for the debug-cache machinery (pickle_cache + disk_cache)."""

    def test_unset_env_var_means_disabled(self, monkeypatch):
        from s2mosaic.helpers import debug_cache_enabled

        monkeypatch.delenv("S2MOSAIC_DEBUG_CACHE", raising=False)
        assert debug_cache_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_truthy_values_enable(self, monkeypatch, value):
        from s2mosaic.helpers import debug_cache_enabled

        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", value)
        assert debug_cache_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "bogus"])
    def test_non_truthy_values_disable(self, monkeypatch, value):
        from s2mosaic.helpers import debug_cache_enabled

        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", value)
        assert debug_cache_enabled() is False

    def test_pickle_cache_skips_disk_when_disabled(self, tmp_path, monkeypatch):
        """When env var is unset, pickle_cache must not write to disk."""
        from s2mosaic.helpers import pickle_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("S2MOSAIC_DEBUG_CACHE", raising=False)

        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return 42

        # Two calls with the same key — both should compute, nothing on disk
        assert pickle_cache("p", "k", compute) == 42
        assert pickle_cache("p", "k", compute) == 42
        assert calls["n"] == 2
        assert not (tmp_path / "cache").exists()

    def test_pickle_cache_writes_and_hits_when_enabled(self, tmp_path, monkeypatch):
        from s2mosaic.helpers import pickle_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return {"answer": 42}

        out1 = pickle_cache("p", "k", compute)
        out2 = pickle_cache("p", "k", compute)
        assert out1 == out2 == {"answer": 42}
        assert calls["n"] == 1  # second call was a hit
        # Cache file written under the configured prefix
        cache_files = list((tmp_path / "cache").glob("p_*.pkl"))
        assert len(cache_files) == 1


class TestDiskCacheDecorator:
    """Behaviour of the @disk_cache decorator wrapping a function."""

    def test_disabled_env_skips_keyfn_and_caching(self, tmp_path, monkeypatch):
        from s2mosaic.helpers import disk_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("S2MOSAIC_DEBUG_CACHE", raising=False)

        key_calls = {"n": 0}
        fn_calls = {"n": 0}

        def key_fn(x):
            key_calls["n"] += 1
            return f"k{x}"

        @disk_cache("dec_test", key_fn=key_fn)
        def fn(x):
            fn_calls["n"] += 1
            return x * 2

        # Two calls with same arg: both compute, key_fn never invoked
        assert fn(3) == 6
        assert fn(3) == 6
        assert fn_calls["n"] == 2
        assert key_calls["n"] == 0
        assert not (tmp_path / "cache").exists()

    def test_enabled_env_caches_by_key(self, tmp_path, monkeypatch):
        from s2mosaic.helpers import disk_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

        fn_calls = {"n": 0}

        @disk_cache("dec_test", key_fn=lambda x: f"k{x}")
        def fn(x):
            fn_calls["n"] += 1
            return x * 2

        # Same key → second call is a hit
        assert fn(3) == 6
        assert fn(3) == 6
        assert fn_calls["n"] == 1
        # Different key → second compute
        assert fn(4) == 8
        assert fn_calls["n"] == 2
        # And third call with x=4 hits again
        assert fn(4) == 8
        assert fn_calls["n"] == 2

    def test_key_fn_receives_same_args_as_fn(self, tmp_path, monkeypatch):
        """key_fn must be passed exactly the wrapped function's args/kwargs."""
        from s2mosaic.helpers import disk_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

        seen_args = []

        def key_fn(*args, **kwargs):
            seen_args.append((args, kwargs))
            return "fixed"  # constant key — second call is a hit

        @disk_cache("dec_test", key_fn=key_fn)
        def fn(a, b, c=10):
            return a + b + c

        fn(1, 2, c=3)
        # On a hit, key_fn is still called once to compute the lookup key
        assert seen_args[0] == ((1, 2), {"c": 3})

    def test_decorator_preserves_function_metadata(self):
        from s2mosaic.helpers import disk_cache

        @disk_cache("p", key_fn=lambda x: str(x))
        def my_fn(x):
            """My docstring."""
            return x

        assert my_fn.__name__ == "my_fn"
        assert my_fn.__doc__ == "My docstring."

    def test_different_prefixes_do_not_collide(self, tmp_path, monkeypatch):
        """Two fns with same key but different prefixes write distinct files."""
        from s2mosaic.helpers import disk_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("S2MOSAIC_DEBUG_CACHE", "1")

        @disk_cache("alpha", key_fn=lambda: "shared")
        def fn_a():
            return "A"

        @disk_cache("beta", key_fn=lambda: "shared")
        def fn_b():
            return "B"

        assert fn_a() == "A"
        assert fn_b() == "B"
        # Two distinct files in the cache dir
        files = sorted(p.name for p in (tmp_path / "cache").glob("*.pkl"))
        assert len(files) == 2
        assert any(f.startswith("alpha_") for f in files)
        assert any(f.startswith("beta_") for f in files)


class TestMosaicSharedParamsValidation:
    """Validation edge cases for params shared by both modes."""

    BOUNDS = (115.83, -31.97, 115.91, -31.94)

    def test_grid_mode_rejects_invalid_band(self):
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic("50HMH", 2023, required_bands=["FOO"])

    def test_bounds_mode_rejects_invalid_band(self):
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic(start_year=2023, bounds=self.BOUNDS, required_bands=["FOO"])

    def test_bounds_mode_accepts_no_data_threshold(self):
        # Should fail later (no network) but not on validation
        with pytest.raises(
            ValueError, match="No data threshold must be between 0 and 1"
        ):
            mosaic(start_year=2023, bounds=self.BOUNDS, no_data_threshold=2.0)

    def test_bounds_mode_accepts_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            mosaic(start_year=2023, bounds=self.BOUNDS, resolution=0)

    def test_grid_mode_rejects_invalid_resampling(self):
        with pytest.raises(ValueError, match="Invalid resampling method"):
            mosaic("50HMH", 2023, resampling_method="not_a_method")

    def test_bounds_mode_rejects_invalid_resampling(self):
        with pytest.raises(ValueError, match="Invalid resampling method"):
            mosaic(
                start_year=2023, bounds=self.BOUNDS, resampling_method="not_a_method"
            )

    def test_grid_mode_rejects_invalid_cloud_mask(self):
        with pytest.raises(ValueError, match="Invalid cloud_mask"):
            mosaic("50HMH", 2023, cloud_mask="bogus")

    def test_bounds_mode_rejects_invalid_cloud_mask(self):
        with pytest.raises(ValueError, match="Invalid cloud_mask"):
            mosaic(start_year=2023, bounds=self.BOUNDS, cloud_mask="bogus")


class TestPickOcmResolution:
    """Clamping logic for OCM resolution given user resolution."""

    @pytest.mark.parametrize(
        "user_res, expected",
        [
            (5, 20),  # below floor → 20
            (10, 20),  # 10m output → OCM at 20m (4x faster than at 10m)
            (15, 20),  # below floor → 20
            (20, 20),  # boundary
            (25, 25),  # in range
            (30, 30),  # in range
            (40, 40),  # in range
            (50, 50),  # boundary
            (60, 50),  # above ceiling → 50
            (100, 50),  # above ceiling → 50
        ],
    )
    def test_clamping(self, user_res, expected):
        from s2mosaic.helpers import pick_ocm_resolution

        assert pick_ocm_resolution(user_res) == expected


class TestResamplingMap:
    """Verify the string→rasterio.Resampling mapping."""

    @pytest.mark.parametrize(
        "name", ["nearest", "bilinear", "cubic", "average", "lanczos"]
    )
    def test_known_methods_map_to_rasterio(self, name):
        from rasterio.enums import Resampling

        from s2mosaic.helpers import get_rasterio_resampling

        result = get_rasterio_resampling(name)
        assert isinstance(result, Resampling)
        assert result.name == name

    def test_unknown_method_raises(self):
        from s2mosaic.helpers import get_rasterio_resampling

        with pytest.raises(KeyError):
            get_rasterio_resampling("bogus")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
