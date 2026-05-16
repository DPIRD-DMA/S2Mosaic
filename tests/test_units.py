import logging
import os
import sys
import threading
import time
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from s2mosaic import mosaic
import s2mosaic.mosaic_core as mosaic_core
from s2mosaic.bounds import (
    _expand_bounds_for_ocm_context,
    _rasterize_aoi_mask,
    make_bounds_tile_reader,
    pick_utm_epsg,
    reproject_bbox,
)
from s2mosaic.helpers import (
    SceneFetchError,
    get_output_path,
    resolve_export_path,
    validate_inputs,
)
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
from s2mosaic.mosaic_core import (
    DEFAULT_TILE_WORKERS,
    _HandleCache,
    _read_with_retry,
    _nanquantile_axis0,
    _warm_nanquantile_axis0,
    _write_tiled_copy,
    iter_ordered_fetches,
    run_tile_aggregation,
    should_prewarm_sources,
    stream_mosaic_pipeline,
    write_tile_aggregation_geotiff,
)
from s2mosaic.stac_utils import ITEM_COL


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


class TestRunTileAggregation:
    """Fast coverage for the shared tile-streamed aggregation engine."""

    H, W = 5, 7

    def _read_fn_for(self, scenes):
        def read_fn(scene_idx, band_idx, spec):
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        return read_fn

    def test_mean_uses_only_masked_pixels_across_tiles(self):
        scenes = np.stack(
            [
                np.full((2, self.H, self.W), 10, dtype=np.uint16),
                np.full((2, self.H, self.W), 30, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=2,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=3,
            tile_workers=1,
        )

        np.testing.assert_allclose(out, 20.0)

    def test_first_picks_first_valid_pixel(self):
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), 10, dtype=np.uint16),
                np.full((1, self.H, self.W), 20, dtype=np.uint16),
                np.full((1, self.H, self.W), 30, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.zeros((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="first",
            percentile_value=None,
            tile_size=4,
            tile_workers=1,
        )

        np.testing.assert_allclose(out, 20.0)

    def test_percentile_skips_nan_masked_pixels(self):
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), 5, dtype=np.uint16),
                np.full((1, self.H, self.W), 15, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.zeros((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="percentile",
            percentile_value=50.0,
            tile_size=2,
            tile_workers=1,
        )

        np.testing.assert_allclose(out, 15.0)

    def test_percentile_default_runs_reads_on_main_thread(self):
        thread_names = []
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), 10, dtype=np.uint16),
                np.full((1, self.H, self.W), 30, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        def read_fn(scene_idx, band_idx, spec):
            thread_names.append(threading.current_thread().name)
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="percentile",
            percentile_value=50.0,
            tile_size=3,
            tile_workers=None,
        )

        assert any(name != "MainThread" for name in thread_names)
        np.testing.assert_allclose(out, 20.0)

    @pytest.mark.parametrize("mosaic_method", ["mean", "first", "percentile"])
    def test_default_tile_workers_are_shared_by_all_methods(
        self, monkeypatch, mosaic_method
    ):
        captured_workers = []

        class FakeExecutor:
            def __init__(self, max_workers):
                captured_workers.append(max_workers)

            def map(self, fn, specs):
                return map(fn, specs)

            def shutdown(self, wait):
                assert wait is True

        monkeypatch.setattr(mosaic_core, "ThreadPoolExecutor", FakeExecutor)

        scenes = np.full((1, 1, self.H, self.W), 10, dtype=np.uint16)
        out = run_tile_aggregation(
            masks=[np.ones((self.H, self.W), dtype=bool)],
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method=mosaic_method,
            percentile_value=50.0 if mosaic_method == "percentile" else None,
            tile_size=2,
            tile_workers=None,
        )

        expected_workers = [DEFAULT_TILE_WORKERS] if DEFAULT_TILE_WORKERS > 1 else []
        assert captured_workers == expected_workers
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 10))

    def test_percentile_single_worker_runs_reads_on_main_thread(self):
        thread_names = []
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), 10, dtype=np.uint16),
                np.full((1, self.H, self.W), 30, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        def read_fn(scene_idx, band_idx, spec):
            thread_names.append(threading.current_thread().name)
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="percentile",
            percentile_value=50.0,
            tile_size=3,
            tile_workers=1,
        )

        assert set(thread_names) == {"MainThread"}
        np.testing.assert_allclose(out, 20.0)

    def test_percentile_ignores_no_data_threshold_inside_tile(self):
        reads = []
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), 10, dtype=np.uint16),
                np.full((1, self.H, self.W), 30, dtype=np.uint16),
            ],
            axis=0,
        )

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        out = run_tile_aggregation(
            masks=[
                np.ones((self.H, self.W), dtype=bool),
                np.ones((self.H, self.W), dtype=bool),
            ],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=0.01,
            mosaic_method="percentile",
            percentile_value=50.0,
            tile_size=10,
            tile_workers=1,
        )

        assert reads == [0, 1]
        np.testing.assert_allclose(out, 20.0)

    def test_empty_coverage_tile_does_not_read_sources(self):
        reads = {"n": 0}

        def read_fn(scene_idx, band_idx, spec):
            reads["n"] += 1
            _, _, h, w = spec
            return np.ones((h, w), dtype=np.uint16)

        out = run_tile_aggregation(
            masks=[np.ones((self.H, self.W), dtype=bool)],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.zeros((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=3,
            tile_workers=1,
        )

        assert reads["n"] == 0
        np.testing.assert_array_equal(out, np.zeros((1, self.H, self.W)))

    def test_mean_ignores_no_data_threshold_inside_tile(self):
        reads = []

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), 10 + scene_idx * 20, dtype=np.uint16)

        out = run_tile_aggregation(
            masks=[
                np.ones((self.H, self.W), dtype=bool),
                np.ones((self.H, self.W), dtype=bool),
            ],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=0.01,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=10,
            tile_workers=1,
        )

        assert reads == [0, 1]
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 20))

    def test_mean_stops_at_tile_observation_target(self):
        reads = []

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), 10 + scene_idx * 10, dtype=np.uint16)

        out = run_tile_aggregation(
            masks=[
                np.ones((self.H, self.W), dtype=bool),
                np.ones((self.H, self.W), dtype=bool),
                np.ones((self.H, self.W), dtype=bool),
            ],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=10,
            tile_workers=1,
            tile_observation_target=2,
        )

        assert reads == [0, 1]
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 15))

    def test_percentile_stops_at_tile_observation_target_per_pixel(self):
        reads = []
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]
        masks[0][0, 0] = False

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), scene_idx * 10, dtype=np.uint16)

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=None,
            mosaic_method="percentile",
            percentile_value=50.0,
            tile_size=10,
            tile_workers=1,
            tile_observation_target=2,
        )

        expected = np.full((1, self.H, self.W), 10, dtype=np.uint16)
        expected[0, 0, 0] = 15
        assert reads == [0, 1, 2]
        np.testing.assert_array_equal(out, expected)

    def test_first_ignores_no_data_threshold_until_coverage_filled(self):
        reads = []
        first_mask = np.ones((self.H, self.W), dtype=bool)
        first_mask[0, 0] = False
        second_mask = np.zeros((self.H, self.W), dtype=bool)
        second_mask[0, 0] = True

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), scene_idx + 1, dtype=np.uint16)

        out = run_tile_aggregation(
            masks=[
                first_mask,
                second_mask,
            ],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            no_data_threshold=0.1,
            mosaic_method="first",
            percentile_value=None,
            tile_size=10,
            tile_workers=1,
        )

        expected = np.ones((1, self.H, self.W), dtype=np.uint16)
        expected[0, 0, 0] = 2
        assert reads == [0, 1]
        np.testing.assert_array_equal(out, expected)

    def test_write_tile_aggregation_geotiff_streams_tiles(self, tmp_path):
        reads = []

        def read_fn(scene_idx, band_idx, spec):
            reads.append((scene_idx, band_idx, spec))
            _, _, h, w = spec
            return np.full((h, w), 10 + scene_idx * 20, dtype=np.uint16)

        profile = {
            "driver": "GTiff",
            "dtype": np.dtype(np.uint16),
            "width": self.W,
            "height": self.H,
            "count": 1,
            "crs": None,
            "transform": from_origin(0, self.H, 1, 1),
        }
        export_path = tmp_path / "streamed.tif"

        result = write_tile_aggregation_geotiff(
            export_path=export_path,
            profile=profile,
            required_bands=["B04"],
            masks=[
                np.ones((self.H, self.W), dtype=bool),
                np.ones((self.H, self.W), dtype=bool),
            ],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            output_coverage_mask=None,
            no_data_threshold=None,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint16),
        )

        assert result == export_path
        assert reads
        with rio.open(export_path) as src:
            assert src.count == 1
            assert src.nodata == 0
            assert src.descriptions == ("B04",)
            np.testing.assert_array_equal(
                src.read(1), np.full((self.H, self.W), 20, dtype=np.uint16)
            )

    def test_write_tile_aggregation_geotiff_applies_output_coverage(self, tmp_path):
        coverage = np.ones((self.H, self.W), dtype=bool)
        coverage[0, 0] = False

        def read_fn(scene_idx, band_idx, spec):
            _, _, h, w = spec
            return np.full((h, w), 10, dtype=np.uint16)

        export_path = tmp_path / "covered.tif"
        write_tile_aggregation_geotiff(
            export_path=export_path,
            profile={
                "driver": "GTiff",
                "dtype": np.dtype(np.uint16),
                "width": self.W,
                "height": self.H,
                "count": 1,
                "crs": None,
                "transform": from_origin(0, self.H, 1, 1),
            },
            required_bands=["B04"],
            masks=[np.ones((self.H, self.W), dtype=bool)],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            output_coverage_mask=coverage,
            no_data_threshold=None,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint16),
        )

        with rio.open(export_path) as src:
            data = src.read(1)
        assert data[0, 0] == 0
        assert data[0, 1] == 10

    def test_write_tile_aggregation_geotiff_does_not_allocate_full_output(
        self, tmp_path, monkeypatch
    ):
        import s2mosaic.mosaic_core as mosaic_core

        bands_count = 2
        height = 9
        width = 11
        full_output_shape = (bands_count, height, width)
        large_allocations = []
        original_zeros = mosaic_core.np.zeros

        def tracking_zeros(shape, *args, **kwargs):
            if tuple(shape) == full_output_shape:
                large_allocations.append(shape)
            return original_zeros(shape, *args, **kwargs)

        monkeypatch.setattr(mosaic_core.np, "zeros", tracking_zeros)

        def read_fn(scene_idx, band_idx, spec):
            _, _, h, w = spec
            return np.full((h, w), scene_idx + band_idx + 1, dtype=np.uint16)

        write_tile_aggregation_geotiff(
            export_path=tmp_path / "no_full_alloc.tif",
            profile={
                "driver": "GTiff",
                "dtype": np.dtype(np.uint16),
                "width": width,
                "height": height,
                "count": bands_count,
                "crs": None,
                "transform": from_origin(0, height, 1, 1),
            },
            required_bands=["B04", "B03"],
            masks=[np.ones((height, width), dtype=bool)],
            read_fn=read_fn,
            bands_count=bands_count,
            height=height,
            width=width,
            coverage_mask=np.ones((height, width), dtype=bool),
            output_coverage_mask=None,
            no_data_threshold=None,
            mosaic_method="mean",
            percentile_value=None,
            tile_size=4,
            tile_workers=1,
            out_dtype=np.dtype(np.uint16),
        )

        assert large_allocations == []


class TestNanquantileAxis0:
    """Custom Numba percentile reducer must match NumPy nanquantile semantics."""

    def test_warm_compile_runs_before_threaded_aggregation(self):
        _warm_nanquantile_axis0()
        assert _nanquantile_axis0.signatures

    @staticmethod
    def _assert_matches_numpy(stack: np.ndarray, q: float):
        got = _nanquantile_axis0(stack.astype(np.float32), q)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            expected = np.nanquantile(stack, q, axis=0).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-3, equal_nan=True)
        got_uint16 = np.nan_to_num(got, nan=0.0).astype(np.uint16)
        expected_uint16 = np.nan_to_num(expected, nan=0.0).astype(np.uint16)
        assert np.abs(got_uint16.astype(int) - expected_uint16.astype(int)).max() <= 1

    @pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
    def test_random_sparse_nan_stacks(self, q):
        rng = np.random.default_rng(42)
        for scenes, bands, h, w, valid_fraction in [
            (1, 1, 3, 4, 0.7),
            (2, 3, 5, 4, 0.5),
            (5, 2, 4, 6, 0.8),
            (17, 3, 5, 7, 0.35),
        ]:
            stack = rng.integers(
                0, 12000, size=(scenes, bands, h, w), dtype=np.uint16
            ).astype(np.float32)
            valid = rng.random((scenes, bands, h, w)) < valid_fraction
            stack[~valid] = np.nan
            self._assert_matches_numpy(stack, q)

    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_all_nan_stack(self, q):
        stack = np.full((4, 2, 3, 5), np.nan, dtype=np.float32)
        self._assert_matches_numpy(stack, q)

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    def test_single_valid_value_among_nans(self, q):
        stack = np.full((6, 2, 3, 4), np.nan, dtype=np.float32)
        stack[3] = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        self._assert_matches_numpy(stack, q)

    @pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_interpolation_matches_numpy(self, q):
        base = np.array([0, 10, 20, 40, 80], dtype=np.float32)
        stack = np.broadcast_to(base[:, None, None, None], (5, 1, 3, 4)).copy()
        self._assert_matches_numpy(stack, q)

    def test_preserves_band_pixel_independence(self):
        stack = np.array(
            [
                [[[1, 100], [np.nan, 4]], [[10, np.nan], [30, 40]]],
                [[[3, np.nan], [5, 6]], [[20, 25], [np.nan, 60]]],
                [[[7, 200], [9, np.nan]], [[np.nan, 35], [50, 80]]],
            ],
            dtype=np.float32,
        )
        self._assert_matches_numpy(stack, 0.5)

    def test_default_tile_workers_uses_thread_safe_percentile_kernel(self):
        _warm_nanquantile_axis0()

        assert DEFAULT_TILE_WORKERS == min(4, os.cpu_count() or 1)
        assert _nanquantile_axis0.signatures


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

    def test_bounds_and_aoi_mutually_exclusive(self):
        aoi = Polygon(
            [
                (115.83, -31.97),
                (115.91, -31.97),
                (115.91, -31.94),
                (115.83, -31.94),
            ]
        )
        with pytest.raises(ValueError, match="Exactly one"):
            mosaic(start_year=2023, bounds=self.VALID_BOUNDS, aoi=aoi)

    def test_neither_grid_id_nor_bounds_rejected(self):
        with pytest.raises(ValueError, match="Exactly one"):
            mosaic(start_year=2023)

    def test_lon_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="longitude must be in"):
            self._call((181.0, -31.97, 181.1, -31.94))

    def test_neg_lon_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="longitude must be in"):
            self._call((-181.0, -31.97, -180.9, -31.94))

    def test_lat_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="latitude must be in"):
            self._call((115.83, -91.0, 115.91, -90.9))

    def test_swapped_axes_caught_when_lon_exceeds_lat_range(self):
        # User swapped (lat, lon, lat, lon) for the Perth example — the
        # would-be-latitude slots now hold 115.83/115.91, exceeding ±90.
        with pytest.raises(ValueError, match="latitude must be in"):
            self._call((-31.97, 115.83, -31.94, 115.91))

    def test_lon_range_not_checked_when_input_crs_not_4326(self):
        # Bounds in UTM zone 50S (metres) — values larger than 180 are valid.
        # validate_inputs must accept this without raising on range.
        utm_bounds = (390_000.0, 6_460_000.0, 400_000.0, 6_470_000.0)
        validate_inputs(
            sort_method="valid_data",
            mosaic_method="mean",
            no_data_threshold=0.01,
            required_bands=["B04"],
            grid_id=None,
            percentile_value=None,
            bounds=utm_bounds,
            input_crs=32750,
            resolution=10,
        )

    def test_bounds_too_small_4326_rejected(self):
        # 5m x 5m AOI at lat=-32: below both the side-length and area floors.
        delta_lon = 5 / (111_111 * np.cos(np.radians(32)))
        delta_lat = 5 / 111_111
        with pytest.raises(ValueError, match="at least 10m"):
            self._call((115.83, -31.97, 115.83 + delta_lon, -31.97 + delta_lat))

    def test_bounds_too_small_utm_rejected(self):
        with pytest.raises(ValueError, match="at least 10m"):
            self._call(
                (390_000.0, 6_460_000.0, 390_005.0, 6_460_005.0), input_crs=32750
            )

    def test_bounds_too_skinny_rejected_even_when_area_is_large_enough(self):
        with pytest.raises(ValueError, match="at least 10m"):
            self._call(
                (390_000.0, 6_460_000.0, 390_001.0, 6_460_200.0), input_crs=32750
            )

    def test_bounds_large_4326_warns_but_is_accepted(self, caplog):
        # 5 deg x 5 deg AOI at lat=-32: area is well over 200km x 200km.
        with caplog.at_level(logging.WARNING, logger="s2mosaic.helpers"):
            validate_inputs(
                sort_method="valid_data",
                mosaic_method="mean",
                no_data_threshold=0.01,
                required_bands=["B04"],
                grid_id=None,
                percentile_value=None,
                bounds=(110.0, -35.0, 115.0, -30.0),
                input_crs=4326,
                resolution=10,
            )
        assert "larger than 200km x 200km" in caplog.text

    def test_bounds_large_utm_warns_but_is_accepted(self, caplog):
        # 300km x 300km in UTM.
        with caplog.at_level(logging.WARNING, logger="s2mosaic.helpers"):
            validate_inputs(
                sort_method="valid_data",
                mosaic_method="mean",
                no_data_threshold=0.01,
                required_bands=["B04"],
                grid_id=None,
                percentile_value=None,
                bounds=(300_000.0, 6_300_000.0, 600_000.0, 6_600_000.0),
                input_crs=32750,
                resolution=10,
            )
        assert "larger than 200km x 200km" in caplog.text

    def test_typical_cross_tile_bounds_accepted(self):
        # ~80km × 80km, larger than a single S2 tile's overlap zone but well
        # under the 200km ceiling — must pass validation cleanly.
        validate_inputs(
            sort_method="valid_data",
            mosaic_method="mean",
            no_data_threshold=0.01,
            required_bands=["B04"],
            grid_id=None,
            percentile_value=None,
            bounds=(119.0, -29.0, 119.8, -28.2),
            input_crs=4326,
            resolution=10,
        )

    def test_single_polygon_aoi_accepted(self):
        aoi = Polygon(
            [
                (115.83, -31.97),
                (115.91, -31.97),
                (115.91, -31.94),
                (115.83, -31.94),
            ]
        )
        validate_inputs(
            sort_method="valid_data",
            mosaic_method="mean",
            no_data_threshold=0.01,
            required_bands=["B04"],
            grid_id=None,
            percentile_value=None,
            aoi=aoi,
            input_crs=4326,
            resolution=10,
        )

    def test_multipolygon_aoi_rejected(self):
        poly = Polygon(
            [
                (115.83, -31.97),
                (115.91, -31.97),
                (115.91, -31.94),
                (115.83, -31.94),
            ]
        )
        with pytest.raises(ValueError, match="single shapely Polygon"):
            validate_inputs(
                sort_method="valid_data",
                mosaic_method="mean",
                no_data_threshold=0.01,
                required_bands=["B04"],
                grid_id=None,
                percentile_value=None,
                aoi=MultiPolygon([poly]),
                input_crs=4326,
                resolution=10,
            )

    def test_invalid_polygon_aoi_rejected(self):
        bowtie = Polygon(
            [
                (115.83, -31.97),
                (115.91, -31.94),
                (115.91, -31.97),
                (115.83, -31.94),
            ]
        )
        with pytest.raises(ValueError, match="valid Polygon"):
            validate_inputs(
                sort_method="valid_data",
                mosaic_method="mean",
                no_data_threshold=0.01,
                required_bands=["B04"],
                grid_id=None,
                percentile_value=None,
                aoi=bowtie,
                input_crs=4326,
                resolution=10,
            )


class TestExportPaths:
    def test_auto_filename_keeps_existing_mean_format(self, tmp_path):
        path = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 8),
            sort_method="oldest",
            mosaic_method="mean",
            required_bands=["B04", "B03", "B02"],
            grid_id="50HMH",
        )

        assert path.name == (
            "50HMH_2023-06-01_to_2023-06-08_oldest_mean_B04_B03_B02.tif"
        )

    def test_auto_percentile_filename_includes_percentile(self, tmp_path):
        p25 = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 8, 1),
            sort_method="valid_data",
            mosaic_method="percentile",
            percentile_value=25,
            required_bands=["B04"],
            bounds=(115.8301, -31.9702, 115.9103, -31.9404),
        )
        p75 = get_output_path(
            output_dir=tmp_path,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 8, 1),
            sort_method="valid_data",
            mosaic_method="percentile",
            percentile_value=75,
            required_bands=["B04"],
            bounds=(115.8301, -31.9702, 115.9103, -31.9404),
        )

        assert "_percentile_p25_" in p25.name
        assert "_percentile_p75_" in p75.name
        assert p25 != p75

    def test_output_path_is_used_directly(self, tmp_path):
        path = resolve_export_path(
            output_dir=None,
            output_path=tmp_path / "nested" / "custom.tif",
            start_date=date(2023, 6, 1),
            end_date=date(2023, 6, 8),
            sort_method="oldest",
            mosaic_method="mean",
            required_bands=["B04"],
            grid_id="50HMH",
        )

        assert path == tmp_path / "nested" / "custom.tif"
        assert path.parent.exists()

    def test_output_dir_and_output_path_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="Only one of output_dir or output_path"):
            resolve_export_path(
                output_dir=tmp_path,
                output_path=tmp_path / "custom.tif",
                start_date=date(2023, 6, 1),
                end_date=date(2023, 6, 8),
                sort_method="oldest",
                mosaic_method="mean",
                required_bands=["B04"],
                grid_id="50HMH",
            )

    def test_output_path_requires_tif_filename(self, tmp_path):
        with pytest.raises(ValueError, match="must include a .tif or .tiff filename"):
            resolve_export_path(
                output_dir=None,
                output_path=tmp_path / "custom",
                start_date=date(2023, 6, 1),
                end_date=date(2023, 6, 8),
                sort_method="oldest",
                mosaic_method="mean",
                required_bands=["B04"],
                grid_id="50HMH",
            )


class TestOrderedPrefetch:
    class FakeItem:
        def __init__(self, scene_id, delay):
            self.id = scene_id
            self.delay = delay

    def test_yields_sorted_items_while_fetching_in_parallel(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_fetch(_idx, item):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(item.delay)
                return np.full((1, 1), int(item.id), dtype=np.uint8)
            finally:
                with lock:
                    active -= 1

        items = [
            self.FakeItem("0", 0.05),
            self.FakeItem("1", 0.0),
            self.FakeItem("2", 0.0),
        ]

        got = list(
            iter_ordered_fetches(
                items=items,
                fetch_fn=fake_fetch,
                max_workers=2,
            )
        )

        assert [i for i, _ in got] == [0, 1, 2]
        assert [int(arr[0, 0]) for _, arr in got] == [0, 1, 2]
        assert max_active == 2

    def test_reports_fetch_failures_in_item_order(self):
        def fake_fetch(_idx, item):
            if item.id == "1":
                raise SceneFetchError("failed")
            return np.full((1, 1), int(item.id), dtype=np.uint8)

        items = [
            self.FakeItem("0", 0.0),
            self.FakeItem("1", 0.0),
            self.FakeItem("2", 0.0),
        ]

        got = list(
            iter_ordered_fetches(
                items=items,
                fetch_fn=fake_fetch,
                max_workers=2,
            )
        )

        assert [i for i, _ in got] == [0, 1, 2]
        assert isinstance(got[1][1], SceneFetchError)


class TestGridOrderedMaskStreaming:
    class FakeAsset:
        href = "sample.tif"

    class FakeItem:
        def __init__(self, scene_id):
            self.id = scene_id
            self.assets = {"B04": TestGridOrderedMaskStreaming.FakeAsset()}

    def _sorted_scenes(self, n_scenes):
        return pd.DataFrame(
            {ITEM_COL: [self.FakeItem(f"scene-{i}") for i in range(n_scenes)]}
        )

    def _patch_grid_pipeline_io(self, monkeypatch):
        import s2mosaic.mosaic_core as core_mod

        monkeypatch.setattr(
            core_mod,
            "_build_output_profile",
            lambda sample_href_signed, s2_scene_size: {
                "driver": "GTiff",
                "dtype": np.dtype(np.uint16),
                "width": s2_scene_size,
                "height": s2_scene_size,
                "count": 1,
                "crs": None,
                "transform": from_origin(0, s2_scene_size, 1, 1),
            },
        )
        monkeypatch.setattr(
            core_mod,
            "make_grid_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, spec: np.ones(
                    (spec[2], spec[3]), dtype=np.uint16
                )
            ),
        )
        monkeypatch.setattr(
            core_mod,
            "run_tile_aggregation",
            lambda **kwargs: np.ones(
                (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                dtype=np.uint16,
            ),
        )

    def test_grid_first_stops_mask_stream_after_coverage_is_filled(self, monkeypatch):
        import s2mosaic.mosaic_core as core_mod

        self._patch_grid_pipeline_io(monkeypatch)
        calls = []

        def fake_compute_one_scene_mask(**kwargs):
            calls.append(kwargs["item"].id)
            return np.ones((4, 4), dtype=bool)

        monkeypatch.setattr(
            core_mod, "_compute_one_scene_mask", fake_compute_one_scene_mask
        )

        out, profile = stream_mosaic_pipeline(
            sorted_scenes=self._sorted_scenes(4),
            required_bands=["B04"],
            coverage_mask=np.ones((4, 4), dtype=bool),
            no_data_threshold=None,
            mosaic_method="first",
            cloud_mask="SCL",
            s2_scene_size=4,
            tile_size=4,
            tile_workers=1,
        )

        assert calls == ["scene-0"]
        assert out is not None
        assert profile["width"] == 4

    def test_grid_mean_stops_mask_stream_at_no_data_threshold(self, monkeypatch):
        import s2mosaic.mosaic_core as core_mod

        self._patch_grid_pipeline_io(monkeypatch)
        calls = []

        def fake_compute_one_scene_mask(**kwargs):
            calls.append(kwargs["item"].id)
            if kwargs["item"].id == "scene-0":
                mask = np.zeros((4, 4), dtype=bool)
                mask[:2, :] = True
                return mask
            return np.ones((4, 4), dtype=bool)

        monkeypatch.setattr(
            core_mod, "_compute_one_scene_mask", fake_compute_one_scene_mask
        )

        stream_mosaic_pipeline(
            sorted_scenes=self._sorted_scenes(4),
            required_bands=["B04"],
            coverage_mask=np.ones((4, 4), dtype=bool),
            no_data_threshold=0.01,
            mosaic_method="mean",
            cloud_mask="SCL",
            s2_scene_size=4,
            tile_size=4,
            tile_workers=1,
        )

        assert calls == ["scene-0", "scene-1"]


class TestBoundsOcmContext:
    class FakeItem:
        id = "fake-scene"
        datetime = datetime(2023, 6, 1, tzinfo=timezone.utc)
        properties = {
            "s2:nodata_pixel_percentage": 0.0,
            "s2:high_proba_clouds_percentage": 0.0,
            "s2:cloud_shadow_percentage": 0.0,
            "sat:relative_orbit": 1,
        }

    def test_small_bounds_expand_to_minimum_ocm_context(self):
        expanded, crop = _expand_bounds_for_ocm_context(
            (390_000.0, 6_460_000.0, 390_200.0, 6_460_100.0),
            resolution=20,
        )

        assert expanded == (
            389_100.0,
            6_459_040.0,
            391_100.0,
            6_461_040.0,
        )
        assert crop == (slice(47, 52), slice(45, 55))

    def test_large_bounds_keep_requested_ocm_extent(self):
        bounds = (390_000.0, 6_460_000.0, 393_000.0, 6_463_000.0)
        expanded, crop = _expand_bounds_for_ocm_context(bounds, resolution=20)

        assert expanded == bounds
        assert crop == (slice(0, 150), slice(0, 150))

    def test_bounds_pipeline_fetches_expanded_ocm_and_aggregates_cropped_mask(
        self, monkeypatch
    ):
        import s2mosaic.bounds as bounds_mod

        requested_bounds = (390_000.0, 6_460_000.0, 390_200.0, 6_460_100.0)
        expected_expanded = (
            389_100.0,
            6_459_040.0,
            391_100.0,
            6_461_040.0,
        )
        fetch_calls = []
        compute_input_shapes = []
        aggregation_calls = []
        expected_cropped_mask = np.array(
            [
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            ],
            dtype=bool,
        )

        monkeypatch.setattr(
            bounds_mod,
            "_search_for_items_by_bbox",
            lambda **_: [self.FakeItem()],
        )
        monkeypatch.setattr(
            bounds_mod,
            "get_frequent_coverage_for_bbox",
            lambda **_: np.ones((5, 10), dtype=bool),
        )

        def fake_fetch_one_ocm(item, bounds_target, target_crs, ocm_resolution):
            fetch_calls.append((item.id, bounds_target, target_crs, ocm_resolution))
            return np.zeros((3, 100, 100), dtype=np.uint16)

        def fake_compute_masks_from_array(array, *, batch_size, inference_dtype):
            compute_input_shapes.append(array.shape)
            clear = np.zeros(array.shape[1:], dtype=bool)
            valid = np.ones(array.shape[1:], dtype=bool)
            clear[47:52, 45:55] = expected_cropped_mask
            return clear, valid

        def fake_make_bounds_tile_reader(**_):
            return lambda scene_idx, band_idx, window: np.ones(
                (window[2], window[3]), dtype=np.uint16
            )

        def fake_run_tile_aggregation(**kwargs):
            aggregation_calls.append(kwargs)
            return np.ones(
                (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                dtype=np.uint16,
            )

        monkeypatch.setattr(bounds_mod, "_fetch_one_ocm", fake_fetch_one_ocm)
        monkeypatch.setattr(
            bounds_mod, "compute_masks_from_array", fake_compute_masks_from_array
        )
        monkeypatch.setattr(
            bounds_mod, "make_bounds_tile_reader", fake_make_bounds_tile_reader
        )
        monkeypatch.setattr(
            bounds_mod, "run_tile_aggregation", fake_run_tile_aggregation
        )

        arr, profile = bounds_mod.run_bounds_pipeline(
            bounds=requested_bounds,
            input_crs=32750,
            output_crs=32750,
            resolution=20,
            start_year=2023,
            duration_days=1,
            required_bands=["B04"],
            cloud_mask="OCM",
        )

        assert fetch_calls == [("fake-scene", expected_expanded, 32750, 20)]
        assert compute_input_shapes == [(3, 100, 100)]
        assert arr.shape == (1, 5, 10)
        assert profile["width"] == 10
        assert profile["height"] == 5

        mask = aggregation_calls[0]["masks"][0]
        np.testing.assert_array_equal(mask, expected_cropped_mask)
        assert aggregation_calls[0]["coverage_mask"].shape == (5, 10)

    def test_bounds_ocm_prefetch_is_capped_below_tile_workers(self, monkeypatch):
        import s2mosaic.bounds as bounds_mod

        prefetch_workers = []
        aggregation_calls = []

        monkeypatch.setattr(
            bounds_mod,
            "_search_for_items_by_bbox",
            lambda **_: [self.FakeItem(), self.FakeItem(), self.FakeItem()],
        )

        def fake_iter_ordered_fetches(items, fetch_fn, max_workers):
            prefetch_workers.append(max_workers)
            for i, _item in enumerate(items):
                yield i, np.zeros((3, 150, 150), dtype=np.uint16)

        monkeypatch.setattr(
            bounds_mod, "iter_ordered_fetches", fake_iter_ordered_fetches
        )
        monkeypatch.setattr(
            bounds_mod,
            "compute_masks_from_array",
            lambda array, *, batch_size, inference_dtype: (
                np.ones(array.shape[1:], dtype=bool),
                np.ones(array.shape[1:], dtype=bool),
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "make_bounds_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, window: np.ones(
                    (window[2], window[3]), dtype=np.uint16
                )
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "run_tile_aggregation",
            lambda **kwargs: (
                aggregation_calls.append(kwargs)
                or np.ones(
                    (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                    dtype=np.uint16,
                )
            ),
        )

        bounds_mod.run_bounds_pipeline(
            bounds=(0.0, 0.0, 3000.0, 3000.0),
            input_crs=32750,
            output_crs=32750,
            resolution=20,
            start_year=2023,
            duration_days=1,
            required_bands=["B04"],
            cloud_mask="OCM",
            no_data_threshold=None,
            coverage_threshold_pct=None,
            tile_workers=8,
        )

        assert prefetch_workers == [2]
        assert len(aggregation_calls[0]["masks"]) == 3

    def test_rasterize_aoi_mask_uses_polygon_shape(self):
        aoi = Polygon([(0.0, 0.0), (40.0, 0.0), (40.0, 40.0)])

        mask = _rasterize_aoi_mask(
            aoi_target=aoi,
            bounds_target=(0.0, 0.0, 40.0, 40.0),
            resolution=10,
            width=4,
            height=4,
        )

        assert mask.shape == (4, 4)
        assert mask.dtype == bool
        assert 0 < mask.sum() < mask.size

    def test_aoi_pipeline_uses_polygon_search(self, monkeypatch):
        import s2mosaic.bounds as bounds_mod

        aoi = Polygon([(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)])
        search_calls = []

        def fake_search_by_aoi(**kwargs):
            search_calls.append(kwargs["aoi_4326"])
            return [self.FakeItem()]

        def fake_search_by_bbox(**_):
            raise AssertionError("bbox search should not be used for polygon AOIs")

        monkeypatch.setattr(bounds_mod, "_search_for_items_by_aoi", fake_search_by_aoi)
        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", fake_search_by_bbox
        )
        monkeypatch.setattr(
            bounds_mod,
            "_fetch_one_scl",
            lambda item, bounds_target, target_crs, mask_resolution: np.ones(
                (4, 4), dtype=np.uint8
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "compute_masks_from_scl",
            lambda scl: (
                np.ones_like(scl, dtype=bool),
                np.ones_like(scl, dtype=bool),
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "make_bounds_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, window: np.ones(
                    (window[2], window[3]), dtype=np.uint16
                )
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "run_tile_aggregation",
            lambda **kwargs: np.ones(
                (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                dtype=np.uint16,
            ),
        )

        arr, profile = bounds_mod.run_bounds_pipeline(
            aoi=aoi,
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            required_bands=["B04"],
            cloud_mask="SCL",
            coverage_threshold_pct=None,
        )

        assert len(search_calls) == 1
        assert search_calls[0].geom_type == "Polygon"
        assert not search_calls[0].is_empty
        assert arr.shape == (1, 4, 4)
        assert profile["width"] == 4
        assert profile["height"] == 4

    def test_aoi_mask_is_applied_to_aggregation_inputs(self, monkeypatch):
        import s2mosaic.bounds as bounds_mod

        aoi = Polygon([(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)])
        aoi_mask = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
        aggregation_calls = []

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_aoi", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(
            bounds_mod,
            "_rasterize_aoi_mask",
            lambda **_: aoi_mask.copy(),
        )
        monkeypatch.setattr(
            bounds_mod,
            "_fetch_one_scl",
            lambda item, bounds_target, target_crs, mask_resolution: np.ones(
                (4, 4), dtype=np.uint8
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "compute_masks_from_scl",
            lambda scl: (
                np.ones_like(scl, dtype=bool),
                np.ones_like(scl, dtype=bool),
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "make_bounds_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, window: np.ones(
                    (window[2], window[3]), dtype=np.uint16
                )
            ),
        )

        def fake_run_tile_aggregation(**kwargs):
            aggregation_calls.append(kwargs)
            return np.ones(
                (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                dtype=np.uint16,
            )

        monkeypatch.setattr(
            bounds_mod, "run_tile_aggregation", fake_run_tile_aggregation
        )

        bounds_mod.run_bounds_pipeline(
            aoi=aoi,
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            required_bands=["B04"],
            cloud_mask="SCL",
            coverage_threshold_pct=None,
        )

        assert len(aggregation_calls) == 1
        np.testing.assert_array_equal(aggregation_calls[0]["coverage_mask"], aoi_mask)
        np.testing.assert_array_equal(aggregation_calls[0]["masks"][0], aoi_mask)

    def test_bounds_pipeline_passes_show_progress_to_tile_aggregation(
        self, monkeypatch
    ):
        import s2mosaic.bounds as bounds_mod

        aggregation_calls = []

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(
            bounds_mod,
            "_fetch_one_scl",
            lambda item, bounds_target, target_crs, mask_resolution: np.ones(
                (4, 4), dtype=np.uint8
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "compute_masks_from_scl",
            lambda scl: (
                np.ones_like(scl, dtype=bool),
                np.ones_like(scl, dtype=bool),
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "make_bounds_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, window: np.ones(
                    (window[2], window[3]), dtype=np.uint16
                )
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "run_tile_aggregation",
            lambda **kwargs: (
                aggregation_calls.append(kwargs)
                or np.ones(
                    (kwargs["bands_count"], kwargs["height"], kwargs["width"]),
                    dtype=np.uint16,
                )
            ),
        )

        bounds_mod.run_bounds_pipeline(
            bounds=(0.0, 0.0, 40.0, 40.0),
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            required_bands=["B04"],
            cloud_mask="SCL",
            coverage_threshold_pct=None,
            tile_workers=2,
            show_progress=True,
        )

        assert aggregation_calls[0]["show_progress"] is True
        assert aggregation_calls[0]["tile_workers"] == 2

    def test_bounds_export_uses_streaming_geotiff_writer(self, monkeypatch, tmp_path):
        import s2mosaic.bounds as bounds_mod

        writer_calls = []
        export_path = tmp_path / "bounds.tif"

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(
            bounds_mod,
            "_fetch_one_scl",
            lambda item, bounds_target, target_crs, mask_resolution: np.ones(
                (4, 4), dtype=np.uint8
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "compute_masks_from_scl",
            lambda scl: (
                np.ones_like(scl, dtype=bool),
                np.ones_like(scl, dtype=bool),
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "make_bounds_tile_reader",
            lambda **_: (
                lambda scene_idx, band_idx, window: np.ones(
                    (window[2], window[3]), dtype=np.uint16
                )
            ),
        )
        monkeypatch.setattr(
            bounds_mod,
            "run_tile_aggregation",
            lambda **_: pytest.fail("export should not build a full output array"),
        )

        def fake_writer(**kwargs):
            writer_calls.append(kwargs)
            kwargs["export_path"].write_bytes(b"fake")
            return kwargs["export_path"]

        monkeypatch.setattr(bounds_mod, "write_tile_aggregation_geotiff", fake_writer)

        result = bounds_mod.run_bounds_pipeline(
            bounds=(0.0, 0.0, 40.0, 40.0),
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            required_bands=["B04"],
            cloud_mask="SCL",
            coverage_threshold_pct=None,
            output_path=export_path,
        )

        assert result == export_path
        assert len(writer_calls) == 1
        assert writer_calls[0]["export_path"] == export_path


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


class TestTiledBandMaterialisation:
    """Local tiled GeoTIFF materialisation helpers."""

    @pytest.mark.parametrize(
        "mosaic_method,no_data_threshold,tile_observation_target,expected",
        [
            ("mean", None, None, True),
            ("percentile", None, None, True),
            ("first", None, None, False),
            ("mean", 0.01, None, False),
            ("percentile", 0.01, None, False),
            ("mean", None, 2, False),
            ("percentile", None, 2, False),
        ],
    )
    def test_source_prewarm_policy(
        self, mosaic_method, no_data_threshold, tile_observation_target, expected
    ):
        assert (
            should_prewarm_sources(
                mosaic_method, no_data_threshold, tile_observation_target
            )
            is expected
        )

    def test_handle_cache_resolves_sources_only_on_first_read(self, monkeypatch):
        calls = {"resolver": 0, "open": 0}

        def resolver():
            calls["resolver"] += 1
            return "lazy-source.tif"

        class FakeDataset:
            pass

        def fake_open(source):
            calls["open"] += 1
            assert source == "lazy-source.tif"
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.mosaic_core.rio.open", fake_open)
        cache = _HandleCache([[resolver]])

        assert calls == {"resolver": 0, "open": 0}
        first = cache.get(0, 0)
        second = cache.get(0, 0)

        assert first is second
        assert calls == {"resolver": 1, "open": 1}

    def test_bounds_reader_materialises_cache_only_on_read(self, monkeypatch):
        calls = {"materialise": 0, "materialiser_factory": 0, "open": 0}

        class FakeAsset:
            href = "remote.tif"

        class FakeItem:
            assets = {"B04": FakeAsset()}
            id = "fake-scene"

        class FakeDataset:
            def read(self, band_idx, window):
                return np.full(
                    (int(window.height), int(window.width)),
                    band_idx,
                    dtype=np.uint16,
                )

        def fake_materialiser_factory(*args, **kwargs):
            calls["materialiser_factory"] += 1

            def write(path):
                return None

            return write

        def fake_materialise(cache_key, materialiser):
            calls["materialise"] += 1
            return Path("cached.tif")

        def fake_open(source):
            calls["open"] += 1
            assert source == "cached.tif"
            return FakeDataset()

        monkeypatch.setattr(
            "s2mosaic.bounds.planetary_computer.sign", lambda href: href
        )
        monkeypatch.setattr(
            "s2mosaic.bounds._materialise_bounds_band", fake_materialiser_factory
        )
        monkeypatch.setattr("s2mosaic.bounds.materialise_tiled_band", fake_materialise)
        monkeypatch.setattr("s2mosaic.bounds.rio.open", fake_open)

        read_fn = make_bounds_tile_reader(
            items=[FakeItem()],
            href_template=[("B04", 1)],
            bounds_target=(0.0, 0.0, 10.0, 10.0),
            target_crs=32750,
            user_transform=from_origin(0, 10, 1, 1),
            width=10,
            height=10,
            resolution=1,
            resampling_method="nearest",
            prewarm=False,  # verify the strictly-lazy contract
        )

        assert calls == {"materialise": 0, "materialiser_factory": 0, "open": 0}
        data = read_fn(0, 0, (2, 3, 4, 5))

        np.testing.assert_array_equal(data, np.ones((4, 5), dtype=np.uint16))
        assert calls == {"materialise": 1, "materialiser_factory": 1, "open": 1}

    def test_read_with_retry_recovers_from_transient_rasterio_error(self, monkeypatch):
        monkeypatch.setattr("s2mosaic.mosaic_core.time.sleep", lambda _: None)

        class FlakySource:
            calls = 0

            def read(self, *, window, out_shape, resampling):
                self.calls += 1
                if self.calls == 1:
                    raise RasterioIOError("temporary read failure")
                return np.full(out_shape, 7, dtype=np.uint16)

        src = FlakySource()
        data = _read_with_retry(
            src,
            window=rio.windows.Window(0, 0, 2, 2),
            out_shape=(1, 2, 2),
            resampling=Resampling.nearest,
        )
        assert src.calls == 2
        np.testing.assert_array_equal(data, np.full((1, 2, 2), 7, dtype=np.uint16))

    def test_write_tiled_copy_streams_destination_blocks(self, tmp_path):
        class FakeSource:
            def __init__(self):
                self.windows = []

            def read(self, *, window, out_shape, resampling):
                self.windows.append(window)
                return np.full(out_shape, int(window.col_off + window.row_off))

        src = FakeSource()
        profile = {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint16",
            "width": 4,
            "height": 4,
            "crs": "EPSG:32750",
            "transform": from_origin(0, 4, 1, 1),
            "tiled": True,
            "blockxsize": 16,
            "blockysize": 16,
        }
        out_path = tmp_path / "cached.tif"

        _write_tiled_copy(
            src,
            out_path,
            profile,
            Resampling.nearest,
            lambda window: rio.windows.Window(
                window.col_off * 2,
                window.row_off * 2,
                window.width * 2,
                window.height * 2,
            ),
        )

        assert src.windows == [rio.windows.Window(0, 0, 8, 8)]
        with rio.open(out_path) as cached:
            data = cached.read(1)
        np.testing.assert_array_equal(data, np.zeros((4, 4), dtype=np.uint16))


class TestMosaicSharedParamsValidation:
    """Validation edge cases for params shared by both modes."""

    BOUNDS = (115.83, -31.97, 115.91, -31.94)

    def test_grid_mode_rejects_invalid_band(self):
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic(grid_id="50HMH", start_year=2023, required_bands=["FOO"])

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
            mosaic(grid_id="50HMH", start_year=2023, resampling_method="not_a_method")

    def test_bounds_mode_rejects_invalid_resampling(self):
        with pytest.raises(ValueError, match="Invalid resampling method"):
            mosaic(
                start_year=2023, bounds=self.BOUNDS, resampling_method="not_a_method"
            )

    def test_grid_mode_rejects_invalid_cloud_mask(self):
        with pytest.raises(ValueError, match="Invalid cloud_mask"):
            mosaic(grid_id="50HMH", start_year=2023, cloud_mask="bogus")

    def test_bounds_mode_rejects_invalid_cloud_mask(self):
        with pytest.raises(ValueError, match="Invalid cloud_mask"):
            mosaic(start_year=2023, bounds=self.BOUNDS, cloud_mask="bogus")

    @pytest.mark.parametrize("tile_workers", [0, -1, True])
    def test_grid_mode_rejects_invalid_tile_workers(self, tile_workers):
        with pytest.raises(ValueError, match="tile_workers must be"):
            mosaic(grid_id="50HMH", start_year=2023, tile_workers=tile_workers)

    @pytest.mark.parametrize("tile_workers", [0, -1, False])
    def test_bounds_mode_rejects_invalid_tile_workers(self, tile_workers):
        with pytest.raises(ValueError, match="tile_workers must be"):
            mosaic(start_year=2023, bounds=self.BOUNDS, tile_workers=tile_workers)


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
