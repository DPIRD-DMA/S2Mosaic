from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import Polygon

from s2mosaic._types import MaskFetch
from s2mosaic.config import MosaicRequest
from s2mosaic.geometry import (
    _expand_bounds_for_ocm_context,
    _expand_window_for_ocm_context,
    _rasterize_aoi_mask,
)
from s2mosaic.pipelines.bounds import _mask_resolution_for_request
from s2mosaic.pipelines.bounds_scl import (
    _pick_overview_level,
    _read_band_at_target_window,
)
from s2mosaic.sources import MPC


def run_bounds_for_test(bounds_mod, **kwargs):
    source = kwargs.pop("source", MPC)
    request = MosaicRequest(**kwargs).normalized()
    request.validate()
    return bounds_mod.run_bounds_pipeline(request, source=source)


def _fake_scl_fetch_full_window(item, source, bt, tc, mr, scene_window):
    """All-ones SCL fetch returned at the requested scene_window — shared
    test stub for the bounds pipeline's per-scene-window fetch contract."""
    return MaskFetch(
        arr=np.ones((scene_window[3], scene_window[2]), dtype=np.uint8),
        target_window=scene_window,
        crop=(slice(0, scene_window[3]), slice(0, scene_window[2])),
    )


class TestBoundsSclHelpers:
    def test_pick_overview_level_sorts_overview_factors(self):
        assert _pick_overview_level(10, 80, [16, 2, 4, 8]) == 3


class TestReadBandAtTargetWindow:
    """Boundary behaviour at the source-COG extent.

    Regression: a previous same-CRS fast path used
    ``src.read(window, out_shape=..., boundless=True, fill_value=0)``. When
    the requested window extended west of the source extent, ``out_shape``
    downsampling did not honour the boundless padding for the leftmost
    output pixels — they came back with valid in-data values instead of 0.
    That made SCL/OCM masks claim "valid" for out-of-source pixels while
    the WarpedVRT-based band reader correctly returned nodata, producing
    1-pixel dark stripes at MGRS overlap-zone edges in the final mosaic.
    """

    @staticmethod
    def _write_source(tmp_path, *, src_minx, src_maxy, native_res, size, fill):
        """Write a tiny single-band COG-like GeoTIFF filled with ``fill``."""
        import rasterio as rio
        from rasterio.transform import Affine

        path = tmp_path / "src.tif"
        transform = Affine(native_res, 0, src_minx, 0, -native_res, src_maxy)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": size,
            "height": size,
            "crs": "EPSG:32750",
            "transform": transform,
        }
        with rio.open(path, "w", **profile) as dst:
            dst.write(np.full((size, size), fill, dtype=np.uint8), 1)
        return path

    def test_pixels_west_of_source_return_zero(self, tmp_path):
        """Output pixel whose centre is west of source must be 0.

        Uses a *fractional* source col_off — the failure mode in production
        only appears when the target grid origin is misaligned to the source
        pixel grid. Integer-aligned offsets accidentally hide the bug.
        """
        from rasterio.crs import CRS

        from s2mosaic.helpers import get_rasterio_resampling

        # Source: UTM x ∈ [100_000, 102_000) at 20 m native.
        src_path = self._write_source(
            tmp_path,
            src_minx=100_000.0,
            src_maxy=200_000.0,
            native_res=20.0,
            size=100,
            fill=5,
        )

        # Target grid at 60 m starting at x = 99_953.36 so source col_off is
        # the fractional value (-2.332) that triggered the production bug.
        # Output pixel 0 covers x ∈ [99_953.36, 100_013.36) — centre 99_983.36
        # is *outside* source (west of 100_000); pixel 1 onward is inside.
        target_minx = 99_953.36
        target_res = 60.0
        target_width = 6
        target_height = 10
        read_bounds = (
            target_minx,
            200_000.0 - target_height * target_res,
            target_minx + target_width * target_res,
            200_000.0,
        )
        arr = _read_band_at_target_window(
            str(src_path),
            band_idx=1,
            read_bounds=read_bounds,
            target_crs_obj=CRS.from_epsg(32750),
            target_width=target_width,
            target_height=target_height,
            rio_resampling=get_rasterio_resampling("nearest"),
        )

        assert arr.shape == (target_height, target_width)
        # Leftmost output pixel's centre is west of source — must be nodata.
        np.testing.assert_array_equal(arr[:, 0], 0)
        # Inside columns must carry the source fill.
        np.testing.assert_array_equal(arr[:, 1:], 5)


class TestBoundsMaskResolution:
    def test_scl_mask_resolution_uses_native_floor(self):
        request = MosaicRequest(
            bounds=(0, 0, 100, 100), resolution=10, cloud_mask="SCL"
        )

        assert _mask_resolution_for_request(request) == 20

    def test_scl_mask_resolution_preserves_coarser_user_resolution(self):
        request = MosaicRequest(
            bounds=(0, 0, 100, 100), resolution=30, cloud_mask="SCL"
        )

        assert _mask_resolution_for_request(request) == 30

    def test_ocm_mask_resolution_still_uses_ocm_policy(self):
        request = MosaicRequest(
            bounds=(0, 0, 100, 100), resolution=60, cloud_mask="OCM"
        )

        assert _mask_resolution_for_request(request) == 50


class TestBoundsOcmContext:
    class FakeItem:
        id = "fake-scene"
        datetime = datetime(2023, 6, 1, tzinfo=timezone.utc)
        # Wide enough that the reprojected bbox covers any test bounds in
        # EPSG:32750 (whether near origin or in real UTM coords). Avoids the
        # extreme poles so reprojection stays well-defined.
        bbox = (-90.0, -45.0, 90.0, 45.0)
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
        import s2mosaic.pipelines.bounds as bounds_mod

        requested_bounds = (390_000.0, 6_460_000.0, 390_200.0, 6_460_100.0)
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
        monkeypatch.setattr(
            bounds_mod,
            "_scene_window_in_target",
            lambda item_bbox, bounds_target, target_crs, resolution: (0, 0, 10, 5),
        )

        def fake_fetch_one_ocm(
            item, source, bounds_target, target_crs, ocm_resolution, scene_window
        ):
            expanded, crop = _expand_window_for_ocm_context(
                bounds_target, ocm_resolution, scene_window
            )
            fetch_calls.append(
                (
                    item.id,
                    bounds_target,
                    scene_window,
                    expanded,
                    target_crs,
                    ocm_resolution,
                )
            )
            return MaskFetch(
                arr=np.zeros((3, expanded[3], expanded[2]), dtype=np.uint16),
                target_window=scene_window,
                crop=crop,
            )

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

        arr, profile = run_bounds_for_test(
            bounds_mod,
            bounds=requested_bounds,
            input_crs=32750,
            output_crs=32750,
            resolution=20,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="OCM",
        )

        assert len(fetch_calls) == 1
        call = fetch_calls[0]
        assert call[0] == "fake-scene"
        assert call[1] == requested_bounds  # bounds_target, not expanded
        assert call[2] == (0, 0, 10, 5)  # scene_window equals full bounds (small AOI)
        assert call[3] == (-45, -47, 100, 100)  # padded to OCM context
        assert call[4] == 32750
        assert call[5] == 20
        assert compute_input_shapes == [(3, 100, 100)]
        assert arr.shape == (1, 5, 10)
        assert profile["width"] == 10
        assert profile["height"] == 5

        mask = aggregation_calls[0]["masks"][0]
        np.testing.assert_array_equal(mask, expected_cropped_mask)
        assert aggregation_calls[0]["coverage_mask"].shape == (5, 10)

    def test_bounds_ocm_prefetch_is_capped_below_tile_workers(self, monkeypatch):
        import s2mosaic.pipelines.bounds as bounds_mod

        prefetch_workers = []
        aggregation_calls = []

        monkeypatch.setattr(
            bounds_mod,
            "_search_for_items_by_bbox",
            lambda **_: [self.FakeItem(), self.FakeItem(), self.FakeItem()],
        )
        monkeypatch.setattr(
            bounds_mod,
            "_scene_window_in_target",
            lambda item_bbox, bounds_target, target_crs, resolution: (0, 0, 100, 100),
        )

        def fake_iter_ordered_fetches(items, fetch_fn, max_workers, on_complete=None):
            prefetch_workers.append(max_workers)
            for i, _item in enumerate(items):
                if on_complete is not None:
                    on_complete(i)
                yield (
                    i,
                    MaskFetch(
                        arr=np.zeros((3, 100, 100), dtype=np.uint16),
                        target_window=(0, 0, 100, 100),
                        crop=(slice(0, 100), slice(0, 100)),
                    ),
                )

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

        run_bounds_for_test(
            bounds_mod,
            bounds=(0.0, 0.0, 3000.0, 3000.0),
            input_crs=32750,
            output_crs=32750,
            resolution=20,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="OCM",
            min_coverage_fraction=None,
            tile_workers=8,
        )

        assert prefetch_workers == [2]
        assert len(aggregation_calls[0]["masks"]) == 3

    def test_bounds_mask_stream_closes_iterator_on_unexpected_error(self, monkeypatch):
        import s2mosaic.pipelines.bounds as bounds_mod

        class FailingIterator:
            def __init__(self):
                self.closed = False
                self._used = False

            def __iter__(self):
                return self

            def __next__(self):
                if self._used:
                    raise StopIteration
                self._used = True
                return 0, ValueError("boom")

            def close(self):
                self.closed = True

        iterator = FailingIterator()

        monkeypatch.setattr(
            bounds_mod,
            "_scene_window_for_item",
            lambda item, bounds_target, target_crs, resolution: (0, 0, 1, 1),
        )
        monkeypatch.setattr(
            bounds_mod,
            "iter_ordered_fetches",
            lambda **_: iterator,
        )

        with pytest.raises(ValueError, match="boom"):
            bounds_mod._stream_bounds_combo_masks(
                items_list=[self.FakeItem()],
                source=MPC,
                bounds_target=(0.0, 0.0, 10.0, 10.0),
                target_crs=32750,
                mask_resolution=10,
                mask_w=1,
                mask_h=1,
                coverage_mask=np.ones((1, 1), dtype=bool),
                cloud_mask="SCL",
                mosaic_method="mean",
                tile_workers=1,
                ocm_batch_size=1,
                ocm_inference_dtype="bf16",
                scl_tile_specs=None,
                show_progress=False,
            )

        assert iterator.closed

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

    def test_aoi_scl_uses_adaptive_tile_windows_for_sparse_masks(self, monkeypatch):
        import s2mosaic.pipelines.bounds as bounds_mod

        aoi = Polygon(
            [
                (390_000.0, 6_460_000.0),
                (410_480.0, 6_460_000.0),
                (410_480.0, 6_480_480.0),
                (390_000.0, 6_480_480.0),
            ]
        )
        fetch_tile_specs = []

        def fake_rasterize_aoi_mask(**kwargs):
            mask = np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)
            mask[:64, :] = True
            return mask

        def fake_fetch_one_scl_tiled(
            item,
            source,
            bounds_target,
            target_crs,
            mask_resolution,
            width,
            height,
            tile_specs,
            scene_window,
        ):
            fetch_tile_specs.append(tile_specs)
            return MaskFetch(
                arr=np.ones((height, width), dtype=np.uint8),
                target_window=(0, 0, width, height),
                crop=(slice(0, height), slice(0, width)),
            )

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_aoi", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_rasterize_aoi_mask", fake_rasterize_aoi_mask)
        monkeypatch.setattr(
            bounds_mod, "_should_use_tiled_scl_fetch", lambda *_, **__: True
        )
        monkeypatch.setattr(
            bounds_mod, "_fetch_one_scl_tiled", fake_fetch_one_scl_tiled
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

        run_bounds_for_test(
            bounds_mod,
            aoi=aoi,
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            adaptive_tiling=True,
        )

        assert fetch_tile_specs
        assert fetch_tile_specs[0] != [(0, 0, 1024, 1024)]
        assert all(h <= 512 and w <= 512 for _, _, h, w in fetch_tile_specs[0])
        assert len(fetch_tile_specs[0]) == 2

    def test_aoi_scl_uses_full_read_when_block_gate_rejects_tiling(self, monkeypatch):
        import s2mosaic.pipelines.bounds as bounds_mod

        aoi = Polygon(
            [
                (390_000.0, 6_460_000.0),
                (410_480.0, 6_460_000.0),
                (410_480.0, 6_480_480.0),
                (390_000.0, 6_480_480.0),
            ]
        )
        full_fetches = []

        def fake_rasterize_aoi_mask(**kwargs):
            mask = np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)
            mask[:64, :] = True
            return mask

        def fake_fetch_one_scl(
            item, source, bounds_target, target_crs, mask_resolution, scene_window
        ):
            full_fetches.append(item.id)
            _, _, w, h = scene_window
            return MaskFetch(
                arr=np.ones((h, w), dtype=np.uint8),
                target_window=scene_window,
                crop=(slice(0, h), slice(0, w)),
            )

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_aoi", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_rasterize_aoi_mask", fake_rasterize_aoi_mask)
        monkeypatch.setattr(
            bounds_mod, "_should_use_tiled_scl_fetch", lambda *_, **__: False
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", fake_fetch_one_scl)
        monkeypatch.setattr(
            bounds_mod,
            "_fetch_one_scl_tiled",
            lambda *_, **__: pytest.fail("block gate should choose full SCL read"),
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

        run_bounds_for_test(
            bounds_mod,
            aoi=aoi,
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            adaptive_tiling=True,
        )

        assert full_fetches == ["fake-scene"]

    def test_aoi_pipeline_uses_polygon_search(self, monkeypatch):
        import s2mosaic.pipelines.bounds as bounds_mod

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
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        arr, profile = run_bounds_for_test(
            bounds_mod,
            aoi=aoi,
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
        )

        assert len(search_calls) == 1
        assert search_calls[0].geom_type == "Polygon"
        assert not search_calls[0].is_empty
        assert arr.shape == (1, 4, 4)
        assert profile["width"] == 4
        assert profile["height"] == 4

    def test_aoi_mask_is_applied_to_aggregation_inputs(self, monkeypatch):
        import s2mosaic.pipelines.bounds as bounds_mod

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

        def fake_rasterize_aoi_mask(**kwargs):
            if kwargs["height"] == 2:
                return np.ones((2, 2), dtype=bool)
            return aoi_mask.copy()

        monkeypatch.setattr(bounds_mod, "_rasterize_aoi_mask", fake_rasterize_aoi_mask)
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        run_bounds_for_test(
            bounds_mod,
            aoi=aoi,
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
        )

        assert len(aggregation_calls) == 1
        np.testing.assert_array_equal(aggregation_calls[0]["coverage_mask"], aoi_mask)
        np.testing.assert_array_equal(aggregation_calls[0]["masks"][0], aoi_mask)

    def test_bounds_pipeline_passes_show_progress_to_tile_aggregation(
        self, monkeypatch
    ):
        import s2mosaic.pipelines.bounds as bounds_mod

        aggregation_calls = []

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        run_bounds_for_test(
            bounds_mod,
            bounds=(0.0, 0.0, 40.0, 40.0),
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            tile_workers=2,
            show_progress=True,
        )

        assert aggregation_calls[0]["show_progress"] is True
        assert aggregation_calls[0]["tile_workers"] == 2

    def test_bounds_export_uses_streaming_geotiff_writer(self, monkeypatch, tmp_path):
        import s2mosaic.pipelines.bounds as bounds_mod

        writer_calls = []
        export_path = tmp_path / "bounds.tif"

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        result = run_bounds_for_test(
            bounds_mod,
            bounds=(0.0, 0.0, 40.0, 40.0),
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            output_path=export_path,
        )

        assert result == export_path
        assert len(writer_calls) == 1
        assert writer_calls[0]["export_path"] == export_path

    def test_bounds_with_crs_mismatch_uses_bbox_search_and_no_mask(self, monkeypatch):
        """``bounds=`` always fills the rectangle (or its reprojected envelope
        for cross-CRS); there is no implicit polygon mask. Verify cross-CRS
        bounds searches by bbox (not by polygon) and never invokes
        ``_rasterize_aoi_mask`` — that path is reserved for explicit ``aoi=``.
        """
        import s2mosaic.pipelines.bounds as bounds_mod

        rasterize_calls = []

        def fake_search_by_aoi(**_):
            raise AssertionError(
                "bounds= must search by bbox even when input_crs != output_crs"
            )

        def tracking_rasterize(**kwargs):
            rasterize_calls.append(kwargs)
            return np.ones((kwargs["height"], kwargs["width"]), dtype=bool)

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_search_for_items_by_aoi", fake_search_by_aoi)
        monkeypatch.setattr(bounds_mod, "_rasterize_aoi_mask", tracking_rasterize)
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        run_bounds_for_test(
            bounds_mod,
            bounds=(114.80, -32.35, 125.20, -31.75),
            input_crs=4326,
            output_crs=32751,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            resolution=160,
            adaptive_tiling=False,
        )

        assert rasterize_calls == []

    def test_bounds_with_matching_crs_keeps_bbox_search(self, monkeypatch):
        """When input_crs == output_crs there is no reprojection envelope to
        worry about, so bounds mode should keep its bbox-search behaviour and
        not synthesise a clip polygon.
        """
        import s2mosaic.pipelines.bounds as bounds_mod

        rasterize_called = []

        def fake_search_by_aoi(**_):
            raise AssertionError(
                "matched-CRS bounds mode must use bbox search, not polygon"
            )

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_search_for_items_by_aoi", fake_search_by_aoi)
        monkeypatch.setattr(
            bounds_mod,
            "_rasterize_aoi_mask",
            lambda **kwargs: (
                rasterize_called.append(kwargs)
                or np.ones((kwargs["height"], kwargs["width"]), dtype=bool)
            ),
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        run_bounds_for_test(
            bounds_mod,
            bounds=(0.0, 0.0, 40.0, 40.0),
            input_crs=32750,
            output_crs=32750,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
        )

        assert rasterize_called == []

    def test_bounds_mode_preserved_in_sidecar_for_cross_crs_input(
        self, monkeypatch, tmp_path
    ):
        """Cross-CRS bounds= must still be reported as ``mode: bounds`` in the
        sidecar — the user asked for bounds mode, regardless of any internal
        polygon handling.
        """
        import s2mosaic.pipelines.bounds as bounds_mod

        captured_mode = []

        original_metadata = bounds_mod.output_sidecar_metadata

        def tracking_metadata(request, **kwargs):
            captured_mode.append(kwargs["mode"])
            return original_metadata(request, **kwargs)

        monkeypatch.setattr(bounds_mod, "output_sidecar_metadata", tracking_metadata)
        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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
            "write_tile_aggregation_geotiff",
            lambda **kwargs: (
                kwargs["export_path"].write_bytes(b"fake") or kwargs["export_path"]
            ),
        )

        export_path = tmp_path / "wa.tif"
        run_bounds_for_test(
            bounds_mod,
            bounds=(114.80, -32.35, 125.20, -31.75),
            input_crs=4326,
            output_crs=32751,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            resolution=160,
            adaptive_tiling=False,
            output_path=export_path,
        )

        assert captured_mode == ["bounds"]

    def test_bounds_with_crs_mismatch_fills_envelope_no_implicit_clip(
        self, monkeypatch
    ):
        """``bounds=`` fills the rectangle (or its reprojected envelope for
        cross-CRS) with no implicit polygon clip. The coverage mask passed to
        aggregation must therefore be all-True across the whole envelope —
        envelope corners that sit outside the original lat/lon rectangle are
        still written with imagery (callers who want them clipped use
        ``aoi=shapely.box(*bounds)`` explicitly).
        """
        import s2mosaic.pipelines.bounds as bounds_mod

        aggregation_calls = []

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", lambda **_: [self.FakeItem()]
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        run_bounds_for_test(
            bounds_mod,
            bounds=(114.80, -32.35, 125.20, -31.75),
            input_crs=4326,
            output_crs=32751,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            resolution=160,
            adaptive_tiling=False,
        )

        assert len(aggregation_calls) == 1
        coverage_mask = np.asarray(aggregation_calls[0]["coverage_mask"])
        h, w = aggregation_calls[0]["height"], aggregation_calls[0]["width"]
        assert coverage_mask.shape == (h, w)
        # No implicit clip — every envelope pixel is in coverage.
        assert coverage_mask.all()

    def test_cross_crs_bounds_search_covers_target_envelope(self, monkeypatch):
        """The STAC search bbox must cover the *target-CRS output envelope*
        reprojected back to 4326, not just the user's original lat/lng box.

        Parallels and meridians curve in UTM, so the axis-aligned UTM envelope
        of a lat/lng rectangle extends a few km beyond the rectangle at the
        corners. A search keyed off the original lat/lng box misses scenes
        whose footprint only touches those corner pixels and leaves nodata
        wedges in the output. Regression for that bug.
        """
        import s2mosaic.pipelines.bounds as bounds_mod
        from s2mosaic.geometry import reproject_bbox

        captured_search_bboxes = []

        def capture_bbox_search(**kwargs):
            captured_search_bboxes.append(kwargs["bbox_4326"])
            return [self.FakeItem()]

        monkeypatch.setattr(
            bounds_mod, "_search_for_items_by_bbox", capture_bbox_search
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
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

        # Wide WA strip — same shape as the Advanced notebook AOI that
        # surfaced the bug.
        user_bounds = (114.80, -32.35, 120.20, -31.75)
        output_crs = 32750

        run_bounds_for_test(
            bounds_mod,
            bounds=user_bounds,
            input_crs=4326,
            output_crs=output_crs,
            start_year=2023,
            duration_days=1,
            bands=["B04"],
            cloud_mask="SCL",
            min_coverage_fraction=None,
            resolution=160,
            adaptive_tiling=False,
        )

        assert len(captured_search_bboxes) == 1
        search_bbox = captured_search_bboxes[0]

        # The search bbox must equal the UTM output envelope reprojected back
        # to 4326 — that is what guarantees every output pixel has a chance of
        # being filled by a returned scene.
        target_envelope = reproject_bbox(user_bounds, 4326, output_crs)
        expected_search_bbox = reproject_bbox(target_envelope, output_crs, 4326)
        assert search_bbox == pytest.approx(expected_search_bbox, abs=1e-6)

        # And it must strictly cover the original user bounds (at least one
        # edge expanded outward) — otherwise the fix has regressed and corner
        # pixels of the UTM envelope can again be missed by the search.
        assert search_bbox[0] <= user_bounds[0]
        assert search_bbox[1] <= user_bounds[1]
        assert search_bbox[2] >= user_bounds[2]
        assert search_bbox[3] >= user_bounds[3]
        assert search_bbox != pytest.approx(user_bounds, abs=1e-6)

    def test_non_overlapping_scenes_are_silently_skipped(self, monkeypatch, caplog):
        """Scenes returned by the (inflated) STAC search whose footprint
        doesn't actually overlap ``bounds_target`` must be silently skipped
        — they shouldn't count toward ``dropped_scenes`` or log at WARNING.

        The expanded search bbox brings in some scenes that touch the lat/lng
        envelope but not the target-CRS extent. That's expected, not a fetch
        failure, and shouldn't pollute the log or the dropped-scenes report.
        """
        import logging

        import s2mosaic.pipelines.bounds as bounds_mod

        items = [self.FakeItem(), self.FakeItem(), self.FakeItem()]
        scene_windows = [None, (0, 0, 4, 4), None]

        def fake_window_for_item(item, bounds_target, target_crs, resolution):
            return scene_windows.pop(0)

        def fake_iter_ordered_fetches(items, fetch_fn, max_workers, on_complete=None):
            for i, item in enumerate(items):
                try:
                    result = fetch_fn(i, item)
                except Exception as e:
                    result = e
                if on_complete is not None:
                    on_complete(i)
                yield (i, result)

        monkeypatch.setattr(bounds_mod, "_scene_window_for_item", fake_window_for_item)
        monkeypatch.setattr(
            bounds_mod, "iter_ordered_fetches", fake_iter_ordered_fetches
        )
        monkeypatch.setattr(bounds_mod, "_fetch_one_scl", _fake_scl_fetch_full_window)
        monkeypatch.setattr(
            bounds_mod,
            "compute_masks_from_scl",
            lambda scl: (
                np.ones_like(scl, dtype=bool),
                np.ones_like(scl, dtype=bool),
            ),
        )

        with caplog.at_level(logging.WARNING, logger="s2mosaic.pipelines.bounds"):
            kept, dropped = bounds_mod._stream_bounds_combo_masks(
                items_list=items,
                source=MPC,
                bounds_target=(0.0, 0.0, 40.0, 40.0),
                target_crs=32750,
                mask_resolution=10,
                mask_w=4,
                mask_h=4,
                coverage_mask=np.ones((4, 4), dtype=bool),
                cloud_mask="SCL",
                mosaic_method="mean",
                tile_workers=1,
                ocm_batch_size=1,
                ocm_inference_dtype="fp32",
                scl_tile_specs=None,
                show_progress=False,
            )

        # Only the middle scene contributed; the two no-overlap scenes are
        # silently dropped — not counted as failures, not warned about.
        assert list(kept.keys()) == [1]
        assert dropped == []
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []
