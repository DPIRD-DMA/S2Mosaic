import threading
import warnings

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

import s2mosaic.aggregation as aggregation_mod
from s2mosaic.aggregation import (
    DEFAULT_TILE_WORKERS,
    _nanquantile_axis0,
    _split_tile_size_aligned,
    _warm_nanquantile_axis0,
    adaptive_tile_specs_for_masks,
    iter_tile_aggregation,
    run_tile_aggregation,
    write_tile_aggregation_geotiff,
)


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
            mosaic_method="mean",
            percentile=None,
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
            mosaic_method="first",
            percentile=None,
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
            mosaic_method="percentile",
            percentile=50.0,
            tile_size=2,
            tile_workers=1,
        )

        np.testing.assert_allclose(out, 15.0)

    def test_adaptive_tile_specs_skip_empty_and_split_sparse_tiles(self):
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[0:32, 0:32] = True
        mask[1024:2048, 1024:2048] = True

        specs = adaptive_tile_specs_for_masks(
            masks=[mask],
            height=2048,
            width=2048,
            max_tile_size=2048,
            min_tile_size=512,
            dense_fraction=0.5,
        )

        assert (0, 0, 512, 512) in specs
        assert (1024, 1024, 1024, 1024) in specs
        assert all(r < 1536 or c < 1536 for r, c, _, _ in specs)
        assert (0, 512, 512, 512) not in specs

    @pytest.mark.parametrize(
        "length, expected",
        [
            (256, [256]),
            (512, [512]),
            (738, [512, 226]),
            (1024, [512, 512]),
            (1536, [1024, 512]),
            (2048, [1024, 1024]),
        ],
    )
    def test_adaptive_tile_split_prefers_512_aligned_boundaries(self, length, expected):
        assert _split_tile_size_aligned(length, 512) == expected

    def test_adaptive_tile_specs_use_aligned_splits_for_uneven_tiles(self):
        mask = np.zeros((738, 738), dtype=bool)
        mask[0:32, 0:32] = True

        specs = adaptive_tile_specs_for_masks(
            masks=[mask],
            height=738,
            width=738,
            max_tile_size=738,
            min_tile_size=512,
            dense_fraction=0.5,
        )

        assert specs == [(0, 0, 512, 512)]

    def test_fixed_tiling_opt_out_yields_empty_tiles(self):
        scenes = np.ones((1, 1, 4, 4), dtype=np.uint16)
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True

        def read_fn(scene_idx, band_idx, spec):
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        specs = [
            spec
            for spec, _ in iter_tile_aggregation(
                masks=[mask],
                read_fn=read_fn,
                bands_count=1,
                height=4,
                width=4,
                coverage_mask=np.ones((4, 4), dtype=bool),
                mosaic_method="mean",
                percentile=None,
                tile_size=2,
                tile_workers=1,
                adaptive_tiling=False,
            )
        ]

        assert specs == [
            (0, 0, 2, 2),
            (0, 2, 2, 2),
            (2, 0, 2, 2),
            (2, 2, 2, 2),
        ]

    def test_adaptive_tiling_skips_empty_tiles(self):
        scenes = np.ones((1, 1, 4, 4), dtype=np.uint16)
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True

        def read_fn(scene_idx, band_idx, spec):
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        specs = [
            spec
            for spec, _ in iter_tile_aggregation(
                masks=[mask],
                read_fn=read_fn,
                bands_count=1,
                height=4,
                width=4,
                coverage_mask=np.ones((4, 4), dtype=bool),
                mosaic_method="mean",
                percentile=None,
                tile_size=2,
                tile_workers=1,
                adaptive_tiling=True,
            )
        ]

        assert specs == [(0, 0, 2, 2)]

    def test_adaptive_tiling_preserves_output_values(self):
        reads = []
        scenes = np.ones((1, 1, 4, 4), dtype=np.uint16)
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True

        def read_fn(scene_idx, band_idx, spec):
            reads.append(spec)
            r, c, h, w = spec
            return scenes[scene_idx, band_idx, r : r + h, c : c + w]

        out = run_tile_aggregation(
            masks=[mask],
            read_fn=read_fn,
            bands_count=1,
            height=4,
            width=4,
            coverage_mask=np.ones((4, 4), dtype=bool),
            mosaic_method="mean",
            percentile=None,
            tile_size=2,
            tile_workers=1,
            adaptive_tiling=True,
        )

        np.testing.assert_array_equal(out[0, :2, :2], np.ones((2, 2)))
        np.testing.assert_array_equal(out[0, 2:, :], np.zeros((2, 4)))
        assert reads == [(0, 0, 2, 2)]

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
            mosaic_method="percentile",
            percentile=50.0,
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

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, fn, specs):
                return map(fn, specs)

        monkeypatch.setattr(aggregation_mod, "ThreadPoolExecutor", FakeExecutor)

        scenes = np.full((1, 1, self.H, self.W), 10, dtype=np.uint16)
        out = run_tile_aggregation(
            masks=[np.ones((self.H, self.W), dtype=bool)],
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method=mosaic_method,
            percentile=50.0 if mosaic_method == "percentile" else None,
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
            mosaic_method="percentile",
            percentile=50.0,
            tile_size=3,
            tile_workers=1,
        )

        assert set(thread_names) == {"MainThread"}
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
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        assert reads["n"] == 0
        np.testing.assert_array_equal(out, np.zeros((1, self.H, self.W)))

    @pytest.mark.parametrize("mosaic_method", ["mean", "first", "percentile"])
    def test_aggregation_ignores_mask_pixels_outside_coverage(self, mosaic_method):
        reads = []
        coverage = np.zeros((self.H, self.W), dtype=bool)
        coverage[0, 0] = True
        mask = ~coverage

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), 99, dtype=np.uint16)

        out = run_tile_aggregation(
            masks=[mask],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=coverage,
            mosaic_method=mosaic_method,
            percentile=50.0 if mosaic_method == "percentile" else None,
            tile_size=10,
            tile_workers=1,
        )

        assert reads == []
        np.testing.assert_array_equal(out, np.zeros((1, self.H, self.W)))

    def test_mean_stops_at_min_observations(self):
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
            mosaic_method="mean",
            percentile=None,
            tile_size=10,
            tile_workers=1,
            min_observations=2,
        )

        assert reads == [0, 1]
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 15))

    def test_percentile_stops_at_min_observations_per_pixel(self):
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
            mosaic_method="percentile",
            percentile=50.0,
            tile_size=10,
            tile_workers=1,
            min_observations=2,
        )

        expected = np.full((1, self.H, self.W), 10, dtype=np.uint16)
        expected[0, 0, 0] = 15
        assert reads == [0, 1, 2]
        np.testing.assert_array_equal(out, expected)

    def test_mean_caps_at_max_observations_per_pixel(self):
        # Three uniformly-clear scenes with values 10, 20, 30. With
        # max_observations=2 each pixel should only see the first two scenes,
        # so the mean is 15 (not 20).
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
            mosaic_method="mean",
            percentile=None,
            tile_size=10,
            tile_workers=1,
            max_observations=2,
        )

        assert reads == [0, 1]
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 15))

    def test_mean_max_observations_skips_capped_pixels_per_scene(self):
        # Pixel (0, 0) is cloudy in scenes 0 and 1, so it doesn't hit the cap
        # until scene 2. Other pixels hit cap=1 after scene 0; scene 1 has no
        # uncapped contribution left (it's cloudy on the only uncapped pixel),
        # so the contributor scan skips it entirely.
        reads = []
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]
        masks[0][0, 0] = False
        masks[1][0, 0] = False

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), (scene_idx + 1) * 10, dtype=np.uint16)

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="mean",
            percentile=None,
            tile_size=10,
            tile_workers=1,
            max_observations=1,
        )

        assert reads == [0, 2]
        expected = np.full((1, self.H, self.W), 10, dtype=np.uint16)
        expected[0, 0, 0] = 30
        np.testing.assert_array_equal(out, expected)

    def test_percentile_caps_at_max_observations_per_pixel(self):
        # Six clear scenes with values 0, 10, 20, 30, 40, 50. With
        # max_observations=3 each pixel sees only 0, 10, 20 — median is 10
        # rather than 25.
        reads = []
        n_scenes = 6
        scene_values = np.arange(n_scenes) * 10

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), scene_values[scene_idx], dtype=np.uint16)

        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(n_scenes)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="percentile",
            percentile=50.0,
            tile_size=10,
            tile_workers=1,
            max_observations=3,
        )

        assert sorted(set(reads)) == [0, 1, 2]
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 10))

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
            bands=["B04"],
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
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint16),
        )

        assert result == export_path
        assert reads
        assert not list(tmp_path.glob("streamed.tmp.*.tif"))
        with rio.open(export_path) as src:
            assert src.count == 1
            assert src.nodata == 0
            assert src.descriptions == ("B04",)
            np.testing.assert_array_equal(
                src.read(1), np.full((self.H, self.W), 20, dtype=np.uint16)
            )

    def test_write_tile_aggregation_geotiff_commits_final_path_after_close(
        self, tmp_path
    ):
        export_path = tmp_path / "committed.tif"
        final_path_seen_during_stream = []

        def read_fn(scene_idx, band_idx, spec):
            final_path_seen_during_stream.append(export_path.exists())
            _, _, h, w = spec
            return np.full((h, w), 10, dtype=np.uint16)

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
            bands=["B04"],
            masks=[np.ones((self.H, self.W), dtype=bool)],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            output_coverage_mask=None,
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint16),
        )

        assert final_path_seen_during_stream
        assert final_path_seen_during_stream == [False] * len(
            final_path_seen_during_stream
        )
        assert export_path.exists()
        assert not list(tmp_path.glob("committed.tmp.*.tif"))

    def test_write_tile_aggregation_geotiff_cleans_temp_on_error(self, tmp_path):
        export_path = tmp_path / "failed.tif"

        def read_fn(scene_idx, band_idx, spec):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
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
                bands=["B04"],
                masks=[np.ones((self.H, self.W), dtype=bool)],
                read_fn=read_fn,
                bands_count=1,
                height=self.H,
                width=self.W,
                coverage_mask=np.ones((self.H, self.W), dtype=bool),
                output_coverage_mask=None,
                mosaic_method="mean",
                percentile=None,
                tile_size=3,
                tile_workers=1,
                out_dtype=np.dtype(np.uint16),
            )

        assert not export_path.exists()
        assert not list(tmp_path.glob("failed.tmp.*.tif"))

    def test_write_tile_aggregation_geotiff_preserves_existing_output_on_error(
        self, tmp_path
    ):
        export_path = tmp_path / "existing.tif"
        profile = {
            "driver": "GTiff",
            "dtype": np.dtype(np.uint16),
            "width": self.W,
            "height": self.H,
            "count": 1,
            "crs": None,
            "transform": from_origin(0, self.H, 1, 1),
        }
        with rio.open(export_path, "w", **profile) as dst:
            dst.write(np.full((1, self.H, self.W), 7, dtype=np.uint16))

        def read_fn(scene_idx, band_idx, spec):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            write_tile_aggregation_geotiff(
                export_path=export_path,
                profile=profile,
                bands=["B04"],
                masks=[np.ones((self.H, self.W), dtype=bool)],
                read_fn=read_fn,
                bands_count=1,
                height=self.H,
                width=self.W,
                coverage_mask=np.ones((self.H, self.W), dtype=bool),
                output_coverage_mask=None,
                mosaic_method="mean",
                percentile=None,
                tile_size=3,
                tile_workers=1,
                out_dtype=np.dtype(np.uint16),
            )

        with rio.open(export_path) as src:
            np.testing.assert_array_equal(
                src.read(1), np.full((self.H, self.W), 7, dtype=np.uint16)
            )
        assert not list(tmp_path.glob("existing.tmp.*.tif"))

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
            bands=["B04"],
            masks=[np.ones((self.H, self.W), dtype=bool)],
            read_fn=read_fn,
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            output_coverage_mask=coverage,
            mosaic_method="mean",
            percentile=None,
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
        bands_count = 2
        height = 9
        width = 11
        full_output_shape = (bands_count, height, width)
        large_allocations = []
        original_zeros = aggregation_mod.np.zeros

        def tracking_zeros(shape, *args, **kwargs):
            if tuple(shape) == full_output_shape:
                large_allocations.append(shape)
            return original_zeros(shape, *args, **kwargs)

        monkeypatch.setattr(aggregation_mod.np, "zeros", tracking_zeros)

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
            bands=["B04", "B03"],
            masks=[np.ones((height, width), dtype=bool)],
            read_fn=read_fn,
            bands_count=bands_count,
            height=height,
            width=width,
            coverage_mask=np.ones((height, width), dtype=bool),
            output_coverage_mask=None,
            mosaic_method="mean",
            percentile=None,
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

        # The kernel-warming guard exists because the default runs threaded.
        # Pin the bound (>1) rather than the value so the test survives tuning.
        assert DEFAULT_TILE_WORKERS > 1
        assert _nanquantile_axis0.signatures
