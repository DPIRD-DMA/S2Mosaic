import threading
import warnings

import numpy as np
import pytest
import rasterio as rio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin

import s2mosaic.aggregation as aggregation_mod
from s2mosaic.aggregation import (
    DEFAULT_TILE_WORKERS,
    _drain_with_requeue,
    MEDOID_PIXEL_BLOCK,
    _medoid_axis0_u16,
    _quantile_axis0_u16,
    _source_valid_from_bands,
    _split_tile_size_aligned,
    _selection_network,
    _warm_medoid_axis0_u16,
    _warm_quantile_axis0_u16,
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

    def test_mean_ignores_all_zero_multi_band_source_pixels(self):
        scenes = np.stack(
            [
                np.full((3, self.H, self.W), 0, dtype=np.uint16),
                np.full((3, self.H, self.W), 30, dtype=np.uint16),
            ],
            axis=0,
        )
        scenes[0, :, :, 1:] = 10
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        expected = np.full((3, self.H, self.W), 20, dtype=np.uint16)
        expected[:, :, 0] = 30
        np.testing.assert_array_equal(out, expected)

    def test_mean_max_observations_does_not_count_all_zero_multi_band_pixels(self):
        scenes = np.stack(
            [
                np.zeros((3, self.H, self.W), dtype=np.uint16),
                np.full((3, self.H, self.W), 20, dtype=np.uint16),
                np.full((3, self.H, self.W), 40, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
            max_observations=2,
        )

        np.testing.assert_array_equal(out, np.full((3, self.H, self.W), 30))

    def test_mean_can_append_observation_count_band(self):
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
        masks[0][0, 0] = False

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
            include_observation_count=True,
        )

        expected_value = np.full((self.H, self.W), 20, dtype=np.uint16)
        expected_value[0, 0] = 30
        expected_count = np.full((self.H, self.W), 2, dtype=np.uint16)
        expected_count[0, 0] = 1
        assert out.shape == (2, self.H, self.W)
        np.testing.assert_array_equal(out[0], expected_value)
        np.testing.assert_array_equal(out[1], expected_count)

    def test_visual_observation_count_uses_uint16_output(self):
        scenes = np.stack(
            [
                np.full((3, self.H, self.W), 10, dtype=np.uint8),
                np.full((3, self.H, self.W), 30, dtype=np.uint8),
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
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="mean",
            percentile=None,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint8),
            include_observation_count=True,
        )

        assert out.dtype == np.uint16
        assert out.shape == (4, self.H, self.W)
        np.testing.assert_array_equal(out[:3], np.full((3, self.H, self.W), 20))
        np.testing.assert_array_equal(out[3], np.full((self.H, self.W), 2))

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

    def test_first_skips_all_zero_multi_band_source_pixels(self):
        scenes = np.stack(
            [
                np.zeros((3, self.H, self.W), dtype=np.uint16),
                np.full((3, self.H, self.W), 20, dtype=np.uint16),
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
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="first",
            percentile=None,
            tile_size=4,
            tile_workers=1,
        )

        np.testing.assert_array_equal(out, np.full((3, self.H, self.W), 20))

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

    def test_percentile_ignores_all_zero_multi_band_source_pixels(self):
        scenes = np.stack(
            [
                np.full((3, self.H, self.W), 0, dtype=np.uint16),
                np.full((3, self.H, self.W), 15, dtype=np.uint16),
            ],
            axis=0,
        )
        scenes[0, :, :, 1:] = 5
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="percentile",
            percentile=50.0,
            tile_size=2,
            tile_workers=1,
        )

        expected = np.full((3, self.H, self.W), 10, dtype=np.uint16)
        expected[:, :, 0] = 15
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("percentile, expected", [(25.0, 17), (75.0, 32)])
    def test_percentile_percentage_converts_to_kernel_fraction(
        self, percentile, expected
    ):
        # Ties the public 0-100 parameter to the kernel's 0-1 fraction. With
        # four evenly spaced observations q=0.25 interpolates to 17.5 and
        # q=0.75 to 32.5, both of which truncate on the cast to uint16, so a
        # percentage/fraction mix-up cannot pass by landing on a round number.
        # Values start at 10 because 0 is NODATA, not an observation.
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), v, dtype=np.uint16)
                for v in (10, 20, 30, 40)
            ],
            axis=0,
        )
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(4)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="percentile",
            percentile=percentile,
            tile_size=2,
            tile_workers=1,
        )

        np.testing.assert_array_equal(out, expected)

    def test_percentile_works_with_uint8_visual_source(self):
        # Visual mode hands uint8 RGB tiles to the aggregator and expects
        # uint8 output. tile_percentile stacks them as uint16 (the kernel is
        # specialised to uint16) and _finalise_tile clips back to uint8.
        scenes = np.stack(
            [
                np.full((3, self.H, self.W), 50, dtype=np.uint8),
                np.full((3, self.H, self.W), 120, dtype=np.uint8),
                np.full((3, self.H, self.W), 200, dtype=np.uint8),
            ],
            axis=0,
        )
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(3)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="percentile",
            percentile=50.0,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint8),
        )

        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, 120)

    def test_medoid_picks_scene_closest_to_band_median(self):
        # Three scenes, 3 bands. The per-band median is the middle scene's
        # values; the medoid must return that scene's full spectrum (not a
        # synthetic mix of per-band medians).
        scenes = np.stack(
            [
                np.array([[[10]], [[10]], [[10]]], dtype=np.uint16),
                np.array([[[20]], [[20]], [[20]]], dtype=np.uint16),
                np.array([[[30]], [[30]], [[30]]], dtype=np.uint16),
            ],
            axis=0,
        )
        scenes = np.broadcast_to(scenes, (3, 3, self.H, self.W)).copy()
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(3)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        np.testing.assert_array_equal(out, 20)

    def test_medoid_returns_observed_spectrum_not_synthetic(self):
        # Two scenes whose per-band median spectrum is closer to scene 0
        # than scene 1. The medoid must return scene 0's actual values, the
        # property that distinguishes medoid from per-band median (which
        # would interpolate).
        s0 = np.array([5, 8, 12], dtype=np.uint16)
        s1 = np.array([100, 80, 60], dtype=np.uint16)
        scenes = np.zeros((2, 3, self.H, self.W), dtype=np.uint16)
        for b in range(3):
            scenes[0, b].fill(s0[b])
            scenes[1, b].fill(s1[b])
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(2)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        # With two scenes the per-band median lies exactly between them,
        # so squared distances tie. The kernel breaks ties on first-seen,
        # so scene 0 wins. Either scene is a valid medoid; both are
        # actually-observed spectra.
        for b in range(3):
            assert np.all((out[b] == s0[b]) | (out[b] == s1[b]))
        # And the output is uniform per band (only one scene chosen
        # tile-wide).
        for b in range(3):
            assert np.unique(out[b]).size == 1

    def test_medoid_skips_masked_scenes(self):
        # Outlier scene is masked off; medoid should ignore it entirely
        # and pick from the remaining two.
        scenes = np.stack(
            [
                np.full((1, self.H, self.W), 100, dtype=np.uint16),  # outlier
                np.full((1, self.H, self.W), 20, dtype=np.uint16),
                np.full((1, self.H, self.W), 22, dtype=np.uint16),
            ],
            axis=0,
        )
        masks = [
            np.zeros((self.H, self.W), dtype=bool),  # fully masked
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
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        # Must come from one of the unmasked scenes, never the outlier.
        assert np.all((out == 20) | (out == 22))

    def test_medoid_zero_where_no_scene_is_valid(self):
        scenes = np.full((2, 1, self.H, self.W), 10, dtype=np.uint16)
        masks = [
            np.zeros((self.H, self.W), dtype=bool),
            np.zeros((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=1,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        np.testing.assert_array_equal(out, 0)

    def test_medoid_ignores_all_zero_multi_band_source_pixels(self):
        # First scene's column-0 pixels are all-zero across bands, so the
        # source-valid check should treat those as no-data and exclude
        # them from medoid candidates. Remaining scene wins.
        scenes = np.stack(
            [
                np.full((3, self.H, self.W), 0, dtype=np.uint16),
                np.full((3, self.H, self.W), 15, dtype=np.uint16),
            ],
            axis=0,
        )
        scenes[0, :, :, 1:] = 5
        masks = [
            np.ones((self.H, self.W), dtype=bool),
            np.ones((self.H, self.W), dtype=bool),
        ]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        # Column 0 has only one valid candidate (scene 1) → 15.
        # Other columns have two candidates with values 5 and 15; medoid
        # picks whichever wins the tie-break, but the answer must be an
        # actual observation.
        assert np.all(out[:, :, 0] == 15)
        for col in range(1, self.W):
            for b in range(3):
                assert np.all((out[b, :, col] == 5) | (out[b, :, col] == 15))

    def test_medoid_works_with_uint8_visual_source(self):
        # Visual mode hands uint8 RGB tiles to the aggregator and expects
        # uint8 output. tile_medoid widens uint8 → uint16 inside the kernel
        # then _finalise_tile clips back to uint8.
        scenes = np.stack(
            [
                np.full((3, self.H, self.W), 50, dtype=np.uint8),
                np.full((3, self.H, self.W), 120, dtype=np.uint8),
                np.full((3, self.H, self.W), 200, dtype=np.uint8),
            ],
            axis=0,
        )
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(3)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=3,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
            out_dtype=np.dtype(np.uint8),
        )

        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, 120)

    def test_medoid_single_scene_falls_back_to_copy(self):
        scenes = np.full((1, 2, self.H, self.W), 42, dtype=np.uint16)
        masks = [np.ones((self.H, self.W), dtype=bool)]

        out = run_tile_aggregation(
            masks=masks,
            read_fn=self._read_fn_for(scenes),
            bands_count=2,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=3,
            tile_workers=1,
        )

        np.testing.assert_array_equal(out, 42)

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

        from concurrent.futures import Future

        class FakeExecutor:
            def __init__(self, max_workers):
                captured_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, fn, specs):
                return map(fn, specs)

            def submit(self, fn, *args, **kwargs):
                fut: Future = Future()
                try:
                    fut.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    fut.set_exception(exc)
                return fut

            def shutdown(self, wait=True):
                return None

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

        # Scene values 10/20/30: 0 would read as NODATA, not an observation.
        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), 10 + scene_idx * 10, dtype=np.uint16)

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

        # Every pixel but (0, 0) sees all three scenes, so the median is 20;
        # (0, 0) is masked out of scene 0 and medians 20 and 30 to 25.
        expected = np.full((1, self.H, self.W), 20, dtype=np.uint16)
        expected[0, 0, 0] = 25
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
        # Six clear scenes with values 10, 20, 30, 40, 50, 60. With
        # max_observations=3 each pixel sees only 10, 20, 30, so the median is
        # 20 rather than 35. Values start at 10 because 0 is NODATA.
        reads = []
        n_scenes = 6
        scene_values = 10 + np.arange(n_scenes) * 10

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
        np.testing.assert_array_equal(out, np.full((1, self.H, self.W), 20))

    def test_medoid_stops_at_min_observations(self):
        # Documented as applying to medoid as well as mean/percentile, but
        # ``tile_medoid`` keeps its own copy of the early-stop rather than
        # sharing ``tile_percentile``'s, so it needs its own coverage.
        # Four clear scenes; stopping after three leaves the median at 20 and
        # the outlier scene unread.
        reads = []
        scene_values = [10, 20, 30, 99]

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), scene_values[scene_idx], dtype=np.uint16)

        out = run_tile_aggregation(
            masks=[np.ones((self.H, self.W), dtype=bool) for _ in scene_values],
            read_fn=read_fn,
            bands_count=2,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=10,
            tile_workers=1,
            min_observations=3,
        )

        assert sorted(set(reads)) == [0, 1, 2]
        np.testing.assert_array_equal(out, np.full((2, self.H, self.W), 20))

    def test_medoid_caps_at_max_observations_per_pixel(self):
        # The cap is per pixel, not per tile, so a pixel that is cloudy early
        # keeps accepting later scenes after its neighbours have filled up.
        # Pixel (0, 0) misses scene 0, so with max_observations=2 it composes
        # scenes 1 and 2 (medoid 20) while every other pixel composes scenes
        # 0 and 1 (medoid 10). A cap applied tile-wide would make them equal.
        reads = []
        scene_values = [10, 20, 30, 40]
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in scene_values]
        masks[0][0, 0] = False

        def read_fn(scene_idx, band_idx, spec):
            reads.append(scene_idx)
            _, _, h, w = spec
            return np.full((h, w), scene_values[scene_idx], dtype=np.uint16)

        out = run_tile_aggregation(
            masks=masks,
            read_fn=read_fn,
            bands_count=2,
            height=self.H,
            width=self.W,
            coverage_mask=np.ones((self.H, self.W), dtype=bool),
            mosaic_method="medoid",
            percentile=None,
            tile_size=10,
            tile_workers=1,
            max_observations=2,
        )

        expected = np.full((2, self.H, self.W), 10, dtype=np.uint16)
        expected[:, 0, 0] = 20
        assert sorted(set(reads)) == [0, 1, 2]
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize(
        "limits", [{"min_observations": 2}, {"max_observations": 2}]
    )
    def test_medoid_and_percentile_read_the_same_scenes(self, limits):
        # The two methods carry duplicate copies of the contributor scan and
        # the per-scene early-stop. They currently agree scene for scene; this
        # pins that, so a change to one that is not made to the other fails
        # here rather than silently changing how much a medoid mosaic reads.
        # A scene fully cloudy at one pixel makes the stop depend on the
        # per-pixel counts rather than a flat scene count.
        masks = [np.ones((self.H, self.W), dtype=bool) for _ in range(4)]
        masks[1][0, 0] = False

        def reads_for(mosaic_method):
            reads = []

            def read_fn(scene_idx, band_idx, spec):
                reads.append(scene_idx)
                _, _, h, w = spec
                return np.full((h, w), 10 + scene_idx * 10, dtype=np.uint16)

            run_tile_aggregation(
                masks=masks,
                read_fn=read_fn,
                bands_count=2,
                height=self.H,
                width=self.W,
                coverage_mask=np.ones((self.H, self.W), dtype=bool),
                mosaic_method=mosaic_method,
                percentile=50.0 if mosaic_method == "percentile" else None,
                tile_size=10,
                tile_workers=1,
                **limits,
            )
            return reads

        assert reads_for("medoid") == reads_for("percentile")

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
            assert src.nodata is None
            assert src.descriptions == ("B04",)
            np.testing.assert_array_equal(
                src.read(1), np.full((self.H, self.W), 20, dtype=np.uint16)
            )
            np.testing.assert_array_equal(
                src.dataset_mask(), np.full((self.H, self.W), 255, dtype=np.uint8)
            )

    def test_write_tile_aggregation_geotiff_appends_observation_count(self, tmp_path):
        def read_fn(scene_idx, band_idx, spec):
            _, _, h, w = spec
            return np.full((h, w), 10 + scene_idx * 20, dtype=np.uint16)

        export_path = tmp_path / "with-count.tif"

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
            include_observation_count=True,
        )

        with rio.open(export_path) as src:
            assert src.count == 2
            assert src.dtypes == ("uint16", "uint16")
            assert src.descriptions == ("B04", "Observation count")
            np.testing.assert_array_equal(
                src.read(1), np.full((self.H, self.W), 20, dtype=np.uint16)
            )
            np.testing.assert_array_equal(
                src.read(2), np.full((self.H, self.W), 2, dtype=np.uint16)
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
            mask = src.dataset_mask()
        assert data[0, 0] == 0
        assert data[0, 1] == 10
        assert mask[0, 0] == 0
        assert mask[0, 1] == 255

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


class TestSourceValidFromBands:
    """A pixel is usable only where every requested band has a value.

    0 is Sentinel-2 L2A's NODATA, never a measurable reflectance, so a zero
    in any band means that band dropped out at that pixel. These pin the
    stricter reading, because the earlier all-bands-zero test passed such a
    pixel through and wrote the 0 into the output as if it were data.
    """

    def test_rejects_pixel_where_a_single_band_dropped_out(self):
        # The MPC defect this was written for: one band reads 0 while the
        # rest carry ordinary dark-water values around DN 1000.
        band_data = [
            np.array([[1026, 1040]], dtype=np.uint16),
            np.array([[0, 1013]], dtype=np.uint16),
            np.array([[977, 986]], dtype=np.uint16),
        ]
        valid = _source_valid_from_bands(band_data)
        assert valid is not None
        np.testing.assert_array_equal(valid, [[False, True]])

    def test_rejects_pixel_where_every_band_is_zero(self):
        # The scene-footprint case the previous test already handled; it must
        # keep working, since that is what keeps all-zero scenes from
        # consuming the max_observations budget.
        band_data = [np.zeros((1, 2), dtype=np.uint16) for _ in range(3)]
        valid = _source_valid_from_bands(band_data)
        assert valid is not None
        np.testing.assert_array_equal(valid, [[False, False]])

    def test_keeps_pixel_where_every_band_has_data(self):
        band_data = [
            np.array([[1, 65535]], dtype=np.uint16),
            np.array([[1000, 1]], dtype=np.uint16),
            np.array([[65535, 1000]], dtype=np.uint16),
        ]
        valid = _source_valid_from_bands(band_data)
        assert valid is not None
        np.testing.assert_array_equal(valid, [[True, True]])

    def test_single_band_reads_are_filtered_too(self):
        # This returned None before, so a one-band request got no zero
        # filtering at all and wrote NODATA out as if it were a reading.
        band_data = [np.array([[0, 1200]], dtype=np.uint16)]
        valid = _source_valid_from_bands(band_data)
        assert valid is not None
        np.testing.assert_array_equal(valid, [[False, True]])

    def test_empty_read_returns_none(self):
        assert _source_valid_from_bands([]) is None

    @pytest.mark.parametrize("n_bands", [1, 2, 3, 4, 13])
    def test_matches_an_explicit_all_bands_non_zero_reduction(self, n_bands):
        rng = np.random.default_rng(0)
        band_data = [
            rng.integers(0, 3, size=(8, 8), dtype=np.uint16) for _ in range(n_bands)
        ]
        expected = np.logical_and.reduce([b != 0 for b in band_data])
        np.testing.assert_array_equal(_source_valid_from_bands(band_data), expected)

    def test_visual_channel_dropout_is_rejected(self):
        # uint8 TCI: nodata-region quantisation noise leaves 1-2 DN in one
        # channel while another is 0. Same rule, different dtype.
        band_data = [
            np.array([[1, 40]], dtype=np.uint8),
            np.array([[0, 38]], dtype=np.uint8),
            np.array([[2, 44]], dtype=np.uint8),
        ]
        valid = _source_valid_from_bands(band_data)
        assert valid is not None
        np.testing.assert_array_equal(valid, [[False, True]])


class TestSelectionNetwork:
    """The shared comparator builder must sort exactly what it promises."""

    @pytest.mark.parametrize("n_scenes", [1, 2, 3, 5, 8, 9, 16, 17, 33])
    @pytest.mark.parametrize("live_fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_sorts_every_position_up_to_max_live(self, n_scenes, live_fraction):
        # Both kernels prune the network to a live range, then read a position
        # that moves per pixel with that pixel's valid count. The medoid asks
        # for n // 2; the quantile asks for ceil(q*(n-1)), which spans 0 (q=0)
        # to n-1 (q=1). A builder that under-sorts the top of the range fails
        # only on the pixels that read there, so pin the contract here rather
        # than relying on the kernels to notice.
        max_live = int(np.ceil(live_fraction * (n_scenes - 1)))
        net = _selection_network(n_scenes, max_live)
        rng = np.random.default_rng(n_scenes * 100 + max_live)

        for _ in range(100):
            values = rng.integers(0, 65536, size=n_scenes).astype(np.uint16)
            wires = values.copy()
            for lo_w, hi_w in net:
                if wires[lo_w] > wires[hi_w]:
                    wires[lo_w], wires[hi_w] = wires[hi_w], wires[lo_w]
            expected = np.sort(values)
            np.testing.assert_array_equal(
                wires[: max_live + 1], expected[: max_live + 1]
            )

    @pytest.mark.parametrize("n_scenes", [4, 7, 16])
    def test_max_live_beyond_the_last_wire_is_clamped(self, n_scenes):
        # ceil(q*(n-1)) tops out at n-1, but nothing in the signature stops a
        # caller passing more; clamping keeps it a full sort rather than an
        # out-of-bounds write into the liveness array.
        net = _selection_network(n_scenes, n_scenes + 5)
        rng = np.random.default_rng(n_scenes)
        for _ in range(50):
            values = rng.integers(0, 65536, size=n_scenes).astype(np.uint16)
            wires = values.copy()
            for lo_w, hi_w in net:
                if wires[lo_w] > wires[hi_w]:
                    wires[lo_w], wires[hi_w] = wires[hi_w], wires[lo_w]
            np.testing.assert_array_equal(wires, np.sort(values))

    def test_single_wire_needs_no_comparators(self):
        assert _selection_network(1, 0).shape == (0, 2)


class TestQuantileAxis0U16:
    """Vectorised integer quantile reducer must match NumPy nanquantile."""

    def test_warm_compile_runs_before_threaded_aggregation(self):
        _warm_quantile_axis0_u16()
        assert _quantile_axis0_u16.signatures

    @staticmethod
    def _assert_matches_numpy(stack: np.ndarray, valid: np.ndarray, q: float):
        """Compare against ``np.nanquantile`` on an equivalent float stack.

        NumPy needs NaN sentinels to skip invalid observations, so the
        reference is the same data with ``~valid`` positions set to NaN.

        ``valid`` is per (scene, pixel), matching what the tile workers build:
        ``_source_valid_from_bands`` resolves all bands of a scene down to one
        mask, so a scene is either a candidate at a pixel or it is not.

        The comparison is exact, not approximate. The kernel computes the read
        position and the blend the way NumPy does, so anything short of
        bit-identity means one of them drifted.

        The reference is built as float64 even though the kernel returns
        float32, because that is the quantile the kernel actually targets: it
        finds the bracketing order statistics in integer space and does the
        position and blend in float64, casting only at the end. Handing
        ``nanquantile`` a float32 stack instead makes the reference depend on
        the NumPy version. Up to 1.26 it computed the lerp in float64 and
        agreed; from 2.x it computes in float32 and lands an ulp off for
        quantiles falling between two observations, so the same correct kernel
        output started failing. Comparing against the float64 quantile pins
        the property that matters and is stable across both.
        """
        reference = stack.astype(np.float64)
        reference[~np.broadcast_to(valid[:, None], stack.shape)] = np.nan

        got = _quantile_axis0_u16(stack, valid, q)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            expected = np.nanquantile(reference, q, axis=0).astype(np.float32)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
    def test_random_sparse_stacks(self, q):
        rng = np.random.default_rng(42)
        for scenes, bands, h, w, valid_fraction in [
            (1, 1, 3, 4, 0.7),
            (2, 3, 5, 4, 0.5),
            (5, 2, 4, 6, 0.8),
            (17, 3, 5, 7, 0.35),
        ]:
            stack = rng.integers(0, 12000, size=(scenes, bands, h, w), dtype=np.uint16)
            valid = rng.random((scenes, h, w)) < valid_fraction
            self._assert_matches_numpy(stack, valid, q)

    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_all_invalid_stack(self, q):
        stack = np.zeros((4, 2, 3, 5), dtype=np.uint16)
        valid = np.zeros((4, 3, 5), dtype=bool)
        self._assert_matches_numpy(stack, valid, q)

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    def test_single_valid_scene_among_invalid(self, q):
        stack = np.zeros((6, 2, 3, 4), dtype=np.uint16)
        stack[3] = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        valid = np.zeros((6, 3, 4), dtype=bool)
        valid[3] = True
        self._assert_matches_numpy(stack, valid, q)

    @pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_interpolation_matches_numpy(self, q):
        base = np.array([0, 10, 20, 40, 80], dtype=np.uint16)
        stack = np.broadcast_to(base[:, None, None, None], (5, 1, 3, 4)).copy()
        valid = np.ones((5, 3, 4), dtype=bool)
        self._assert_matches_numpy(stack, valid, q)

    def test_preserves_band_pixel_independence(self):
        stack = np.array(
            [
                [[[1, 100], [0, 4]], [[10, 0], [30, 40]]],
                [[[3, 0], [5, 6]], [[20, 25], [0, 60]]],
                [[[7, 200], [9, 0]], [[0, 35], [50, 80]]],
            ],
            dtype=np.uint16,
        )
        valid = np.array(
            [
                [[True, True], [False, True]],
                [[True, False], [True, True]],
                [[True, True], [False, True]],
            ]
        )
        self._assert_matches_numpy(stack, valid, 0.5)

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_every_valid_count_is_covered_by_the_pruned_network(self, q):
        # The network is pruned to ceil(q*(n-1)) live positions, but a pixel
        # with k valid observations reads floor/ceil(q*(k-1)), a position that
        # moves with k. One pixel per possible k pins that the pruning is not
        # tighter than the shallowest pixel needs.
        n_scenes = 12
        rng = np.random.default_rng(int(q * 100) + 5)
        stack = rng.integers(
            0, 12000, size=(n_scenes, 2, 1, n_scenes + 1), dtype=np.uint16
        )
        valid = np.zeros((n_scenes, 1, n_scenes + 1), dtype=bool)
        for k in range(n_scenes + 1):
            valid[:k, 0, k] = True
        self._assert_matches_numpy(stack, valid, q)

    @pytest.mark.parametrize(
        "n_scenes,percentile", [(26, 60.0), (46, 60.0), (51, 30.0), (101, 15.0)]
    )
    def test_read_position_matches_numpy_where_float32_would_not(
        self, n_scenes, percentile
    ):
        # For these (n, percentile) pairs q*(n-1) lands exactly on an integer
        # in float64 but just above it in float32. Computing the read position
        # in float32 both interpolated where NumPy returns an order statistic
        # outright, and — for the deepest pixel — read one position past what
        # the pruned network sorted. Deviation was small enough to survive the
        # uint16 cast, so only an exact comparison catches it.
        rng = np.random.default_rng(n_scenes)
        stack = rng.integers(0, 65536, size=(n_scenes, 1, 1, 512), dtype=np.uint16)
        valid = np.ones((n_scenes, 1, 512), dtype=bool)

        got = _quantile_axis0_u16(stack, valid, percentile / 100.0)
        expected = np.quantile(stack.astype(np.float64), percentile / 100.0, axis=0)
        np.testing.assert_array_equal(got, expected.astype(np.float32))

    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
    def test_saturated_values_are_not_confused_with_padding(self, q):
        # Invalid observations are padded with 65535; real 65535 observations
        # must still be ranked normally.
        stack = np.full((3, 2, 1, 1), 65535, dtype=np.uint16)
        stack[1, :, 0, 0] = 10
        valid = np.array([True, True, False]).reshape(3, 1, 1)
        self._assert_matches_numpy(stack, valid, q)

    def test_invalid_positions_need_not_be_zeroed(self):
        # Callers flag validity separately, so whatever sits at an invalid
        # position must never reach the output.
        rng = np.random.default_rng(3)
        stack = rng.integers(0, 12000, size=(5, 2, 4, 4), dtype=np.uint16)
        valid = rng.random((5, 4, 4)) < 0.6

        clean = np.where(np.broadcast_to(valid[:, None], stack.shape), stack, 0)
        dirty = np.where(
            np.broadcast_to(valid[:, None], stack.shape), stack, np.uint16(54321)
        )

        np.testing.assert_array_equal(
            _quantile_axis0_u16(clean, valid, 0.5),
            _quantile_axis0_u16(dirty, valid, 0.5),
        )

    def test_default_tile_workers_uses_thread_safe_percentile_kernel(self):
        _warm_quantile_axis0_u16()

        # The kernel-warming guard exists because the default runs threaded.
        # Pin the bound (>1) rather than the value so the test survives tuning.
        assert DEFAULT_TILE_WORKERS > 1
        assert _quantile_axis0_u16.signatures


class TestMedoidAxis0U16:
    """Integer medoid reducer must match closest-to-median-vector semantics."""

    @staticmethod
    def _reference_medoid(
        stack: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_scenes, n_bands, h, w = stack.shape
        out = np.zeros((n_bands, h, w), dtype=np.uint16)
        out_valid = np.zeros((h, w), dtype=bool)

        for y in range(h):
            for x in range(w):
                scene_valid = valid[:, y, x]
                if not scene_valid.any():
                    continue
                spectra = stack[scene_valid, :, y, x].astype(np.float64)
                target = np.median(spectra, axis=0)
                distances = ((spectra - target) ** 2).sum(axis=1)
                best_rel = int(np.argmin(distances))
                best_scene = np.flatnonzero(scene_valid)[best_rel]
                out[:, y, x] = stack[best_scene, :, y, x]
                out_valid[y, x] = True

        return out, out_valid

    def test_even_count_median_uses_exact_half_integer_target(self):
        stack = np.zeros((4, 2, 1, 1), dtype=np.uint16)
        stack[:, :, 0, 0] = np.array(
            [
                [0, 0],
                [0, 2],
                [0, 2],
                [1, 1],
            ],
            dtype=np.uint16,
        )
        valid = np.ones((4, 1, 1), dtype=bool)

        got, got_valid = _medoid_axis0_u16(stack, valid)

        np.testing.assert_array_equal(got_valid, [[True]])
        np.testing.assert_array_equal(got[:, 0, 0], [0, 2])

    def test_random_stacks_match_float_reference(self):
        rng = np.random.default_rng(123)
        for scenes, bands, h, w in [
            (2, 1, 4, 5),
            (4, 2, 3, 4),
            (6, 4, 3, 3),
            (9, 3, 4, 2),
        ]:
            for _ in range(10):
                stack = rng.integers(
                    0, 12000, size=(scenes, bands, h, w), dtype=np.uint16
                )
                valid = rng.random((scenes, h, w)) < 0.75

                got, got_valid = _medoid_axis0_u16(stack, valid)
                expected, expected_valid = self._reference_medoid(stack, valid)

                np.testing.assert_array_equal(got_valid, expected_valid)
                np.testing.assert_array_equal(got, expected)

    def test_warm_compile_runs_before_threaded_aggregation(self):
        _warm_medoid_axis0_u16()
        assert _medoid_axis0_u16.signatures

    @pytest.mark.parametrize("n_scenes", [1, 2, 3, 5, 8, 9, 16, 17, 33, 64])
    def test_selection_network_sorts_every_position_it_promises(self, n_scenes):
        # The kernel pads short pixels, so a pixel with k valid observations
        # reads position k // 2, which varies per pixel. Pruning the network
        # to the middle of the full width alone would corrupt shallow pixels,
        # so every position up to half the width must come out sorted.
        net = _selection_network(n_scenes, n_scenes // 2)
        rng = np.random.default_rng(n_scenes)
        wanted = n_scenes // 2
        for _ in range(200):
            values = rng.integers(0, 65536, size=n_scenes).astype(np.uint16)
            wires = values.copy()
            for lo_w, hi_w in net:
                if wires[lo_w] > wires[hi_w]:
                    wires[lo_w], wires[hi_w] = wires[hi_w], wires[lo_w]
            expected = np.sort(values)
            np.testing.assert_array_equal(wires[: wanted + 1], expected[: wanted + 1])

    def test_deep_stack_matches_float_reference(self):
        # Annual composites run far past the scene counts the old per-pixel
        # insertion sort was tuned for.
        rng = np.random.default_rng(7)
        stack = rng.integers(0, 12000, size=(64, 3, 5, 7), dtype=np.uint16)
        valid = rng.random((64, 5, 7)) < 0.6

        got, got_valid = _medoid_axis0_u16(stack, valid)
        expected, expected_valid = self._reference_medoid(stack, valid)

        np.testing.assert_array_equal(got_valid, expected_valid)
        np.testing.assert_array_equal(got, expected)

    def test_saturated_values_are_not_confused_with_padding(self):
        # Invalid observations are padded with 65535; real 65535 observations
        # must still be selected normally.
        stack = np.full((3, 2, 1, 1), 65535, dtype=np.uint16)
        stack[1, :, 0, 0] = 10
        valid = np.array([True, True, False]).reshape(3, 1, 1)

        got, got_valid = _medoid_axis0_u16(stack, valid)

        np.testing.assert_array_equal(got_valid, [[True]])
        # Two valid observations, 65535 and 10: the doubled median is their
        # sum, so both are equidistant and the first wins on strict argmin.
        np.testing.assert_array_equal(got[:, 0, 0], [65535, 65535])

    def test_rows_wider_than_the_pixel_block(self):
        # Widths that are not a multiple of MEDOID_PIXEL_BLOCK exercise the
        # short trailing block.
        rng = np.random.default_rng(11)
        for width in (MEDOID_PIXEL_BLOCK - 1, MEDOID_PIXEL_BLOCK + 1):
            stack = rng.integers(0, 9000, size=(6, 2, 2, width), dtype=np.uint16)
            valid = rng.random((6, 2, width)) < 0.7

            got, got_valid = _medoid_axis0_u16(stack, valid)
            expected, expected_valid = self._reference_medoid(stack, valid)

            np.testing.assert_array_equal(got_valid, expected_valid)
            np.testing.assert_array_equal(got, expected)


class TestDrainWithRequeue:
    """Parallel tile drain with bounded requeue on transient IO failures."""

    @staticmethod
    def _spec(i):
        # Specs are (row, col, h, w) tuples; encode the id in row so worker
        # callbacks can distinguish them.
        return (i, 0, 1, 1)

    def test_yields_all_specs_when_no_failures(self):
        specs = [self._spec(i) for i in range(4)]
        seen = []

        def worker_fn(spec):
            seen.append(spec)
            return spec, np.array([spec[0]])

        results = list(
            _drain_with_requeue(
                specs=specs, worker_fn=worker_fn, n_workers=2, log_every=1
            )
        )

        assert sorted(seen) == specs
        assert {spec for spec, _ in results} == set(specs)

    def test_requeues_after_transient_failure_and_yields_result(self):
        specs = [self._spec(0), self._spec(1)]
        failures = {self._spec(0): 1}  # spec 0 fails once, then succeeds
        lock = threading.Lock()

        def worker_fn(spec):
            with lock:
                remaining = failures.get(spec, 0)
                if remaining > 0:
                    failures[spec] = remaining - 1
                    raise RasterioIOError("transient")
            return spec, np.array([spec[0]])

        results = list(
            _drain_with_requeue(
                specs=specs, worker_fn=worker_fn, n_workers=2, log_every=10
            )
        )

        assert {spec for spec, _ in results} == set(specs)
        assert failures[self._spec(0)] == 0

    def test_raises_when_per_tile_cap_exceeded(self, monkeypatch):
        # MAX_REQUEUES_PER_TILE=2 means the spec gets 3 worker calls total
        # (initial + 2 requeues) before the failure propagates. Use a large
        # total-fraction so the per-tile cap is what trips.
        monkeypatch.setattr(aggregation_mod, "MAX_REQUEUES_PER_TILE", 2)
        monkeypatch.setattr(aggregation_mod, "MAX_TOTAL_REQUEUE_FRACTION", 10.0)
        specs = [self._spec(0)]
        calls = {"count": 0}

        def worker_fn(spec):
            calls["count"] += 1
            raise RasterioIOError("permanent")

        with pytest.raises(RasterioIOError, match="permanent"):
            list(
                _drain_with_requeue(
                    specs=specs, worker_fn=worker_fn, n_workers=1, log_every=10
                )
            )
        assert calls["count"] == 3

    def test_raises_when_total_requeue_cap_exceeded(self, monkeypatch):
        # 4 tiles all failing, total cap of 1 requeue. The first requeue uses
        # the budget; the second failure must surface immediately.
        monkeypatch.setattr(aggregation_mod, "MAX_REQUEUES_PER_TILE", 5)
        monkeypatch.setattr(aggregation_mod, "MAX_TOTAL_REQUEUE_FRACTION", 0.25)
        specs = [self._spec(i) for i in range(4)]

        def worker_fn(spec):
            raise RasterioIOError("transient")

        with pytest.raises(RasterioIOError, match="transient"):
            list(
                _drain_with_requeue(
                    specs=specs, worker_fn=worker_fn, n_workers=1, log_every=10
                )
            )

    def test_non_io_error_propagates_without_requeue(self):
        specs = [self._spec(0)]
        calls = {"count": 0}

        def worker_fn(spec):
            calls["count"] += 1
            raise ValueError("not transient")

        with pytest.raises(ValueError, match="not transient"):
            list(
                _drain_with_requeue(
                    specs=specs, worker_fn=worker_fn, n_workers=1, log_every=10
                )
            )
        assert calls["count"] == 1
