import logging

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from s2mosaic import mosaic
from s2mosaic.config import validate_inputs


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
            self._call(self.VALID_BOUNDS, bands=["FOO"])

    def test_visual_band_with_other_bands_rejected(self):
        with pytest.raises(ValueError, match="Cannot use visual band with other bands"):
            self._call(self.VALID_BOUNDS, bands=["visual", "B04"])

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

    def test_auto_output_crs_rejects_polar_bounds(self):
        with pytest.raises(ValueError, match="pass output_crs explicitly"):
            self._call((0.0, 85.0, 1.0, 85.1))

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
            scene_order="valid_data",
            mosaic_method="mean",
            bands=["B04"],
            grid_id=None,
            percentile=None,
            bounds=utm_bounds,
            input_crs=32750,
            resolution=10,
        )

    def test_geographic_output_crs_rejected(self):
        # `resolution` is metres in the target CRS, so geographic output_crs
        # would produce a degenerate grid. Reject at validation time.
        with pytest.raises(ValueError, match="geographic CRS"):
            self._call(self.VALID_BOUNDS, output_crs=4326)

    def test_geographic_output_crs_rejected_grid_mode(self):
        # Same rule in grid mode — even though output_crs is otherwise ignored
        # there, surfacing the error keeps the message consistent across modes.
        with pytest.raises(ValueError, match="geographic CRS"):
            mosaic(grid_id="50HMH", start_year=2023, output_crs=4326)

    def test_projected_output_crs_accepted(self):
        # Smoke test: projected output_crs passes validation. Use Australian
        # Albers (EPSG:3577) — the canonical multi-zone alternative for AU.
        validate_inputs(
            scene_order="valid_data",
            mosaic_method="mean",
            bands=["B04"],
            grid_id=None,
            percentile=None,
            bounds=self.VALID_BOUNDS,
            input_crs=4326,
            output_crs=3577,
            resolution=30,
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
        # 5 deg x 5 deg AOI at 10m produces more than 20k x 20k pixels.
        with caplog.at_level(logging.WARNING, logger="s2mosaic.config"):
            validate_inputs(
                scene_order="valid_data",
                mosaic_method="mean",
                bands=["B04"],
                grid_id=None,
                percentile=None,
                bounds=(110.0, -35.0, 115.0, -30.0),
                input_crs=4326,
                resolution=10,
            )
        assert "larger than a 20,000 x 20,000 pixel raster" in caplog.text
        assert "width_px=" in caplog.text
        assert "height_px=" in caplog.text

    def test_bounds_large_utm_warns_but_is_accepted(self, caplog):
        # 300km x 300km in UTM at 10m produces 900M output pixels.
        with caplog.at_level(logging.WARNING, logger="s2mosaic.config"):
            validate_inputs(
                scene_order="valid_data",
                mosaic_method="mean",
                bands=["B04"],
                grid_id=None,
                percentile=None,
                bounds=(300_000.0, 6_300_000.0, 600_000.0, 6_600_000.0),
                input_crs=32750,
                resolution=10,
            )
        assert "larger than a 20,000 x 20,000 pixel raster" in caplog.text
        assert "pixels=900000000" in caplog.text

    def test_large_physical_bounds_at_coarse_resolution_do_not_warn(self, caplog):
        # Mirrors the wide visual export case: physically large, but only
        # ~8.4k x 4.9k pixels at 60m.
        with caplog.at_level(logging.WARNING, logger="s2mosaic.config"):
            validate_inputs(
                scene_order="valid_data",
                mosaic_method="mean",
                bands=["visual"],
                grid_id=None,
                percentile=None,
                bounds=(300_000.0, 6_300_000.0, 802_920.0, 6_588_890.0),
                input_crs=32750,
                resolution=60,
            )
        assert "20,000 x 20,000 pixel raster" not in caplog.text

    def test_typical_cross_tile_bounds_accepted(self):
        # ~80km × 80km, larger than a single S2 tile's overlap zone but well
        # under the 200km ceiling — must pass validation cleanly.
        validate_inputs(
            scene_order="valid_data",
            mosaic_method="mean",
            bands=["B04"],
            grid_id=None,
            percentile=None,
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
            scene_order="valid_data",
            mosaic_method="mean",
            bands=["B04"],
            grid_id=None,
            percentile=None,
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
                scene_order="valid_data",
                mosaic_method="mean",
                bands=["B04"],
                grid_id=None,
                percentile=None,
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
                scene_order="valid_data",
                mosaic_method="mean",
                bands=["B04"],
                grid_id=None,
                percentile=None,
                aoi=bowtie,
                input_crs=4326,
                resolution=10,
            )


class TestMosaicSharedParamsValidation:
    """Validation edge cases for params shared by both modes."""

    BOUNDS = (115.83, -31.97, 115.91, -31.94)

    def test_grid_mode_rejects_invalid_band(self):
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic(grid_id="50HMH", start_year=2023, bands=["FOO"])

    def test_bounds_mode_rejects_invalid_band(self):
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic(start_year=2023, bounds=self.BOUNDS, bands=["FOO"])

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

    @pytest.mark.parametrize("adaptive_tiling", [0, 1, "yes", None])
    def test_grid_mode_rejects_invalid_adaptive_tiling(self, adaptive_tiling):
        with pytest.raises(ValueError, match="adaptive_tiling must be"):
            mosaic(
                grid_id="50HMH",
                start_year=2023,
                adaptive_tiling=adaptive_tiling,
            )

    @pytest.mark.parametrize("adaptive_tiling", [0, 1, "yes", None])
    def test_bounds_mode_rejects_invalid_adaptive_tiling(self, adaptive_tiling):
        with pytest.raises(ValueError, match="adaptive_tiling must be"):
            mosaic(
                start_year=2023,
                bounds=self.BOUNDS,
                adaptive_tiling=adaptive_tiling,
            )
