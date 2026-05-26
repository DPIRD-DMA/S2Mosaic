import pytest

from s2mosaic.geometry import (
    _target_grid,
    pick_utm_epsg,
    reproject_bbox,
)


class TestPickUtmEpsg:
    """UTM zone picking from lat/lon."""

    @pytest.mark.parametrize(
        "lon, lat, expected",
        [
            (115.86, -31.95, 32750),  # Perth, WA → UTM 50S
            (115.86, 31.95, 32650),  # mirror in the north → UTM 50N
            (-122.43, 37.77, 32610),  # San Francisco → UTM 10N
            (0.0, 0.0, 32631),  # equator/Greenwich → UTM 31N
            (-180.0, 0.0, 32601),  # exact western dateline clamps to UTM 1N
            (-179.999, 0.0, 32601),
            (-179.9, 0.0, 32601),  # west of dateline → UTM 1N
            (179.9, 0.0, 32660),  # east of dateline → UTM 60N
            (179.999, 0.0, 32660),
            (180.0, 0.0, 32660),  # exact dateline clamps to UTM 60N
        ],
    )
    def test_known_locations(self, lon, lat, expected):
        assert pick_utm_epsg(lon, lat) == expected

    @pytest.mark.parametrize(
        "lon, lat, match",
        [
            (-180.1, 0.0, "longitude"),
            (180.1, 0.0, "longitude"),
            (0.0, -90.1, "latitude"),
            (0.0, 90.1, "latitude"),
            (0.0, -80.1, "pass output_crs explicitly"),
            (0.0, 84.1, "pass output_crs explicitly"),
        ],
    )
    def test_invalid_coordinates_rejected(self, lon, lat, match):
        with pytest.raises(ValueError, match=match):
            pick_utm_epsg(lon, lat)


class TestTargetGrid:
    def test_tiny_valid_bounds_still_get_one_pixel(self):
        _, width, height, crs = _target_grid(
            (390_000.0, 6_460_000.0, 390_010.0, 6_460_010.0),
            resolution=20,
            target_crs=32750,
        )

        assert (width, height) == (1, 1)
        assert crs.to_epsg() == 32750


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
