import pytest
from shapely.geometry import Polygon

from s2mosaic.config import MosaicRequest
from s2mosaic.geometry import (
    _snap_bounds_to_grid,
    _target_grid,
    densify_bbox_to_polygon,
    pick_utm_epsg,
    reproject_aoi,
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


class TestSnapBoundsToGrid:
    @pytest.mark.parametrize(
        "bounds, resolution, expected",
        [
            # Already aligned at 10m → unchanged.
            (
                (390_000.0, 6_460_000.0, 390_100.0, 6_460_100.0),
                10,
                (390_000.0, 6_460_000.0, 390_100.0, 6_460_100.0),
            ),
            # Fractional bounds at 10m → expand outward.
            (
                (390_003.7, 6_460_002.1, 390_096.4, 6_460_098.9),
                10,
                (390_000.0, 6_460_000.0, 390_100.0, 6_460_100.0),
            ),
            # Coarser resolution snaps to multiples of 30m.
            (
                (390_001.0, 6_460_001.0, 390_059.0, 6_460_059.0),
                30,
                (390_000.0, 6_459_990.0, 390_060.0, 6_460_080.0),
            ),
        ],
    )
    def test_snap_expands_outward_to_resolution_multiples(
        self, bounds, resolution, expected
    ):
        assert _snap_bounds_to_grid(bounds, resolution) == expected

    def test_snap_then_target_grid_yields_aligned_transform(self):
        # A fractional UTM bbox of the kind that drops out of a lon/lat
        # reproject. After snap, the transform origin must be a clean
        # multiple of resolution and width/height an integer multiple too.
        raw_utm_bbox = (390_003.7, 6_460_002.1, 390_096.4, 6_460_098.9)
        resolution = 10

        snapped = _snap_bounds_to_grid(raw_utm_bbox, resolution)
        transform, width, height, _ = _target_grid(snapped, resolution, 32750)

        assert transform.c % resolution == 0
        assert transform.f % resolution == 0
        assert (snapped[2] - snapped[0]) % resolution == 0
        assert (snapped[3] - snapped[1]) % resolution == 0
        assert width == int((snapped[2] - snapped[0]) / resolution)
        assert height == int((snapped[3] - snapped[1]) / resolution)

    def test_snap_never_shrinks_extent(self):
        raw = (390_003.7, 6_460_002.1, 390_096.4, 6_460_098.9)

        snapped = _snap_bounds_to_grid(raw, 10)

        assert snapped[0] <= raw[0]
        assert snapped[1] <= raw[1]
        assert snapped[2] >= raw[2]
        assert snapped[3] >= raw[3]


class TestSnapToSourceGridRequest:
    def test_default_is_false(self):
        request = MosaicRequest(bounds=(0, 0, 100, 100), start_year=2023)

        assert request.snap_to_source_grid is False

    def test_can_be_enabled(self):
        request = MosaicRequest(
            bounds=(0, 0, 100, 100), start_year=2023, snap_to_source_grid=True
        )

        assert request.snap_to_source_grid is True


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


class TestDensifyBboxToPolygon:
    def test_returns_polygon_with_densified_edges(self):
        poly = densify_bbox_to_polygon((0.0, 0.0, 10.0, 5.0), points_per_edge=11)

        assert isinstance(poly, Polygon)
        # 11 points/edge × 4 edges − 4 shared corners = 40 unique exterior vertices.
        assert len(poly.exterior.coords) - 1 == 40
        # Envelope must match the input bbox exactly.
        assert poly.bounds == (0.0, 0.0, 10.0, 5.0)

    def test_rejects_degenerate_bbox(self):
        with pytest.raises(ValueError, match="positive-area"):
            densify_bbox_to_polygon((0.0, 0.0, 0.0, 5.0))

    def test_reprojected_envelope_matches_transform_bounds(self):
        # The wide WA strip from the failing notebook: a lon/lat rectangle far
        # west of the target UTM central meridian. The polygon envelope must
        # agree with transform_bounds within metres so swapping the legacy
        # reproject_bbox path for the polygon path does not noticeably resize
        # the output GeoTIFF. (Polygon envelope can be slightly larger because
        # 21 points/edge captures bulge marginally finer than pyproj's default
        # transform_bounds densification.)
        bbox_4326 = (114.80, -32.35, 125.20, -31.75)
        densified = densify_bbox_to_polygon(bbox_4326)
        utm_poly = reproject_aoi(densified, 4326, 32751)

        legacy = reproject_bbox(bbox_4326, 4326, 32751)
        for got, want in zip(utm_poly.bounds, legacy, strict=True):
            assert abs(got - want) < 20.0  # << 160m pixel at the example resolution

    def test_same_crs_polygon_round_trip(self):
        # If input_crs == target_crs, reprojection is a no-op.
        # The densified polygon is preserved, so .bounds matches the input bbox.
        poly = densify_bbox_to_polygon((100_000.0, 200_000.0, 110_000.0, 210_000.0))
        assert reproject_aoi(poly, 32750, 32750).bounds == (
            100_000.0,
            200_000.0,
            110_000.0,
            210_000.0,
        )
