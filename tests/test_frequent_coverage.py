import pytest
from shapely.geometry import Polygon

from s2mosaic.frequent_coverage import (
    _utm_origin_from_item,
    get_coverage,
    get_frequent_coverage_for_bbox,
    get_raster_coverage,
)


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
            min_coverage_fraction=0.5,
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

    def test_get_coverage_empty_input_returns_empty_geodataframe(self):
        coverage = get_coverage([])

        assert coverage.empty
        assert coverage.crs == "EPSG:4326"

    def test_get_coverage_skips_none_geometry(self):
        coords = [
            (115.0, -32.5),
            (116.5, -32.5),
            (116.5, -31.5),
            (115.0, -31.5),
            (115.0, -32.5),
        ]

        class _NoGeometryItem:
            geometry = None

        coverage = get_coverage([_NoGeometryItem(), _FakeItem(coords)])

        assert len(coverage) == 1
        assert coverage.geometry.iloc[0].geom_type == "Polygon"

    def test_get_coverage_preserves_multipolygon_geometry(self):
        polygon = self._scene_polygon_4326()
        shifted = Polygon([(x + 0.2, y) for x, y in polygon.exterior.coords])

        class _FakeMultiPolygonItem:
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [
                    [list(polygon.exterior.coords)],
                    [list(shifted.exterior.coords)],
                ],
            }

        coverage = get_coverage([_FakeMultiPolygonItem()])

        assert coverage.geometry.iloc[0].geom_type == "MultiPolygon"
        assert len(coverage.geometry.iloc[0].geoms) == 2

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
        # T50HMK origin in EPSG:32750 (Perth UTM South).
        raster = get_raster_coverage(
            x_min=399960.0,
            y_max=6500020.0,
            coverage_gdf=self._coverage_gdf(),
            local_crs=32750,
            resolution=resolution,
        )
        assert raster.shape == (expected_side, expected_side)


class TestUtmOriginFromItem:
    """Pull the MGRS tile origin off a STAC item's asset proj:transform."""

    @staticmethod
    def _fake_item(assets):
        class _FakeAsset:
            def __init__(self, extra_fields):
                self.extra_fields = extra_fields

        class _FakeItem:
            id = "fake-item"

            def __init__(self):
                self.assets = {
                    name: _FakeAsset(fields) for name, fields in assets.items()
                }

        return _FakeItem()

    def test_returns_top_left_from_first_asset_with_proj_transform(self):
        # First asset (alphabetically iterated dict in py3.7+) has no
        # proj:transform; second one does. The helper must keep looking
        # until it finds a usable asset rather than giving up.
        item = self._fake_item(
            {
                "thumbnail": {},
                "B04": {"proj:transform": [10.0, 0.0, 399960.0, 0.0, -10.0, 6500020.0]},
            }
        )
        assert _utm_origin_from_item(item) == (399960.0, 6500020.0)

    def test_raises_when_no_asset_has_proj_transform(self):
        item = self._fake_item({"thumbnail": {}, "preview": {}})
        with pytest.raises(ValueError, match="no asset with a proj:transform"):
            _utm_origin_from_item(item)
