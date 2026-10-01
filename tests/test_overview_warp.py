"""Warped (bounds/AOI) reads use COG overviews like direct reads do.

GDAL's warper reads full-resolution source pixels whatever the output
resolution, so before this a 60 m bounds mosaic fetched every 10 m pixel
under it: a 320 m warp of a 30 km DEA window took 32 s, the same as 10 m,
against 0.2 s for the equivalent direct read. Bounds reads now open the
source at the overview matching the target grid, measured in the source CRS
so scenes from a neighbouring UTM zone get the right level too.
"""

from dataclasses import replace

import numpy as np
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from s2mosaic.geometry import overview_level_for_target
from s2mosaic.helpers import get_rasterio_resampling
from s2mosaic.pipelines.bounds_scl import _read_band_at_target_window
from s2mosaic.readers import make_bounds_tile_reader
from s2mosaic.sources import MPC

UTM50 = CRS.from_epsg(32750)
UTM51 = CRS.from_epsg(32751)
SIZE = 512
# Near the east edge of UTM 50 (the 120°E boundary at ~32°S), so the same
# ground is also expressible on a UTM 51 grid.
X0, Y1 = 760_000.0, 6_460_000.0

LOCAL_MPC = replace(MPC, sign=lambda href: href)


def _write(path, arr, crs=UTM50, x0=X0, y1=Y1, factors=(2, 4, 8, 16)):
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": crs,
        "transform": from_origin(x0, y1, 10, 10),
        "nodata": 0,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint16), 1)
        dst.build_overviews(list(factors), Resampling.average)
    return str(path)


def _checkerboard(tmp_path, name="board.tif"):
    """100/300 per pixel: full-res nearest reads keep both values, while the
    averaged overviews read a uniform 200, so the output shows its source."""
    rows, cols = np.indices((SIZE, SIZE))
    return _write(tmp_path / name, np.where((rows + cols) % 2, 300, 100))


def _gradient(tmp_path):
    """Smooth ramp with a nodata strip, so averaging changes little and the
    valid footprint is checkable."""
    _, cols = np.indices((SIZE, SIZE))
    arr = 1000 + 2 * cols
    arr[:, :32] = 0
    return _write(tmp_path / "ramp.tif", arr)


def _level(path, crs, res, x0=X0, y1=Y1, n=64):
    with rio.open(path) as src:
        return overview_level_for_target(src, crs, from_origin(x0, y1, res, res), n, n)


class TestOverviewLevelForTarget:
    @pytest.mark.parametrize(
        "res, expected",
        [
            (10, None),  # native: no overview
            (15, None),  # 1.5x: no overview fits under one pixel
            (20, 0),  # 2x
            (30, 0),  # 3x: 2x is the coarsest that fits
            (40, 1),  # 4x
            (80, 2),  # 8x
            (500, 3),  # beyond the coarsest: use it
        ],
    )
    def test_same_crs_matches_gdal_rule(self, tmp_path, res, expected):
        assert _level(_checkerboard(tmp_path), UTM50, res) == expected

    def test_no_overviews_means_full_resolution(self, tmp_path):
        path = _write(tmp_path / "flat.tif", np.full((64, 64), 5), factors=())
        assert _level(path, UTM50, 80) is None

    @pytest.mark.parametrize("res, expected", [(10, None), (25, 0), (100, 2)])
    def test_cross_zone_measures_footprint_in_source_crs(self, tmp_path, res, expected):
        path = _checkerboard(tmp_path)
        # The same ground on a UTM 51 grid.
        left, bottom, right, top = transform_bounds(
            UTM50, UTM51, X0, Y1 - SIZE * 10, X0 + SIZE * 10, Y1
        )
        assert _level(path, UTM51, res, x0=left, y1=top) == expected

    def test_cross_zone_choice_never_exceeds_the_footprint(self, tmp_path):
        # At 40 m the footprint is ~4 source pixels, give or take the zones'
        # scale difference; whatever level is chosen must not be coarser.
        path = _checkerboard(tmp_path)
        left, _, _, top = transform_bounds(
            UTM50, UTM51, X0, Y1 - SIZE * 10, X0 + SIZE * 10, Y1
        )
        level = _level(path, UTM51, 40, x0=left, y1=top)
        with rio.open(path) as src:
            assert src.overviews(1)[level] <= 4


def _bounds_read(path, asset, res, crs=UTM50, x0=X0, y1=Y1, n=None):
    n = n or (SIZE * 10) // res
    item = type(
        "Item", (), {"id": "s", "assets": {asset: type("A", (), {"href": path})}}
    )
    read_fn = make_bounds_tile_reader(
        items=[item],
        href_template=[(asset, 1)],
        source=LOCAL_MPC,
        bounds_target=(x0, y1 - n * res, x0 + n * res, y1),
        target_crs=crs.to_epsg(),
        user_transform=from_origin(x0, y1, res, res),
        width=n,
        height=n,
        resolution=res,
        resampling_method="nearest",
        prewarm=False,
    )
    try:
        return read_fn(0, 0, (0, 0, n, n))
    finally:
        read_fn.close()


class TestBoundsTileReader:
    def test_native_resolution_reads_full_resolution(self, tmp_path):
        out = _bounds_read(_checkerboard(tmp_path), "B04", 10)
        assert set(np.unique(out)) == {100, 300}

    def test_coarser_resolution_reads_the_overview(self, tmp_path):
        out = _bounds_read(_checkerboard(tmp_path), "B04", 20)
        assert set(np.unique(out)) == {200}

    def test_categorical_assets_stay_at_full_resolution(self, tmp_path):
        out = _bounds_read(_checkerboard(tmp_path), "SCL", 20)
        assert set(np.unique(out)) <= {100, 300}

    def test_cross_zone_overview_read_matches_full_resolution(self, tmp_path):
        path = _gradient(tmp_path)
        left, bottom, right, top = transform_bounds(
            UTM50, UTM51, X0, Y1 - SIZE * 10, X0 + SIZE * 10, Y1
        )
        res, n = 80, int((right - left) // 80)
        target = from_origin(left, top, res, res)
        with rio.open(path) as src:
            level = overview_level_for_target(src, UTM51, target, n, n)
        assert level is not None  # the fast path is actually taken

        fast = _bounds_read(path, "B04", res, crs=UTM51, x0=left, y1=top, n=n)
        full = _read_band_at_target_window(
            path,
            1,
            (left, top - n * res, left + n * res, top),
            UTM51,
            n,
            n,
            get_rasterio_resampling("nearest"),
        )
        valid_fast, valid_full = fast > 0, full > 0
        # Same footprint, allowing one output pixel along the edges where
        # the overview's coarser cells straddle the scene boundary.
        assert np.mean(valid_fast != valid_full) < 0.05
        both = valid_fast & valid_full
        rel = np.abs(fast[both].astype(float) - full[both]) / full[both]
        assert np.median(rel) < 0.01
        assert rel.max() < 0.05


class TestReadBandAtTargetWindow:
    def _read(self, path, use_overviews):
        return _read_band_at_target_window(
            path,
            1,
            (X0, Y1 - SIZE * 10, X0 + SIZE * 10, Y1),
            UTM50,
            SIZE // 2,
            SIZE // 2,
            get_rasterio_resampling("nearest"),
            use_overviews=use_overviews,
        )

    def test_default_stays_at_full_resolution(self, tmp_path):
        assert set(np.unique(self._read(_checkerboard(tmp_path), False))) <= {
            100,
            300,
        }

    def test_use_overviews_reads_the_matching_overview(self, tmp_path):
        assert set(np.unique(self._read(_checkerboard(tmp_path), True))) == {200}
