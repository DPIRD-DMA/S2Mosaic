"""Processing baseline 04.00 BOA offset: harmonise DNs across baselines.

From baseline 04.00 (25 Jan 2022) ESA writes L2A reflectance as
``DN = reflectance * 10000 + 1000``; earlier baselines have no offset.
Microsoft Planetary Computer serves those DNs as-is, so without correction
a mosaic spanning the change stacks scenes on two scales and every MPC
mosaic after 2022 reads ~1000 DN higher than the same scene from Element
84, which has already removed the offset (``earthsearch:boa_offset_applied``).
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from s2mosaic.pipelines.grid import stream_mosaic_pipeline
from s2mosaic.readers import make_bounds_tile_reader, make_grid_tile_reader
from s2mosaic.sources import AWS, MPC, apply_boa_offset, boa_add_offset
from s2mosaic.stac import ITEM_COL

SIZE = 8
# Same surface reflectance (0.05) under each baseline's encoding.
PRE_04_DN = 500
POST_04_DN = 1500
# Negative reflectance (-0.02) under the 04.00 encoding: a real, valid
# observation over dark water that must not become nodata.
DARK_POST_04_DN = 800


def _identity(href):
    return href


# MPC with signing stubbed out so local paths reach rasterio unchanged.
LOCAL_MPC = replace(MPC, sign=_identity)


def _write_band(path, arr):
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": "EPSG:32750",
        "transform": from_origin(400_000, 6_500_000 + SIZE * 10, 10, 10),
        "nodata": 0,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint16), 1)
    return str(path)


def _item(item_id, href, baseline, asset="B04", **extra_props):
    return SimpleNamespace(
        id=item_id,
        assets={asset: SimpleNamespace(href=href)},
        properties={"s2:processing_baseline": baseline, **extra_props},
    )


class TestBoaAddOffset:
    @pytest.mark.parametrize("baseline", ["04.00", "05.09", "05.11"])
    def test_reflectance_bands_carry_offset_from_baseline_04(self, baseline):
        props = {"s2:processing_baseline": baseline}
        for band in ["B01", "B02", "B04", "B08", "B8A", "B09", "B11", "B12"]:
            assert boa_add_offset(props, band) == -1000

    @pytest.mark.parametrize("baseline", ["02.12", "03.01", "03.99"])
    def test_older_baselines_have_no_offset(self, baseline):
        assert boa_add_offset({"s2:processing_baseline": baseline}, "B04") == 0

    @pytest.mark.parametrize("asset", ["SCL", "AOT", "WVP", "visual"])
    def test_non_reflectance_assets_have_no_offset(self, asset):
        assert boa_add_offset({"s2:processing_baseline": "05.09"}, asset) == 0

    def test_offset_already_removed_by_provider(self):
        props = {
            "s2:processing_baseline": "05.09",
            "earthsearch:boa_offset_applied": True,
        }
        assert boa_add_offset(props, "B04") == 0

    def test_provider_flag_false_keeps_offset(self):
        props = {
            "s2:processing_baseline": "05.09",
            "earthsearch:boa_offset_applied": False,
        }
        assert boa_add_offset(props, "B04") == -1000

    @pytest.mark.parametrize("props", [{}, {"s2:processing_baseline": "bogus"}])
    def test_missing_or_bad_baseline_has_no_offset(self, props):
        assert boa_add_offset(props, "B04") == 0


class TestApplyBoaOffset:
    def test_shifts_valid_pixels(self):
        arr = np.array([[1500, 1001, 11000]], dtype=np.uint16)
        out = apply_boa_offset(arr, -1000)
        np.testing.assert_array_equal(out, [[500, 1, 10000]])
        assert out.dtype == np.uint16

    def test_nodata_stays_zero(self):
        arr = np.array([[0, 1500]], dtype=np.uint16)
        np.testing.assert_array_equal(apply_boa_offset(arr, -1000), [[0, 500]])

    def test_negative_reflectance_clips_to_one_not_nodata_or_wraparound(self):
        # DN 1..1000 is reflectance <= 0. It is a real observation: mapping it
        # to 0 would make it nodata, and uint16 subtraction would wrap it to
        # ~65000 and dominate every mean it enters.
        arr = np.array([[1, 800, 1000]], dtype=np.uint16)
        np.testing.assert_array_equal(apply_boa_offset(arr, -1000), [[1, 1, 1]])

    def test_zero_offset_returns_input_unchanged(self):
        arr = np.array([[0, 800, 1500]], dtype=np.uint16)
        assert apply_boa_offset(arr, 0) is arr


class TestGridMosaicHarmonisesBaselines:
    """End to end through the real grid tile reader and aggregation."""

    def _scenes(self, tmp_path):
        old = np.full((SIZE, SIZE), PRE_04_DN, dtype=np.uint16)
        new = np.full((SIZE, SIZE), POST_04_DN, dtype=np.uint16)
        # Row 0: nodata in the new scene only. Row 1: nodata in both.
        # Row 2: the new scene is dark water with negative reflectance.
        new[0, :] = 0
        old[1, :] = 0
        new[1, :] = 0
        new[2, :] = DARK_POST_04_DN
        old_item = _item(
            "old", _write_band(tmp_path / "old.tif", old), baseline="02.12"
        )
        new_item = _item(
            "new", _write_band(tmp_path / "new.tif", new), baseline="04.00"
        )
        return old_item, new_item

    def _run(self, monkeypatch, items, mosaic_method):
        import s2mosaic.pipelines.grid as grid_mod

        monkeypatch.setattr(
            grid_mod,
            "_compute_one_scene_mask",
            lambda **_: np.ones((SIZE, SIZE), dtype=bool),
        )
        out, _profile, dropped = stream_mosaic_pipeline(
            sorted_scenes=pd.DataFrame({ITEM_COL: items}),
            bands=["B04"],
            coverage_mask=np.ones((SIZE, SIZE), dtype=bool),
            mosaic_method=mosaic_method,
            cloud_mask="SCL",
            source=LOCAL_MPC,
            s2_scene_size=SIZE,
            tile_size=SIZE,
            tile_workers=1,
        )
        assert dropped == []
        return out[0]

    def test_mean_across_baseline_change_is_not_biased(self, monkeypatch, tmp_path):
        old_item, new_item = self._scenes(tmp_path)
        out = self._run(monkeypatch, [old_item, new_item], "mean")
        # Both scenes observe reflectance 0.05; the mean must too.
        np.testing.assert_array_equal(out[3:], PRE_04_DN)
        # Only the old scene is valid here.
        np.testing.assert_array_equal(out[0], PRE_04_DN)
        # Neither scene is valid.
        np.testing.assert_array_equal(out[1], 0)
        # Mean of 500 and the clipped dark pixel (1), not of 500 and 800 or
        # of 500 and a wrapped ~65000.
        np.testing.assert_array_equal(out[2], (PRE_04_DN + 1) // 2)

    def test_first_from_post_04_scene_matches_pre_04_scale(self, monkeypatch, tmp_path):
        old_item, new_item = self._scenes(tmp_path)
        out = self._run(monkeypatch, [new_item, old_item], "first")
        np.testing.assert_array_equal(out[3:], PRE_04_DN)
        # The dark pixel is a valid first observation, so it wins over the
        # later scene rather than being skipped as nodata.
        np.testing.assert_array_equal(out[2], 1)

    def test_grid_reader_leaves_aws_pixels_alone(self, tmp_path):
        arr = np.full((SIZE, SIZE), PRE_04_DN, dtype=np.uint16)
        item = _item(
            "aws",
            _write_band(tmp_path / "aws.tif", arr),
            baseline="05.09",
            asset="red",
            **{"earthsearch:boa_offset_applied": True},
        )
        read_fn = make_grid_tile_reader(
            items=[item],
            href_template=[("B04", 1)],
            source=AWS,
            s2_scene_size=SIZE,
            resolution=10,
            resampling_method="nearest",
            prewarm=False,
        )
        try:
            np.testing.assert_array_equal(read_fn(0, 0, (0, 0, SIZE, SIZE)), PRE_04_DN)
        finally:
            read_fn.close()


class TestBoundsReaderHarmonisesBaselines:
    def _read(self, item, asset="B04"):
        read_fn = make_bounds_tile_reader(
            items=[item],
            href_template=[(asset, 1)],
            source=LOCAL_MPC,
            bounds_target=(400_000, 6_500_000, 400_000 + SIZE * 10, 6_500_080),
            target_crs=32750,
            user_transform=from_origin(400_000, 6_500_000 + SIZE * 10, 10, 10),
            width=SIZE,
            height=SIZE,
            resolution=10,
            resampling_method="nearest",
            prewarm=False,
        )
        try:
            return read_fn(0, 0, (0, 0, SIZE, SIZE))
        finally:
            read_fn.close()

    def test_post_04_scene_is_shifted_and_nodata_kept(self, tmp_path):
        arr = np.full((SIZE, SIZE), POST_04_DN, dtype=np.uint16)
        arr[0, :] = 0
        arr[1, :] = DARK_POST_04_DN
        out = self._read(
            _item("new", _write_band(tmp_path / "new.tif", arr), baseline="05.09")
        )
        np.testing.assert_array_equal(out[0], 0)
        np.testing.assert_array_equal(out[1], 1)
        np.testing.assert_array_equal(out[2:], PRE_04_DN)

    def test_pre_04_scene_is_unchanged(self, tmp_path):
        arr = np.full((SIZE, SIZE), PRE_04_DN, dtype=np.uint16)
        out = self._read(
            _item("old", _write_band(tmp_path / "old.tif", arr), baseline="02.12")
        )
        np.testing.assert_array_equal(out, PRE_04_DN)

    def test_scl_is_never_shifted(self, tmp_path):
        arr = np.full((SIZE, SIZE), 4, dtype=np.uint16)
        out = self._read(
            _item(
                "scl",
                _write_band(tmp_path / "scl.tif", arr),
                baseline="05.09",
                asset="SCL",
            ),
            asset="SCL",
        )
        np.testing.assert_array_equal(out, 4)
