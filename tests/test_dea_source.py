"""Digital Earth Australia (Geoscience Australia) Sentinel-2 NBART source.

DEA differs from the two L2A providers in ways that each break a default
assumption: CQL2-only search, three per-satellite collections, ``s3://``
hrefs, int16 pixels with nodata -999, fmask in place of SCL, and ``odc:`` /
``sentinel:`` / ``fmask:`` properties in place of ``s2:`` ones.
"""

import logging
from dataclasses import replace
from datetime import date, datetime, timezone

import numpy as np
import pytest
import rasterio as rio
from pystac import Asset, Item
from pystac.item_collection import ItemCollection
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from s2mosaic import SOURCE_DEA
from s2mosaic.config import VALID_BANDS, validate_inputs
from s2mosaic.data_reader import get_full_band
from s2mosaic.masking import get_scl_masks
from s2mosaic.readers import (
    _build_output_profile,
    make_bounds_tile_reader,
    make_grid_tile_reader,
)
from s2mosaic.sources import (
    DEA,
    MPC,
    SEARCH_CQL2,
    get_source,
    normalise_signed_dn,
)
from s2mosaic.stac import (
    GOOD_DATA_PCT_COL,
    ITEM_COL,
    _extract_mgrs_tile,
    add_item_info,
    drop_unreadable_items,
    filter_latest_processing_baselines,
    query_to_cql2,
    search_for_items,
    search_params,
    sort_items,
)

SIZE = 8
DEA_NODATA = -999
FINAL = {"op": "=", "args": [{"property": "dea:dataset_maturity"}, "final"]}
TILE_50HMK = {"op": "=", "args": [{"property": "odc:region_code"}, "50HMK"]}

# DEA with signing stubbed out so local paths reach rasterio unchanged.
LOCAL_DEA = replace(DEA, sign=lambda href: href)


def _write(path, arr, dtype, nodata):
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "crs": "EPSG:32750",
        "transform": from_origin(400_000, 6_500_000 + SIZE * 10, 10, 10),
        "nodata": nodata,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
    return str(path)


def _nbart(path, arr):
    return _write(path, arr, "int16", DEA_NODATA)


def _dea_item(
    item_id="ga_s2bm_ard_3-2-1_50HMK_2023-01-01_final",
    *,
    datastrip="S2B_OPER_MSI_L1C_DS_2BPS_20230101T041812_S20230101T021513_N05.09",
    geometry=None,
    assets=None,
    **extra_props,
):
    """A DEA-shaped STAC item on 50HMK's grid (origin 399960, 6500020)."""
    geometry = geometry or mapping(box(115.9, -31.7, 117.0, -30.7))
    props = {
        "odc:region_code": "50HMK",
        "dea:dataset_maturity": "final",
        "sentinel:datastrip_id": datastrip,
        "sat:relative_orbit": 60,
        "proj:code": "EPSG:32750",
        "proj:shape": [10980, 10980],
        "proj:transform": [10.0, 0.0, 399960.0, 0.0, -10.0, 6500020.0],
        "fmask:cloud": 10.0,
        "fmask:cloud_shadow": 5.0,
        **extra_props,
    }
    item = Item(
        id=item_id,
        geometry=geometry,
        bbox=None,
        datetime=datetime(2023, 1, 1, 2, 26, 35, tzinfo=timezone.utc),
        properties=props,
    )
    base = "s3://dea-public-data/baseline/ga_s2bm_ard_3/50/HMK/2023/01/01/x/"
    for key, href in (assets or {"nbart_red": base + "nbart_red.tif"}).items():
        item.add_asset(key, Asset(href=href))
    return item


class TestDeaSourceDefinition:
    def test_registered_under_public_constant(self):
        assert SOURCE_DEA == "DEA"
        assert get_source("DEA") is DEA

    def test_searches_all_three_satellite_collections(self):
        assert DEA.collections == ("ga_s2am_ard_3", "ga_s2bm_ard_3", "ga_s2cm_ard_3")

    def test_every_servable_band_maps_to_an_nbart_asset(self):
        servable = VALID_BANDS - DEA.unsupported_bands
        for band in servable:
            assert DEA.asset_name(band).startswith("nbart_"), band

    def test_scl_cloud_mask_reads_fmask(self):
        assert DEA.asset_name("SCL") == "oa_fmask"

    def test_signing_rewrites_the_public_bucket_to_https(self):
        href = "s3://dea-public-data/baseline/ga_s2am_ard_3/50/HMK/x.tif"
        assert DEA.sign(href) == (
            "https://dea-public-data.s3.ap-southeast-2.amazonaws.com/"
            "baseline/ga_s2am_ard_3/50/HMK/x.tif"
        )

    def test_signing_leaves_other_hrefs_alone(self):
        assert DEA.sign("s3://other-bucket/x.tif") == "s3://other-bucket/x.tif"
        assert DEA.sign("https://example.org/x.tif") == "https://example.org/x.tif"

    def test_mgrs_filter_is_cql2_on_region_code(self):
        assert DEA.mgrs_query("50HMK") == TILE_50HMK

    def test_open_catalog_does_not_assert_query_conformance(self, monkeypatch):
        import pystac_client

        class _Client:
            def add_conforms_to(self, name):
                raise AssertionError(f"DEA must not assert {name}")

        monkeypatch.setattr(
            pystac_client.Client, "open", classmethod(lambda cls, *a, **k: _Client())
        )
        assert isinstance(DEA.open_catalog(stac_io=None), _Client)


class TestFmaskToScl:
    def test_fmask_classes_map_to_scl_equivalents(self):
        fmask = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        np.testing.assert_array_equal(DEA.to_scl(fmask), [0, 4, 9, 3, 11, 6])

    def test_undefined_fmask_codes_are_excluded_not_nodata(self):
        np.testing.assert_array_equal(
            DEA.to_scl(np.array([6, 200, 255], dtype=np.uint8)), [7, 7, 7]
        )

    def test_scl_sources_pass_through(self):
        scl = np.array([0, 4, 9], dtype=np.uint8)
        assert MPC.to_scl(scl) is scl

    def test_scl_masks_from_fmask(self, tmp_path):
        # Columns: nodata, clear, cloud, shadow, snow, water, clear, clear.
        fmask = np.tile(np.array([0, 1, 2, 3, 4, 5, 1, 1], dtype=np.uint8), (8, 1))
        href = _write(tmp_path / "fmask.tif", fmask, "uint8", 0)
        item = _dea_item(assets={"oa_fmask": href})
        # get_full_band's grid read targets a 10980 px tile; pass the 10 m
        # resolution scaled so the output is the file's own 8 px.
        clear, valid = get_scl_masks(item, LOCAL_DEA, user_resolution=10980 * 10 // 8)
        np.testing.assert_array_equal(
            clear[0], [True, True, False, False, True, True, True, True]
        )
        assert not valid[0, 0]


class TestNormaliseSignedDn:
    def test_nodata_zero_and_negative_values(self):
        arr = np.array([[-999, -5, 0, 1, 1234]], dtype=np.int16)
        out = normalise_signed_dn(arr, -999)
        np.testing.assert_array_equal(out, [[0, 1, 1, 1, 1234]])
        assert out.dtype == np.uint16

    def test_unsigned_arrays_are_untouched(self):
        arr = np.array([[0, 1500]], dtype=np.uint16)
        assert normalise_signed_dn(arr, 0) is arr

    def test_signed_without_declared_nodata_keeps_everything_valid(self):
        out = normalise_signed_dn(np.array([[-999, 7]], dtype=np.int16), None)
        np.testing.assert_array_equal(out, [[1, 7]])


class TestDeaPixelReads:
    """int16/-999 rasters come out as uint16 with 0 as nodata on every path."""

    def _arr(self):
        arr = np.full((SIZE, SIZE), 700, dtype=np.int16)
        arr[0, :] = DEA_NODATA
        arr[1, :] = -12  # negative NBART: a real, dark observation
        return arr

    def _check(self, out):
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out[0], 0)
        np.testing.assert_array_equal(out[1], 1)
        np.testing.assert_array_equal(out[2:], 700)

    def test_grid_tile_reader(self, tmp_path):
        item = _dea_item(assets={"nbart_red": _nbart(tmp_path / "r.tif", self._arr())})
        read_fn = make_grid_tile_reader(
            items=[item],
            href_template=[("B04", 1)],
            source=LOCAL_DEA,
            s2_scene_size=SIZE,
            resolution=10,
            resampling_method="nearest",
            prewarm=False,
        )
        try:
            self._check(read_fn(0, 0, (0, 0, SIZE, SIZE)))
        finally:
            read_fn.close()

    def test_bounds_tile_reader_including_outside_footprint(self, tmp_path):
        item = _dea_item(assets={"nbart_red": _nbart(tmp_path / "r.tif", self._arr())})
        # One extra column east of the raster: WarpedVRT fills it with the
        # source nodata, -999, which must also land as 0.
        read_fn = make_bounds_tile_reader(
            items=[item],
            href_template=[("B04", 1)],
            source=LOCAL_DEA,
            bounds_target=(400_000, 6_500_000, 400_090, 6_500_080),
            target_crs=32750,
            user_transform=from_origin(400_000, 6_500_000 + SIZE * 10, 10, 10),
            width=SIZE + 1,
            height=SIZE,
            resolution=10,
            resampling_method="nearest",
            prewarm=False,
        )
        try:
            out = read_fn(0, 0, (0, 0, SIZE, SIZE + 1))
        finally:
            read_fn.close()
        self._check(out[:, :SIZE])
        np.testing.assert_array_equal(out[:, SIZE], 0)

    def test_full_band_read_used_for_ocm(self, tmp_path):
        href = _nbart(tmp_path / "r.tif", self._arr())
        arr, _ = get_full_band(href, LOCAL_DEA, res=10980 * 10 // 8, asset_name="B04")
        self._check(arr[0])

    def test_grid_output_profile_reports_zero_nodata(self, tmp_path):
        href = _nbart(tmp_path / "r.tif", self._arr())
        assert _build_output_profile(href, SIZE)["nodata"] == 0


class TestQueryToCql2:
    def test_translates_each_operator(self):
        query = {
            "a": {"eq": 1},
            "b": {"neq": 2},
            "c": {"lt": 3, "gte": 0},
            "d": {"lte": 4},
            "e": {"gt": 5},
            "f": {"in": ["x", "y"]},
        }
        assert query_to_cql2(query) == [
            {"op": "=", "args": [{"property": "a"}, 1]},
            {"op": "<>", "args": [{"property": "b"}, 2]},
            {"op": "<", "args": [{"property": "c"}, 3]},
            {"op": ">=", "args": [{"property": "c"}, 0]},
            {"op": "<=", "args": [{"property": "d"}, 4]},
            {"op": ">", "args": [{"property": "e"}, 5]},
            {"op": "in", "args": [{"property": "f"}, ["x", "y"]]},
        ]

    def test_unknown_operator_raises_rather_than_dropping_the_filter(self):
        with pytest.raises(ValueError, match="startsWith"):
            query_to_cql2({"platform": {"startsWith": "sentinel"}})

    def test_bare_value_raises(self):
        with pytest.raises(ValueError, match="must map operators"):
            query_to_cql2({"eo:cloud_cover": 50})


class TestSearchParams:
    CLOUD = {"eo:cloud_cover": {"lt": 50}}
    CLOUD_CQL2 = {"op": "<", "args": [{"property": "eo:cloud_cover"}, 50]}

    def test_dea_grid_search_ands_final_tile_and_cloud(self):
        params = search_params(DEA, DEA.mgrs_query("50HMK"), self.CLOUD)
        assert params == {
            "filter": {"op": "and", "args": [FINAL, TILE_50HMK, self.CLOUD_CQL2]},
            "filter_lang": SEARCH_CQL2,
        }
        assert "query" not in params

    def test_dea_bounds_search_without_extra_query_is_final_only(self):
        assert search_params(DEA, None, None) == {
            "filter": FINAL,
            "filter_lang": SEARCH_CQL2,
        }

    def test_query_sources_are_unchanged(self):
        assert search_params(MPC, MPC.mgrs_query("50HMK"), self.CLOUD) == {
            "query": {"s2:mgrs_tile": {"eq": "50HMK"}, **self.CLOUD}
        }
        assert search_params(MPC, None, None) == {}


class TestDeaSearchRequests:
    """The kwargs that actually reach ``Client.search``."""

    @staticmethod
    def _capture(monkeypatch):
        captured: dict = {}

        class _Search:
            def item_collection(self):
                return ItemCollection([])

        class _Catalog:
            def search(self, **kwargs):
                captured.update(kwargs)
                return _Search()

        import pystac_client

        monkeypatch.setattr(
            pystac_client.Client, "open", classmethod(lambda cls, *a, **k: _Catalog())
        )
        return captured

    def test_grid_search(self, monkeypatch):
        captured = self._capture(monkeypatch)
        search_for_items(
            grid_id="50HMK",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 1),
            additional_query={"eo:cloud_cover": {"lt": 100}},
            source=DEA,
        )
        assert captured["collections"] == list(DEA.collections)
        assert captured["filter_lang"] == SEARCH_CQL2
        assert TILE_50HMK in captured["filter"]["args"]
        assert FINAL in captured["filter"]["args"]
        assert "query" not in captured
        assert "intersects" not in captured

    def test_bounds_search(self, monkeypatch):
        import s2mosaic.stac_bounds as stac_bounds

        captured = self._capture(monkeypatch)
        stac_bounds._search_for_items_by_bbox(
            bbox_4326=(116.0, -32.0, 116.1, -31.9),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 1),
            source=DEA,
            additional_query={"eo:cloud_cover": {"lt": 100}},
        )
        assert captured["bbox"] == [116.0, -32.0, 116.1, -31.9]
        assert captured["filter"]["op"] == "and"
        assert FINAL in captured["filter"]["args"]
        assert "query" not in captured


class TestDeaItemProperties:
    def test_tile_comes_from_region_code(self):
        assert _extract_mgrs_tile({"odc:region_code": "50HMK"}) == "50HMK"

    def test_dedup_keys_on_sentinel_datastrip_id(self):
        # Two granules of one datatake on one tile (the 50HMH 2019-11-23
        # case) share a datetime but not a datastrip sensing start.
        a = _dea_item(
            "a",
            datastrip="S2A_OPER_MSI_L1C_DS_X_20191123T040000_S20191123T022659_N02.08",
        )
        b = _dea_item(
            "b",
            datastrip="S2A_OPER_MSI_L1C_DS_X_20191123T040000_S20191123T022701_N02.08",
        )
        kept = filter_latest_processing_baselines(ItemCollection([a, b]))
        assert {it.id for it in kept} == {"a", "b"}

    def test_s3_hrefs_in_the_public_bucket_are_readable(self):
        item = _dea_item()
        kept = drop_unreadable_items(ItemCollection([item]), DEA, assets=["B04"])
        assert [it.id for it in kept] == [item.id]

    def test_good_data_uses_fmask_cloud_and_footprint_nodata(self):
        # Footprint covering the western half of 50HMK's grid.
        west_half = box(399960, 6500020 - 109800, 399960 + 54900, 6500020)
        import pyproj
        from shapely.ops import transform

        to_4326 = pyproj.Transformer.from_crs(32750, 4326, always_xy=True).transform
        item = _dea_item(geometry=mapping(transform(to_4326, west_half)))
        good = add_item_info(ItemCollection([item]))[GOOD_DATA_PCT_COL].iloc[0]
        # 50% data x (1 - 15% fmask cloud + shadow).
        assert good == pytest.approx(50 * 0.85, abs=0.1)

    def test_published_nodata_percentage_still_wins(self):
        item = _dea_item(**{"s2:nodata_pixel_percentage": 20.0})
        good = add_item_info(ItemCollection([item]))[GOOD_DATA_PCT_COL].iloc[0]
        assert good == pytest.approx(80 * 0.85)

    # Issue #16: DEA serialises an fmask statistic it couldn't compute as the
    # string "NaN". One such item used to crash the whole period.
    @pytest.mark.parametrize("bad", ["NaN", float("nan"), None, "inf"])
    def test_unknown_fmask_stats_rank_as_fully_cloudy(self, bad, caplog):
        item = _dea_item("bad-stats", **{"fmask:cloud": bad, "fmask:cloud_shadow": bad})

        with caplog.at_level(logging.WARNING, logger="s2mosaic.stac"):
            good = add_item_info(ItemCollection([item]))[GOOD_DATA_PCT_COL].iloc[0]

        assert good == 0.0
        assert "bad-stats" in caplog.text

    def test_one_unknown_fmask_stat_is_enough_to_score_as_cloudy(self):
        item = _dea_item(**{"fmask:cloud": 10.0, "fmask:cloud_shadow": "NaN"})
        good = add_item_info(ItemCollection([item]))[GOOD_DATA_PCT_COL].iloc[0]
        assert good == 0.0

    def test_unknown_stats_sort_after_known_scenes_of_the_same_orbit(self):
        # eo:cloud_cover is 0.0 on the real item, so it is no safe fallback.
        unknown = _dea_item(
            "unknown",
            **{
                "fmask:cloud": "NaN",
                "fmask:cloud_shadow": "NaN",
                "eo:cloud_cover": 0.0,
            },
        )
        cloudy = _dea_item("cloudy", **{"fmask:cloud": 90.0, "fmask:cloud_shadow": 5.0})
        clear = _dea_item("clear", **{"fmask:cloud": 0.0, "fmask:cloud_shadow": 0.0})

        df = add_item_info(ItemCollection([unknown, cloudy, clear]))
        ordered = sort_items(df, scene_order="valid_data")

        assert [it.id for it in ordered[ITEM_COL]] == ["clear", "cloudy", "unknown"]

    def test_missing_fmask_stats_still_count_as_no_cloud(self):
        item = _dea_item(**{"s2:nodata_pixel_percentage": 0.0})
        del item.properties["fmask:cloud"]
        del item.properties["fmask:cloud_shadow"]
        good = add_item_info(ItemCollection([item]))[GOOD_DATA_PCT_COL].iloc[0]
        assert good == pytest.approx(100.0)

    def test_non_numeric_published_nodata_falls_back_to_footprint(self):
        west_half = box(399960, 6500020 - 109800, 399960 + 54900, 6500020)
        import pyproj
        from shapely.ops import transform

        to_4326 = pyproj.Transformer.from_crs(32750, 4326, always_xy=True).transform
        item = _dea_item(
            geometry=mapping(transform(to_4326, west_half)),
            **{"s2:nodata_pixel_percentage": "NaN"},
        )
        good = add_item_info(ItemCollection([item]))[GOOD_DATA_PCT_COL].iloc[0]
        assert good == pytest.approx(50 * 0.85, abs=0.1)


class TestDeaBandValidation:
    BASE = {
        "scene_order": "valid_data",
        "mosaic_method": "mean",
        "min_observations": None,
        "grid_id": "50HMK",
        "percentile": None,
        "source": "DEA",
    }

    @pytest.mark.parametrize("band", ["visual", "SCL", "AOT", "WVP", "B09"])
    def test_rejects_bands_dea_does_not_serve(self, band):
        with pytest.raises(ValueError, match=f"Band {band} is not available"):
            validate_inputs(**self.BASE, bands=[band])

    def test_accepts_nbart_bands(self):
        validate_inputs(
            **self.BASE, bands=["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
        )

    def test_other_sources_still_serve_visual(self):
        validate_inputs(**{**self.BASE, "source": "MPC"}, bands=["visual"])


@pytest.mark.slow
class TestDeaLiveSearch:
    """Against the real DEA STAC API: the filter must actually restrict."""

    def test_grid_search_returns_only_final_items_for_the_tile(self):
        items = search_for_items(
            grid_id="50HMH",
            start_date=date(2025, 2, 1),
            end_date=date(2025, 3, 1),
            additional_query={"eo:cloud_cover": {"lt": 100}},
            source=DEA,
        )
        # Feb 2025 on 50HMH also has an nrt and an interim item.
        assert len(items) >= 10
        assert {it.properties["odc:region_code"] for it in items} == {"50HMH"}
        assert {it.properties["dea:dataset_maturity"] for it in items} == {"final"}

    def test_bounds_mosaic_runs_end_to_end(self):
        from s2mosaic import mosaic

        arr, profile = mosaic(
            bounds=(389410, 6462290, 390410, 6463290),
            input_crs=32750,
            snap_to_source_grid=True,
            start_year=2023,
            start_month=1,
            duration_months=1,
            bands=["B04", "B03", "B02", "B08"],
            mosaic_method="median",
            cloud_mask="SCL",
            source="DEA",
        )
        assert arr.dtype == np.uint16
        valid = arr[:, arr.min(axis=0) > 0]
        assert valid.shape[1] > 0.9 * arr.shape[1] * arr.shape[2]
        # NBART reflectance * 10000: no +1000 offset, no -999 wrap.
        assert 100 < np.median(valid[0]) < 3000
        assert valid.max() < 20000
