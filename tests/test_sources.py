"""Tests for the multi-source (MPC / AWS) abstraction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from s2mosaic import SOURCE_AWS, SOURCE_MPC, mosaic
from s2mosaic.config import validate_inputs
from s2mosaic.sources import AWS, MPC, VALID_SOURCES, Source, get_source
from s2mosaic.stac import STAC_RETRY_STATUS_CODES


class TestSourceConstants:
    def test_mpc_and_aws_exposed_as_constants(self):
        assert SOURCE_MPC == "MPC"
        assert SOURCE_AWS == "AWS"

    def test_valid_sources_contains_both(self):
        assert SOURCE_MPC in VALID_SOURCES
        assert SOURCE_AWS in VALID_SOURCES
        assert len(VALID_SOURCES) == 2

    def test_get_source_returns_singleton(self):
        assert get_source(SOURCE_MPC) is MPC
        assert get_source(SOURCE_AWS) is AWS

    def test_get_source_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown source"):
            get_source("GCS")
        with pytest.raises(ValueError, match="Unknown source"):
            get_source("mpc")  # case-sensitive


class TestSourceAssetMap:
    """Canonical band/asset names map to provider-specific keys."""

    def test_mpc_passes_canonical_names_through(self):
        for canonical in ["B02", "B03", "B04", "B08", "B8A", "SCL", "visual"]:
            assert MPC.asset_name(canonical) == canonical

    def test_aws_remaps_band_ids_to_common_names(self):
        assert AWS.asset_name("B02") == "blue"
        assert AWS.asset_name("B03") == "green"
        assert AWS.asset_name("B04") == "red"
        assert AWS.asset_name("B08") == "nir"
        assert AWS.asset_name("B8A") == "nir08"

    def test_aws_lowercases_scl(self):
        assert AWS.asset_name("SCL") == "scl"

    def test_aws_keeps_visual_unchanged(self):
        # The TCI/visual asset uses the same key on both providers.
        assert AWS.asset_name("visual") == "visual"

    def test_aws_falls_through_for_unmapped_keys(self):
        # An unmapped canonical name returns itself; callers see provider errors
        # downstream rather than silent renames.
        assert AWS.asset_name("anything_else") == "anything_else"


class TestSourceSigning:
    def test_aws_signing_is_identity(self):
        href = "https://example.com/scene/B04.tif"
        assert AWS.sign(href) == href

    def test_mpc_signing_delegates_to_planetary_computer(self, monkeypatch):
        captured = []

        def fake_sign(href):
            captured.append(href)
            return f"{href}?sig=signed"

        monkeypatch.setattr("planetary_computer.sign", fake_sign)
        result = MPC.sign("https://mpc.example/foo.tif")
        assert captured == ["https://mpc.example/foo.tif"]
        assert result == "https://mpc.example/foo.tif?sig=signed"


class TestMgrsQuery:
    def test_mpc_uses_single_property_filter(self):
        q = MPC.mgrs_query("50HMH")
        assert q == {"s2:mgrs_tile": {"eq": "50HMH"}}

    def test_aws_disables_server_side_mgrs_filter(self):
        # Element 84 returns 0 items when ``query`` is combined with
        # ``intersects``/``bbox`` (live-API verified). The AWS source
        # therefore disables its server-side MGRS query; precision is
        # restored by client-side post-filtering on ``grid:code``.
        assert AWS.mgrs_query("50HMH") is None

    def test_aws_split_field_builder_is_well_formed(self):
        # The split-field builder is still importable in case Earth Search
        # ever fixes the ``query``+``intersects`` interaction; keep
        # coverage so the construction logic doesn't bit-rot.
        from s2mosaic.sources import _aws_mgrs_query

        assert _aws_mgrs_query("50HMH") == {
            "mgrs:utm_zone": {"eq": 50},
            "mgrs:latitude_band": {"eq": "H"},
            "mgrs:grid_square": {"eq": "MH"},
        }


class TestSourceCatalogConfig:
    def test_mpc_points_at_planetary_computer(self):
        assert MPC.stac_url == "https://planetarycomputer.microsoft.com/api/stac/v1"
        assert MPC.collection_id == "sentinel-2-l2a"

    def test_aws_points_at_earth_search_l2a(self):
        # Element 84 hosts both L1C and L2A; we explicitly use L2A.
        assert AWS.stac_url == "https://earth-search.aws.element84.com/v1"
        assert AWS.collection_id == "sentinel-2-l2a"

    def test_stac_retry_policy_includes_rate_limit_and_server_errors(self):
        assert {408, 429, 500, 502, 503, 504}.issubset(STAC_RETRY_STATUS_CODES)


class TestValidateInputsRejectsBadSource:
    """validate_inputs should reject sources not in VALID_SOURCES."""

    BASE_KWARGS = {
        "scene_order": "valid_data",
        "mosaic_method": "mean",
        "early_stop_missing_fraction": None,
        "min_observations": None,
        "bands": ["B04"],
        "grid_id": "50HMH",
        "percentile": None,
    }

    def test_rejects_unknown_source(self):
        with pytest.raises(ValueError, match="Invalid source"):
            validate_inputs(**self.BASE_KWARGS, source="GCS")

    def test_accepts_both_known_sources(self):
        # Neither call should raise.
        validate_inputs(**self.BASE_KWARGS, source="MPC")
        validate_inputs(**self.BASE_KWARGS, source="AWS")


class TestMosaicPublicApiSource:
    def test_mosaic_signature_has_source_param_defaulting_to_mpc(self):
        import inspect

        sig = inspect.signature(mosaic)
        assert "source" in sig.parameters
        assert sig.parameters["source"].default == SOURCE_MPC

    def test_mosaic_rejects_invalid_source_string(self):
        with pytest.raises(ValueError, match="GCS"):
            mosaic(grid_id="50HMH", start_year=2023, source="GCS")


class TestSourceThreadsThroughBoundsPipeline:
    """End-to-end smoke: source flows from mosaic() into the STAC search call."""

    def test_aws_source_reaches_bounds_search(self, monkeypatch):
        captured = {}

        def fake_search(*, bbox_4326, start_date, end_date, source, **_):
            captured["source"] = source
            captured["geometry"] = bbox_4326
            captured["start_date"] = start_date
            captured["end_date"] = end_date
            raise RuntimeError("stop-here")  # short-circuit further work

        import s2mosaic.pipelines.bounds as bounds_mod

        monkeypatch.setattr(bounds_mod, "_search_for_items_by_bbox", fake_search)

        with pytest.raises(RuntimeError, match="stop-here"):
            mosaic(
                bounds=(115.83, -31.97, 115.91, -31.94),
                start_year=2023,
                duration_days=1,
                bands=["B04"],
                source=SOURCE_AWS,
            )

        assert captured["source"] is AWS

    def test_mpc_source_reaches_bounds_search_by_default(self, monkeypatch):
        captured = {}

        def fake_search(*, bbox_4326, start_date, end_date, source, **_):
            captured["source"] = source
            raise RuntimeError("stop-here")

        import s2mosaic.pipelines.bounds as bounds_mod

        monkeypatch.setattr(bounds_mod, "_search_for_items_by_bbox", fake_search)

        with pytest.raises(RuntimeError, match="stop-here"):
            mosaic(
                bounds=(115.83, -31.97, 115.91, -31.94),
                start_year=2023,
                duration_days=1,
                bands=["B04"],
            )

        assert captured["source"] is MPC


class TestSourceThreadsThroughGridPipeline:
    """End-to-end smoke: source flows from mosaic() into the grid STAC search."""

    def test_aws_source_reaches_grid_search(self, monkeypatch):
        captured = {}

        def fake_search(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-here")

        import s2mosaic.pipelines.grid as grid_mod

        monkeypatch.setattr(grid_mod, "search_for_items", fake_search)

        with pytest.raises(RuntimeError, match="stop-here"):
            mosaic(
                grid_id="50HMH",
                start_year=2023,
                duration_days=1,
                bands=["B04"],
                source=SOURCE_AWS,
            )

        assert captured["source"] is AWS


class TestStacPropertyFallbacks:
    """Element 84 omits some MPC-shaped properties; the helpers fall back."""

    def test_relative_orbit_falls_back_from_product_uri(self):
        from s2mosaic.stac import _extract_relative_orbit

        # Element 84 publishes s2:product_uri with the R\\d+ token but no
        # sat:relative_orbit; recovery must produce the orbit as int.
        assert _extract_relative_orbit({"sat:relative_orbit": 42}) == 42
        props = {
            "s2:product_uri": (
                "S2B_MSIL2A_20230630T021349_N0509_R060_T50HMH_20230630T054244.SAFE"
            )
        }
        assert _extract_relative_orbit(props) == 60
        # Neither source available -> 0 (caller treats this as a single
        # degenerate orbit group, which still produces a valid sort).
        assert _extract_relative_orbit({}) == 0

    def test_mgrs_tile_falls_back_from_grid_code(self):
        from s2mosaic.stac import _extract_mgrs_tile

        # MPC publishes s2:mgrs_tile directly.
        assert _extract_mgrs_tile({"s2:mgrs_tile": "50HMH"}) == "50HMH"
        # Element 84 publishes grid:code as 'MGRS-50HMH'.
        assert _extract_mgrs_tile({"grid:code": "MGRS-50HMH"}) == "50HMH"
        # Neither -> None so the dedupe groups under 'unknown' rather than
        # silently corrupting a tile-keyed grouping.
        assert _extract_mgrs_tile({}) is None
        assert _extract_mgrs_tile({"grid:code": "garbled"}) is None


class TestAddItemInfoOnAwsShapedItems:
    """add_item_info() must populate orbit and good_data_pct from AWS-style props."""

    class _FakeItem:
        def __init__(self, datetime_, props):
            self.datetime = datetime_
            self.properties = props

    def test_orbit_recovered_from_product_uri_when_sat_field_missing(self):
        from datetime import datetime, timezone

        from s2mosaic.stac import ORBIT_COL, add_item_info

        item = self._FakeItem(
            datetime(2023, 6, 30, tzinfo=timezone.utc),
            {
                "s2:nodata_pixel_percentage": 43.77,
                "s2:high_proba_clouds_percentage": 0.0001,
                "s2:cloud_shadow_percentage": 0.0,
                "s2:product_uri": (
                    "S2B_MSIL2A_20230630T021349_N0509_R060_T50HMH_20230630T054244.SAFE"
                ),
            },
        )
        df = add_item_info([item])
        assert df.iloc[0][ORBIT_COL] == 60


class TestSearchPostFilter:
    """When source can't filter MGRS server-side, search_for_items must
    post-filter items by tile to keep grid_id-mode output tile-strict."""

    @staticmethod
    def _make_item(item_id, grid_code):
        # Real pystac.Item — ItemCollection construction inside
        # search_for_items rejects ad-hoc duck types.
        import datetime as _dt

        import pystac

        return pystac.Item(
            id=item_id,
            geometry={
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
            bbox=[0, 0, 1, 1],
            datetime=_dt.datetime(2023, 6, 15, tzinfo=_dt.timezone.utc),
            properties={"grid:code": grid_code},
        )

    def test_grid_id_post_filter_drops_adjacent_tiles_on_aws(self, monkeypatch):
        # Build a fake item collection mixing the requested tile and a
        # neighbour; search_for_items should keep only the requested one.
        import s2mosaic.stac as stac_mod
        from s2mosaic.sources import AWS

        wanted = self._make_item("S2A_50HMH_2023_0_L2A", "MGRS-50HMH")
        neighbour = self._make_item("S2A_50HLK_2023_0_L2A", "MGRS-50HLK")

        class _FakeSearch:
            def item_collection(self):
                from pystac.item_collection import ItemCollection

                return ItemCollection([wanted, neighbour])

        class _FakeCatalog:
            def search(self, **_):
                return _FakeSearch()

        # AWS.open_catalog uses pystac_client.Client.open under the hood;
        # patch that so we don't need to mutate the frozen Source dataclass.
        import pystac_client

        monkeypatch.setattr(
            pystac_client.Client,
            "open",
            classmethod(lambda cls, *_, **__: _FakeCatalog()),
        )
        # Skip the disk-cache wrapper so the search actually runs.
        monkeypatch.setattr(stac_mod, "pickle_cache", lambda _p, _k, fn: fn())

        from datetime import date as _date

        from shapely.geometry import Polygon

        items = stac_mod.search_for_items(
            bounds=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            grid_id="50HMH",
            start_date=_date(2023, 6, 1),
            end_date=_date(2023, 6, 30),
            additional_query={},
            source=AWS,
            ignore_duplicate_items=False,  # focus on the post-filter only
        )
        kept_ids = [it.id for it in items]
        assert kept_ids == ["S2A_50HMH_2023_0_L2A"]


class TestStacDatetimeFormat:
    """Regression: ``define_dates`` returns ``datetime`` (despite the
    ``Tuple[date, date]`` type hint), so the STAC ``datetime`` range
    string must produce well-formed RFC 3339 for both ``date`` and
    ``datetime`` inputs. The earlier ``f'{d.isoformat()}Z'`` pattern
    happened to produce ``"2023-06-01T00:00:00T00:00:00Z"`` when called
    from production code, and MPC tolerated that — AWS's pystac_client
    validator does not.
    """

    @staticmethod
    def _capture_query(monkeypatch):
        """Monkey-patch pystac_client to capture the query dict and return
        an empty result. Returns the dict the test can read after calling.
        """
        captured: dict = {}

        class _FakeSearch:
            def item_collection(self):
                from pystac.item_collection import ItemCollection

                return ItemCollection([])

        class _FakeCatalog:
            def search(self, **kwargs):
                captured.update(kwargs)
                return _FakeSearch()

        import pystac_client

        monkeypatch.setattr(
            pystac_client.Client,
            "open",
            classmethod(lambda cls, *_, **__: _FakeCatalog()),
        )
        return captured

    def test_grid_search_datetime_is_well_formed_for_datetime_input(self, monkeypatch):
        """``search_for_items`` must emit valid RFC 3339 even when the
        callers pass real ``datetime`` objects (as ``define_dates`` does).
        """
        from datetime import datetime

        from shapely.geometry import Polygon

        import s2mosaic.stac as stac_mod
        from s2mosaic.sources import MPC

        captured = self._capture_query(monkeypatch)
        monkeypatch.setattr(stac_mod, "pickle_cache", lambda _p, _k, fn: fn())

        stac_mod.search_for_items(
            bounds=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            grid_id="50HMH",
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 8, 1),
            additional_query={},
            source=MPC,
            ignore_duplicate_items=False,
        )

        dt = captured["datetime"]
        assert dt == "2023-06-01T00:00:00Z/2023-08-01T00:00:00Z", (
            f"Expected RFC 3339 datetime range, got {dt!r}"
        )
        # Defensive: catch the specific regression where the time
        # component was being appended twice.
        assert "T00:00:00T00:00:00" not in dt

    def test_bounds_search_datetime_is_well_formed_for_datetime_input(
        self, monkeypatch
    ):
        """Same check on the bounds-mode search path."""
        from datetime import datetime

        from shapely.geometry import Polygon

        import s2mosaic.pipelines.bounds as bounds_mod
        from s2mosaic.sources import MPC

        captured = self._capture_query(monkeypatch)
        aoi = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        bounds_mod._search_for_items_by_aoi(
            aoi_4326=aoi,
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 8, 1),
            source=MPC,
            additional_query={},
            ignore_duplicate_items=False,
        )

        dt = captured["datetime"]
        assert dt == "2023-06-01T00:00:00Z/2023-08-01T00:00:00Z", (
            f"Expected RFC 3339 datetime range, got {dt!r}"
        )
        assert "T00:00:00T00:00:00" not in dt

    def test_define_dates_returns_datetime_not_date(self):
        """Locks in the implicit contract the format string depends on:
        ``define_dates`` always returns ``datetime`` objects, so the format
        code must not rely on ``date.isoformat()``'s shorter output.
        """
        from datetime import datetime

        from s2mosaic.helpers import define_dates

        start, end = define_dates(2023, 6, 1, 0, 2, 0)
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)


class TestSourceCustomDataclass:
    """Hand-constructed Source instances behave like the built-in ones."""

    def test_custom_source_with_partial_asset_map(self):
        custom = Source(
            name="CUSTOM",
            stac_url="https://example.org/stac",
            collection_id="my-l2a",
            sign=lambda href: f"{href}?key=abc",
            band_assets={"B04": "red_band"},
        )
        assert custom.sign("h") == "h?key=abc"
        assert custom.asset_name("B04") == "red_band"
        # Unmapped names pass through unchanged.
        assert custom.asset_name("B03") == "B03"
        # No mgrs_query configured -> None (caller falls back to intersects).
        assert custom.mgrs_query("50HMH") is None
