import logging
import warnings
from datetime import date, datetime, timezone

import pandas as pd
import pytest
from pystac import Item
from pystac.item_collection import ItemCollection
from shapely.geometry import Polygon

from s2mosaic.sources import get_source
from s2mosaic.stac import (
    DATETIME_COL,
    GOOD_DATA_PCT_COL,
    ITEM_COL,
    ORBIT_COL,
    _extract_mgrs_tile,
    filter_latest_processing_baselines,
    search_for_items,
    sort_items,
)


class TestStacBoundsSearch:
    class FakeSearch:
        def __init__(self, items):
            self._items = items

        def item_collection(self):
            return self._items

    class FakeCatalog:
        def __init__(self, calls, items):
            self._calls = calls
            self._items = items

        def search(self, **query):
            self._calls.append(query)
            return TestStacBoundsSearch.FakeSearch(self._items)

    class FakeSource:
        name = "fake"
        collection_id = "sentinel-test"

        def __init__(self, catalog):
            self._catalog = catalog

        def open_catalog(self, *, stac_io):
            return self._catalog

    def test_bbox_search_uses_bbox_query(self):
        import s2mosaic.stac_bounds as stac_bounds

        calls = []
        items = ["scene-a"]
        source = self.FakeSource(self.FakeCatalog(calls, items))

        result = stac_bounds._search_for_items_by_bbox(
            bbox_4326=(1.0, 2.0, 3.0, 4.0),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 15),
            source=source,
            additional_query={"eo:cloud_cover": {"lt": 50}},
            ignore_duplicate_items=False,
        )

        assert result == items
        assert calls == [
            {
                "collections": ["sentinel-test"],
                "datetime": "2023-01-01T00:00:00Z/2023-01-15T00:00:00Z",
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "query": {"eo:cloud_cover": {"lt": 50}},
            }
        ]

    def test_aoi_search_uses_intersects_query(self):
        import s2mosaic.stac_bounds as stac_bounds

        calls = []
        source = self.FakeSource(self.FakeCatalog(calls, ["scene-a"]))
        aoi = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)])

        stac_bounds._search_for_items_by_aoi(
            aoi_4326=aoi,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 2),
            source=source,
            ignore_duplicate_items=False,
        )

        assert "bbox" not in calls[0]
        assert calls[0]["intersects"]["type"] == "Polygon"

    def test_search_dedupes_by_default(self, monkeypatch):
        import s2mosaic.stac_bounds as stac_bounds

        calls = []
        items = ["scene-a", "scene-a-duplicate"]
        source = self.FakeSource(self.FakeCatalog(calls, items))
        dedupe_inputs = []

        def fake_filter_latest_processing_baselines(items_arg):
            dedupe_inputs.append(items_arg)
            return ["scene-a"]

        monkeypatch.setattr(
            stac_bounds,
            "filter_latest_processing_baselines",
            fake_filter_latest_processing_baselines,
        )

        result = stac_bounds._search_for_items_by_bbox(
            bbox_4326=(1.0, 2.0, 3.0, 4.0),
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 15),
            source=source,
        )

        assert result == ["scene-a"]
        assert dedupe_inputs == [items]


class TestStacGridSearch:
    class FakeSearch:
        def __init__(self, items):
            self._items = items

        def item_collection(self):
            return self._items

    class FakeCatalog:
        def __init__(self, items):
            self._items = items

        def search(self, **_query):
            return TestStacGridSearch.FakeSearch(self._items)

    class FakeSource:
        name = "fake"
        collection_id = "sentinel-test"

        def __init__(self, items):
            self._catalog = TestStacGridSearch.FakeCatalog(items)

        def mgrs_query(self, grid_id):
            return {"grid:code": {"eq": f"MGRS-{grid_id}"}}

        def open_catalog(self, *, stac_io):
            return self._catalog

    def _item(self, item_id, grid_code):
        return Item(
            id=item_id,
            geometry=None,
            bbox=None,
            datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
            properties={
                "grid:code": grid_code,
                "s2:processing_baseline": "05.11",
            },
        )

    def test_grid_search_post_filters_items_by_mgrs_grid_code(self):
        items = ItemCollection(
            [
                self._item("wanted", "MGRS-50HMH"),
                self._item("wrong", "MGRS-50HNH"),
                self._item("missing", None),
            ]
        )

        result = search_for_items(
            grid_id="50HMH",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 2),
            additional_query={},
            source=self.FakeSource(items),
            ignore_duplicate_items=False,
        )

        assert [item.id for item in result] == ["wanted"]


class TestSortItems:
    def test_invalid_scene_order_raises_value_error(self):
        items = pd.DataFrame(
            {
                GOOD_DATA_PCT_COL: [90.0],
                ORBIT_COL: [1],
                DATETIME_COL: [datetime(2023, 1, 1, tzinfo=timezone.utc)],
                ITEM_COL: [object()],
            }
        )

        with pytest.raises(ValueError, match="Invalid scene_order"):
            sort_items(items, "bogus")


class TestProcessingBaselineFilter:
    def _item(self, item_id, baseline):
        from pystac import Item

        return Item(
            id=item_id,
            geometry=None,
            bbox=None,
            datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
            properties={
                "s2:mgrs_tile": "50HMH",
                "s2:processing_baseline": baseline,
            },
        )

    def _granule(self, item_id, baseline, datastrip_sensing, dt, tile="50HMH"):
        """An item as the providers actually publish one.

        ``datastrip_sensing`` names the granule and survives reprocessing;
        ``dt`` is the datetime property, which does not.
        """
        from pystac import Item

        return Item(
            id=item_id,
            geometry=None,
            bbox=None,
            datetime=dt,
            properties={
                "s2:mgrs_tile": tile,
                "s2:processing_baseline": baseline,
                "s2:datastrip_id": (
                    f"S2A_OPER_MSI_L2A_DS_S2RP_20230614T234954_"
                    f"S{datastrip_sensing}_N{baseline}"
                ),
            },
        )

    def test_a_restamped_reprocessing_is_not_kept_as_a_second_scene(self):
        """Element 84's Collection-1 restamping must not duplicate a scene.

        The real case: 50HMH on 2019-03-23 is published at 02:31:32 under
        baseline 02.11 and at 02:27:06 under 05.00 -- the same granule, four
        minutes apart. Keyed on datetime these look like two acquisitions,
        so both survive and the same scene enters the stack twice.
        """
        from pystac.item_collection import ItemCollection

        items = ItemCollection(
            [
                self._granule(
                    "old",
                    "02.11",
                    "20190323T022444",
                    datetime(2019, 3, 23, 2, 31, 32, tzinfo=timezone.utc),
                ),
                self._granule(
                    "reprocessed",
                    "05.00",
                    "20190323T022444",
                    datetime(2019, 3, 23, 2, 27, 6, tzinfo=timezone.utc),
                ),
            ]
        )

        filtered = filter_latest_processing_baselines(items)
        assert [item.id for item in filtered] == ["reprocessed"]

    def test_two_granules_of_one_datatake_both_survive(self):
        """Microsoft's datatake stamping must not discard a granule.

        MPC stamps every granule of a datatake with the datatake start, so
        50HMH on 2019-11-23 has two granules -- 58% and 74% nodata -- both
        reading 02:13:51. Keyed on datetime they collide, and because their
        baselines tie, ``max`` drops one by response order alone. They are
        different imagery and both belong in the mosaic.
        """
        from pystac.item_collection import ItemCollection

        shared = datetime(2019, 11, 23, 2, 13, 51, tzinfo=timezone.utc)
        items = ItemCollection(
            [
                self._granule("granule-a", "02.12", "20191123T022659", shared),
                self._granule("granule-b", "02.12", "20191123T022134", shared),
            ]
        )

        filtered = filter_latest_processing_baselines(items)
        assert sorted(item.id for item in filtered) == ["granule-a", "granule-b"]

    def test_each_granule_keeps_its_own_latest_baseline(self):
        # The two behaviours together: two granules, each published twice.
        from pystac.item_collection import ItemCollection

        shared = datetime(2019, 11, 23, 2, 13, 51, tzinfo=timezone.utc)
        later = datetime(2019, 11, 23, 2, 27, 8, tzinfo=timezone.utc)
        items = ItemCollection(
            [
                self._granule("a-old", "02.13", "20191123T022659", shared),
                self._granule("a-new", "05.00", "20191123T022659", later),
                self._granule("b-old", "02.13", "20191123T022134", shared),
                self._granule("b-new", "05.00", "20191123T022134", later),
            ]
        )

        filtered = filter_latest_processing_baselines(items)
        assert sorted(item.id for item in filtered) == ["a-new", "b-new"]

    def test_a_missing_datastrip_id_falls_back_to_datetime_and_warns(self, caplog):
        # No item in the sampled archive lacked s2:datastrip_id, so this path
        # is defensive; it must still behave and say so rather than crash.
        from pystac import Item
        from pystac.item_collection import ItemCollection

        def bare(item_id, baseline):
            return Item(
                id=item_id,
                geometry=None,
                bbox=None,
                datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
                properties={
                    "s2:mgrs_tile": "50HMH",
                    "s2:processing_baseline": baseline,
                },
            )

        items = ItemCollection([bare("old", "02.13"), bare("new", "05.00")])
        with caplog.at_level(logging.WARNING):
            filtered = filter_latest_processing_baselines(items)

        assert [item.id for item in filtered] == ["new"]
        assert "no parseable s2:datastrip_id" in caplog.text

    def test_malformed_processing_baseline_is_treated_as_lowest(self, caplog):
        from pystac.item_collection import ItemCollection

        items = ItemCollection(
            [
                self._item("bad", "not-a-number"),
                self._item("good", "05.11"),
            ]
        )

        with caplog.at_level(logging.WARNING):
            filtered = filter_latest_processing_baselines(items)

        assert [item.id for item in filtered] == ["good"]
        assert "Invalid processing baseline" in caplog.text


@pytest.mark.slow
class TestServerSideQueryFiltering:
    """The MGRS filter must be applied by the server, not just by us.

    ``search_for_items`` sends the tile filter through the STAC Query
    extension and then re-checks the result client-side. That second pass is
    a backstop, not the mechanism: without server-side filtering a grid
    search pulls every scene in the window and discards almost all of it,
    turning a wrong query into a slow, sparse mosaic rather than an error.

    Neither provider is trustworthy about what it supports. MPC honours
    Query while declaring only four conformance classes, none of them
    ``item-search#query``, which is why ``Source.open_catalog`` asserts it.
    Element 84 declares Query but answers a CQL2 Filter search with HTTP 200
    and an unfiltered list, so "the search succeeded" says nothing. These
    hit the network to check the filter actually bit.
    """

    GRID_ID = "50HMH"
    START = date(2023, 6, 1)
    END = date(2023, 6, 15)

    @pytest.mark.parametrize("source_name", ["MPC", "AWS"])
    def test_grid_search_returns_only_the_requested_tile(self, source_name):
        items = search_for_items(
            grid_id=self.GRID_ID,
            start_date=self.START,
            end_date=self.END,
            additional_query={"eo:cloud_cover": {"lt": 100}},
            source=get_source(source_name),
        )
        assert len(items) > 0, "no items; the window or collection moved"
        tiles = {_extract_mgrs_tile(item.properties) for item in items}
        assert tiles == {self.GRID_ID}

    @pytest.mark.parametrize("source_name", ["MPC", "AWS"])
    def test_the_server_filtered_rather_than_the_client_backstop(
        self, source_name, caplog
    ):
        # The client-side pass logs when it drops anything. A silent run means
        # the query was honoured; a noisy one means we are searching wide and
        # filtering locally, which still produces a correct mosaic and would
        # otherwise go unnoticed.
        with caplog.at_level(logging.WARNING, logger="s2mosaic.stac"):
            search_for_items(
                grid_id=self.GRID_ID,
                start_date=self.START,
                end_date=self.END,
                additional_query={"eo:cloud_cover": {"lt": 100}},
                source=get_source(source_name),
            )
        dropped = [r for r in caplog.records if "mgrs" in r.getMessage().lower()]
        assert not dropped, (
            f"{source_name} returned items from other tiles: "
            f"{[r.getMessage() for r in dropped]}"
        )

    @pytest.mark.parametrize("source_name", ["MPC", "AWS"])
    def test_cloud_cover_is_filtered_server_side(self, source_name):
        # The other thing `query` carries. Unlike the MGRS filter it has no
        # client-side backstop, so if a provider stopped honouring it the
        # only symptom would be more scenes than asked for.
        threshold = 20
        items = search_for_items(
            grid_id=self.GRID_ID,
            start_date=date(2023, 6, 1),
            end_date=date(2023, 12, 1),
            additional_query={"eo:cloud_cover": {"lt": threshold}},
            source=get_source(source_name),
        )
        assert len(items) > 0
        worst = max(item.properties["eo:cloud_cover"] for item in items)
        assert worst < threshold

    @pytest.mark.parametrize("source_name", ["MPC", "AWS"])
    def test_no_conformance_warning_is_raised(self, source_name):
        # Source.open_catalog asserts QUERY because MPC under-declares. If a
        # provider ever drops real support this stays quiet, which is what
        # the filtering tests above are for; this one only keeps the noise
        # out of users' notebooks.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search_for_items(
                grid_id=self.GRID_ID,
                start_date=self.START,
                end_date=self.END,
                additional_query={"eo:cloud_cover": {"lt": 100}},
                source=get_source(source_name),
            )
        conformance = [w for w in caught if "conform" in str(w.message).lower()]
        assert not conformance, [str(w.message) for w in conformance]
