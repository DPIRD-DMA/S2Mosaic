import logging
from datetime import date, datetime, timezone

import pandas as pd
import pytest
from shapely.geometry import Polygon

from s2mosaic.stac import (
    DATETIME_COL,
    GOOD_DATA_PCT_COL,
    ITEM_COL,
    ORBIT_COL,
    filter_latest_processing_baselines,
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
