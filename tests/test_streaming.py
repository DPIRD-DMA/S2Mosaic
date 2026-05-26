import threading
import time

import numpy as np

from s2mosaic.helpers import SceneFetchError
from s2mosaic.streaming import iter_ordered_fetches


class TestOrderedPrefetch:
    class FakeItem:
        def __init__(self, scene_id, delay):
            self.id = scene_id
            self.delay = delay

    def test_yields_sorted_items_while_fetching_in_parallel(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_fetch(_idx, item):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(item.delay)
                return np.full((1, 1), int(item.id), dtype=np.uint8)
            finally:
                with lock:
                    active -= 1

        items = [
            self.FakeItem("0", 0.05),
            self.FakeItem("1", 0.0),
            self.FakeItem("2", 0.0),
        ]

        got = list(
            iter_ordered_fetches(
                items=items,
                fetch_fn=fake_fetch,
                max_workers=2,
            )
        )

        assert [i for i, _ in got] == [0, 1, 2]
        assert [int(arr[0, 0]) for _, arr in got] == [0, 1, 2]
        assert max_active == 2

    def test_reports_fetch_failures_in_item_order(self):
        def fake_fetch(_idx, item):
            if item.id == "1":
                raise SceneFetchError("failed")
            return np.full((1, 1), int(item.id), dtype=np.uint8)

        items = [
            self.FakeItem("0", 0.0),
            self.FakeItem("1", 0.0),
            self.FakeItem("2", 0.0),
        ]

        got = list(
            iter_ordered_fetches(
                items=items,
                fetch_fn=fake_fetch,
                max_workers=2,
            )
        )

        assert [i for i, _ in got] == [0, 1, 2]
        assert isinstance(got[1][1], SceneFetchError)
