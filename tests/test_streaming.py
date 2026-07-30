import threading

import numpy as np

from s2mosaic.helpers import SceneFetchError
from s2mosaic.streaming import iter_ordered_fetches


class TestOrderedPrefetch:
    class FakeItem:
        def __init__(self, scene_id):
            self.id = scene_id

    def test_yields_sorted_items_while_fetching_in_parallel(self):
        active = 0
        max_active = 0
        lock = threading.Lock()
        # iter_ordered_fetches submits the first max_workers items up front, so
        # items 0 and 1 are both in flight against a 2-thread pool. The barrier
        # holds each of them inside fake_fetch until the other arrives, which
        # makes the overlap deterministic -- a sleep on item 0 would only make
        # it *likely* that item 1 entered before item 0 returned. The timeout
        # turns a regression that serialises the fetches into a clear failure
        # rather than a hang.
        overlap = threading.Barrier(2, timeout=30)

        def fake_fetch(idx, item):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                if idx < 2:
                    overlap.wait()
                return np.full((1, 1), int(item.id), dtype=np.uint8)
            finally:
                with lock:
                    active -= 1

        items = [
            self.FakeItem("0"),
            self.FakeItem("1"),
            self.FakeItem("2"),
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
            self.FakeItem("0"),
            self.FakeItem("1"),
            self.FakeItem("2"),
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
