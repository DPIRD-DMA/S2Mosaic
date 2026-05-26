import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin

from s2mosaic.readers import (
    BoundsTileReader,
    GridTileReader,
    _HandleCache,
    _lazy_signed_url,
    make_grid_tile_reader,
    should_prewarm_sources,
)


class TestTileReaderHelpers:
    """Tile-reader helpers: prewarm policy, lazy handles, retries."""

    @pytest.mark.parametrize(
        "mosaic_method,min_observations,max_observations,expected",
        [
            ("mean", None, None, True),
            ("percentile", None, None, True),
            ("first", None, None, False),
            ("mean", 2, None, False),
            ("percentile", 2, None, False),
            ("mean", None, 2, False),
            ("percentile", None, 2, False),
        ],
    )
    def test_source_prewarm_policy(
        self, mosaic_method, min_observations, max_observations, expected
    ):
        assert (
            should_prewarm_sources(
                mosaic_method,
                min_observations,
                max_observations,
            )
            is expected
        )

    def test_handle_cache_resolves_sources_only_on_first_read(self, monkeypatch):
        calls = {"resolver": 0, "open": 0}

        def resolver(refresh=False):
            calls["resolver"] += 1
            assert refresh is False
            return "lazy-source.tif"

        class FakeDataset:
            pass

        def fake_open(source):
            calls["open"] += 1
            assert source == "lazy-source.tif"
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)
        cache = _HandleCache([[resolver]])

        assert calls == {"resolver": 0, "open": 0}
        first = cache.get(0, 0)
        second = cache.get(0, 0)

        assert first is second
        assert calls == {"resolver": 1, "open": 1}

    def test_handle_cache_rejects_reads_after_close(self):
        cache = _HandleCache([[lambda refresh=False: "lazy-source.tif"]])
        cache.close()

        with pytest.raises(RuntimeError, match="closed"):
            cache.get(0, 0)

    def test_lazy_signed_url_refreshes_after_ttl(self, monkeypatch):
        now = {"value": 100.0}
        calls = []

        class FakeSource:
            def sign(self, href):
                calls.append(href)
                return f"signed-{len(calls)}"

        monkeypatch.setattr("s2mosaic.readers.time.monotonic", lambda: now["value"])

        get_signed_url = _lazy_signed_url(FakeSource(), "remote.tif", ttl_seconds=10)

        assert get_signed_url() == "signed-1"
        assert get_signed_url() == "signed-1"
        now["value"] = 111.0
        assert get_signed_url() == "signed-2"
        assert get_signed_url(refresh=True) == "signed-3"
        assert calls == ["remote.tif", "remote.tif", "remote.tif"]

    def test_grid_reader_signs_only_on_first_read_when_not_prewarmed(self, monkeypatch):
        calls = {"sign": 0, "open": 0}

        class FakeAsset:
            href = "remote.tif"

        class FakeItem:
            assets = {"B04": FakeAsset()}
            id = "fake-scene"

        class FakeSource:
            name = "FAKE"

            def asset_name(self, canonical):
                return canonical

            def sign(self, href):
                calls["sign"] += 1
                return f"signed-{href}"

        class FakeDataset:
            width = 10
            height = 10

            def read(self, band_idx, *, window, out_shape, resampling):
                return np.full(out_shape, band_idx, dtype=np.uint16)

            def close(self):
                pass

        def fake_open(source):
            calls["open"] += 1
            assert source == "signed-remote.tif"
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)

        read_fn = make_grid_tile_reader(
            items=[FakeItem()],
            href_template=[("B04", 1)],
            source=FakeSource(),
            s2_scene_size=10,
            resolution=10,
            resampling_method="nearest",
            prewarm=False,
        )

        assert calls == {"sign": 0, "open": 0}
        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert calls == {"sign": 1, "open": 1}

    def test_grid_reader_prewarm_signs_sources_once(self, monkeypatch):
        calls = {"sign": 0}

        class FakeAsset:
            href = "remote.tif"

        class FakeItem:
            assets = {"B04": FakeAsset()}
            id = "fake-scene"

        class FakeSource:
            name = "FAKE"

            def asset_name(self, canonical):
                return canonical

            def sign(self, href):
                calls["sign"] += 1
                return f"signed-{href}"

        make_grid_tile_reader(
            items=[FakeItem()],
            href_template=[("B04", 1)],
            source=FakeSource(),
            s2_scene_size=10,
            resolution=10,
            resampling_method="nearest",
            prewarm=True,
        )

        assert calls == {"sign": 1}

    def test_grid_reader_reopens_with_refreshed_source_on_read_error(self, monkeypatch):
        resolver_calls = []
        open_calls = []

        def resolver(refresh=False):
            resolver_calls.append(refresh)
            return f"remote-refresh-{refresh}.tif"

        class FakeDataset:
            width = 10
            height = 10

            def __init__(self, source):
                self.source = source

            def read(self, band_idx, *, window, out_shape, resampling):
                if self.source == "remote-refresh-False.tif":
                    raise RasterioIOError("expired token")
                return np.full(out_shape, band_idx, dtype=np.uint16)

            def close(self):
                pass

        def fake_open(source):
            open_calls.append(source)
            return FakeDataset(source)

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)

        cache = _HandleCache([[resolver]])
        read_fn = GridTileReader(
            cache,
            href_band_indices=[1],
            s2_scene_size=10,
            rio_resampling=Resampling.nearest,
        )

        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert resolver_calls == [False, True]
        assert open_calls == ["remote-refresh-False.tif", "remote-refresh-True.tif"]

    def test_grid_reader_recovers_on_third_read_attempt(self, monkeypatch):
        monkeypatch.setattr("s2mosaic.readers.time.sleep", lambda _: None)
        resolver_calls = []
        open_calls = []
        read_calls = {"count": 0}

        def resolver(refresh=False):
            resolver_calls.append(refresh)
            return f"remote-refresh-{refresh}.tif"

        class FakeDataset:
            width = 10
            height = 10

            def read(self, band_idx, *, window, out_shape, resampling):
                read_calls["count"] += 1
                if read_calls["count"] < 3:
                    raise RasterioIOError("temporary read failure")
                return np.full(out_shape, band_idx, dtype=np.uint16)

            def close(self):
                pass

        def fake_open(source):
            open_calls.append(source)
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)

        read_fn = GridTileReader(
            _HandleCache([[resolver]]),
            href_band_indices=[1],
            s2_scene_size=10,
            rio_resampling=Resampling.nearest,
        )

        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert read_calls["count"] == 3
        assert resolver_calls == [False, True, True]
        assert open_calls == [
            "remote-refresh-False.tif",
            "remote-refresh-True.tif",
            "remote-refresh-True.tif",
        ]

    def test_grid_reader_reopens_with_refreshed_source_on_open_error(self, monkeypatch):
        resolver_calls = []
        open_calls = []

        def resolver(refresh=False):
            resolver_calls.append(refresh)
            return f"remote-refresh-{refresh}.tif"

        class FakeDataset:
            width = 10
            height = 10

            def read(self, band_idx, *, window, out_shape, resampling):
                return np.full(out_shape, band_idx, dtype=np.uint16)

            def close(self):
                pass

        def fake_open(source):
            open_calls.append(source)
            if source == "remote-refresh-False.tif":
                raise RasterioIOError("expired token")
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)

        cache = _HandleCache([[resolver]])
        read_fn = GridTileReader(
            cache,
            href_band_indices=[1],
            s2_scene_size=10,
            rio_resampling=Resampling.nearest,
        )

        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert resolver_calls == [False, True]
        assert open_calls == ["remote-refresh-False.tif", "remote-refresh-True.tif"]

    def test_bounds_reader_recovers_on_third_open_attempt(self, monkeypatch):
        monkeypatch.setattr("s2mosaic.readers.time.sleep", lambda _: None)
        resolver_calls = []
        open_calls = []

        def resolver(refresh=False):
            resolver_calls.append(refresh)
            return f"bounds-refresh-{refresh}.tif"

        class FakeDataset:
            def read(self, band_idx, window):
                return np.full(
                    (int(window.height), int(window.width)),
                    band_idx,
                    dtype=np.uint16,
                )

            def close(self):
                pass

        def fake_open(source):
            open_calls.append(source)
            if len(open_calls) < 3:
                raise RasterioIOError("temporary open failure")
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)
        monkeypatch.setattr("s2mosaic.readers.WarpedVRT", lambda src, **_: src)

        read_fn = BoundsTileReader(
            sources=[[resolver]],
            href_band_indices=[1],
            target_crs_obj=CRS.from_epsg(32750),
            user_transform=from_origin(0, 10, 1, 1),
            width=10,
            height=10,
            rio_resampling=Resampling.nearest,
        )

        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert resolver_calls == [False, True, True]
        assert open_calls == [
            "bounds-refresh-False.tif",
            "bounds-refresh-True.tif",
            "bounds-refresh-True.tif",
        ]

    def test_bounds_reader_reopens_with_refreshed_source_on_read_error(
        self, monkeypatch
    ):
        resolver_calls = []
        open_calls = []

        def resolver(refresh=False):
            resolver_calls.append(refresh)
            return f"bounds-refresh-{refresh}.tif"

        class FakeDataset:
            def __init__(self, source):
                self.source = source

            def read(self, band_idx, window):
                if self.source == "bounds-refresh-False.tif":
                    raise RasterioIOError("expired token")
                return np.full(
                    (int(window.height), int(window.width)),
                    band_idx,
                    dtype=np.uint16,
                )

            def close(self):
                pass

        def fake_open(source):
            open_calls.append(source)
            return FakeDataset(source)

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)
        monkeypatch.setattr("s2mosaic.readers.WarpedVRT", lambda src, **_: src)

        read_fn = BoundsTileReader(
            sources=[[resolver]],
            href_band_indices=[1],
            target_crs_obj=CRS.from_epsg(32750),
            user_transform=from_origin(0, 10, 1, 1),
            width=10,
            height=10,
            rio_resampling=Resampling.nearest,
        )

        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert resolver_calls == [False, True]
        assert open_calls == ["bounds-refresh-False.tif", "bounds-refresh-True.tif"]

    def test_bounds_reader_reopens_with_refreshed_source_on_open_error(
        self, monkeypatch
    ):
        resolver_calls = []
        open_calls = []

        def resolver(refresh=False):
            resolver_calls.append(refresh)
            return f"bounds-refresh-{refresh}.tif"

        class FakeDataset:
            def read(self, band_idx, window):
                return np.full(
                    (int(window.height), int(window.width)),
                    band_idx,
                    dtype=np.uint16,
                )

            def close(self):
                pass

        def fake_open(source):
            open_calls.append(source)
            if source == "bounds-refresh-False.tif":
                raise RasterioIOError("expired token")
            return FakeDataset()

        monkeypatch.setattr("s2mosaic.readers.rio.open", fake_open)
        monkeypatch.setattr("s2mosaic.readers.WarpedVRT", lambda src, **_: src)

        read_fn = BoundsTileReader(
            sources=[[resolver]],
            href_band_indices=[1],
            target_crs_obj=CRS.from_epsg(32750),
            user_transform=from_origin(0, 10, 1, 1),
            width=10,
            height=10,
            rio_resampling=Resampling.nearest,
        )

        data = read_fn(0, 0, (0, 0, 2, 3))

        np.testing.assert_array_equal(data, np.ones((2, 3), dtype=np.uint16))
        assert resolver_calls == [False, True]
        assert open_calls == ["bounds-refresh-False.tif", "bounds-refresh-True.tif"]

    def test_bounds_reader_rejects_reads_after_close(self):
        read_fn = BoundsTileReader(
            sources=[[lambda refresh=False: "bounds.tif"]],
            href_band_indices=[1],
            target_crs_obj=CRS.from_epsg(32750),
            user_transform=from_origin(0, 10, 1, 1),
            width=10,
            height=10,
            rio_resampling=Resampling.nearest,
        )
        read_fn.close()

        with pytest.raises(RuntimeError, match="closed"):
            read_fn(0, 0, (0, 0, 2, 3))
