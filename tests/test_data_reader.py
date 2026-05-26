import numpy as np

from s2mosaic.data_reader import get_full_band


class TestGetFullBand:
    def test_visual_detection_uses_asset_name_not_href(self, monkeypatch):
        read_calls = []

        class FakeSource:
            def sign(self, href):
                return href

        class FakeDataset:
            width = 10980
            height = 10980
            profile = {"driver": "GTiff"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, indexes, *, window, out_shape):
                read_calls.append((indexes, out_shape))
                if isinstance(indexes, list):
                    return np.ones(out_shape, dtype=np.uint16)
                return np.ones(out_shape, dtype=np.uint16)

        monkeypatch.setattr(
            "s2mosaic.data_reader.rio.open", lambda _href: FakeDataset()
        )

        array, _ = get_full_band(
            href="https://example.test/B04.tif?token=/visual/not-asset",
            source=FakeSource(),
            res=10,
            asset_name="B04",
        )

        assert array.shape == (1, 10980, 10980)
        assert read_calls == [(1, (10980, 10980))]

    def test_visual_asset_reads_three_bands(self, monkeypatch):
        read_calls = []

        class FakeSource:
            def sign(self, href):
                return href

        class FakeDataset:
            width = 10980
            height = 10980
            profile = {"driver": "GTiff"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, indexes, *, window, out_shape):
                read_calls.append((indexes, out_shape))
                return np.ones(out_shape, dtype=np.uint16)

        monkeypatch.setattr(
            "s2mosaic.data_reader.rio.open", lambda _href: FakeDataset()
        )

        array, _ = get_full_band(
            href="https://example.test/B04.tif",
            source=FakeSource(),
            res=10,
            asset_name="visual",
        )

        assert array.shape == (3, 10980, 10980)
        assert read_calls == [([1, 2, 3], (3, 10980, 10980))]
