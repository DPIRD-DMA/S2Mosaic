import logging

import pytest
from rasterio.errors import RasterioIOError

from s2mosaic.helpers import (
    SceneFetchError,
    _load_s2_grid,
    with_scene_retry,
)


class TestPackagedGrid:
    def test_missing_packaged_grid_raises_runtime_error(self, monkeypatch, tmp_path):
        import s2mosaic.helpers as helpers_mod

        class FakeResource:
            def __truediv__(self, _name):
                return self

        class FakeContext:
            def __enter__(self):
                return tmp_path / "missing.gpkg"

            def __exit__(self, exc_type, exc, tb):
                return False

        _load_s2_grid.cache_clear()
        monkeypatch.setattr(
            helpers_mod.resources, "files", lambda _package: FakeResource()
        )
        monkeypatch.setattr(
            helpers_mod.resources, "as_file", lambda _resource: FakeContext()
        )

        try:
            with pytest.raises(RuntimeError, match="S2 grid file not found"):
                _load_s2_grid()
        finally:
            _load_s2_grid.cache_clear()


class TestSceneRetry:
    def test_invalid_attempt_count_rejected(self):
        with pytest.raises(ValueError, match="attempts must be >= 1, got 0"):
            with_scene_retry(attempts=0)

    def test_exhaustion_raises_scene_fetch_error_with_cause(self, monkeypatch):
        monkeypatch.setattr("s2mosaic.helpers.time.sleep", lambda _: None)
        calls = {"n": 0}

        @with_scene_retry(attempts=3, base_delay=0.01)
        def always_fails():
            calls["n"] += 1
            raise RasterioIOError("temporary read failure")

        with pytest.raises(SceneFetchError) as exc_info:
            always_fails()

        assert calls["n"] == 3
        assert isinstance(exc_info.value.__cause__, RasterioIOError)
        assert "failed after 3 attempts" in str(exc_info.value)

    def test_retry_warning_includes_exception_chain(self, monkeypatch, caplog):
        monkeypatch.setattr("s2mosaic.helpers.time.sleep", lambda _: None)

        @with_scene_retry(attempts=2, base_delay=0.01)
        def fails_with_context():
            try:
                raise RasterioIOError("low-level read failed")
            except RasterioIOError as exc:
                raise RuntimeError("band B04 failed") from exc

        with caplog.at_level(logging.WARNING, logger="s2mosaic.helpers"):
            with pytest.raises(SceneFetchError):
                fails_with_context()

        text = caplog.text
        assert "RuntimeError: band B04 failed" in text
        assert "RasterioIOError: low-level read failed" in text

    def test_retry_does_not_retry_programming_errors(self, monkeypatch):
        monkeypatch.setattr("s2mosaic.helpers.time.sleep", lambda _: None)
        calls = {"n": 0}

        @with_scene_retry(attempts=3, base_delay=0.01)
        def fails_with_programming_error():
            calls["n"] += 1
            raise RuntimeError("bug")

        with pytest.raises(RuntimeError, match="bug"):
            fails_with_programming_error()

        assert calls["n"] == 1


class TestPickOcmResolution:
    """Clamping logic for OCM resolution given user resolution."""

    @pytest.mark.parametrize(
        "user_res, expected",
        [
            (5, 20),  # below floor → 20
            (10, 20),  # 10m output → OCM at 20m (4x faster than at 10m)
            (15, 20),  # below floor → 20
            (20, 20),  # boundary
            (25, 25),  # in range
            (30, 30),  # in range
            (40, 40),  # in range
            (50, 50),  # boundary
            (60, 50),  # above ceiling → 50
            (100, 50),  # above ceiling → 50
        ],
    )
    def test_clamping(self, user_res, expected):
        from s2mosaic.helpers import pick_ocm_resolution

        assert pick_ocm_resolution(user_res) == expected


class TestResamplingMap:
    """Verify the string→rasterio.Resampling mapping."""

    @pytest.mark.parametrize(
        "name", ["nearest", "bilinear", "cubic", "average", "lanczos"]
    )
    def test_known_methods_map_to_rasterio(self, name):
        from rasterio.enums import Resampling

        from s2mosaic.helpers import get_rasterio_resampling

        result = get_rasterio_resampling(name)
        assert isinstance(result, Resampling)
        assert result.name == name

    def test_unknown_method_raises(self):
        from s2mosaic.helpers import get_rasterio_resampling

        with pytest.raises(KeyError):
            get_rasterio_resampling("bogus")
