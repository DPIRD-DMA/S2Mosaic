import io
import logging

import pytest
from rasterio.errors import RasterioIOError

from s2mosaic.helpers import (
    SceneFetchError,
    normalize_grid_id,
    report_dropped_scenes,
    with_scene_retry,
)


class TestNormalizeGridId:
    """Validate + canonicalize Sentinel-2 MGRS grid ids."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("50HMK", "50HMK"),
            ("50hmk", "50HMK"),  # lowercase -> uppercased
            ("  50HMK ", "50HMK"),  # whitespace stripped
            ("1FBE", "1FBE"),  # single-digit zone
            ("60XVQ", "60XVQ"),  # zone 60, last valid row letter
            ("01CAA", None),  # leading zero — bad: regex requires no leading zero
        ],
    )
    def test_accepts_or_rejects(self, raw, expected):
        if expected is None:
            with pytest.raises(ValueError, match="not a valid Sentinel-2"):
                normalize_grid_id(raw)
        else:
            assert normalize_grid_id(raw) == expected

    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "   ",  # whitespace only
            "50AMK",  # latitude band A (polar UPS)
            "50IMK",  # latitude band I (forbidden — looks like 1)
            "50OMK",  # latitude band O (forbidden — looks like 0)
            "50YMK",  # latitude band Y (polar UPS)
            "50HIM",  # first grid letter I forbidden
            "50HOI",  # second grid letter O forbidden
            "50HMV",  # second letter V is valid... actually this should pass
            "50HMW",  # second letter W is OUT OF RANGE (max V for rows)
            "61HMK",  # zone 61 out of range
            "0HMK",  # zone 0 out of range
            "50HMK1",  # too long
            "50HM",  # too short
        ],
    )
    def test_rejects_malformed(self, bad):
        # 50HMV is actually valid (row letter V is the last allowed) — drop it
        # from the rejection list by checking explicitly here.
        if bad == "50HMV":
            assert normalize_grid_id(bad) == "50HMV"
            return
        with pytest.raises(ValueError):
            normalize_grid_id(bad)

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            normalize_grid_id(50)  # type: ignore[arg-type]


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

    def test_retry_delegates_to_backoff_delay_with_base(self, monkeypatch):
        # Guard: the dedupe relies on with_scene_retry going through
        # backoff_delay. If a future refactor inlines the math, jitter would
        # silently disappear from Phase 1 — this catches that drift.
        monkeypatch.setattr("s2mosaic.helpers.time.sleep", lambda _: None)
        calls = []

        def fake_backoff(attempt, *, base):
            calls.append((attempt, base))
            return 0.0

        monkeypatch.setattr("s2mosaic.helpers.backoff_delay", fake_backoff)

        @with_scene_retry(attempts=3, base_delay=2.5)
        def always_fails():
            raise RasterioIOError("transient")

        with pytest.raises(SceneFetchError):
            always_fails()

        # Two sleeps between three attempts; both pass the caller's base_delay.
        assert calls == [(0, 2.5), (1, 2.5)]


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


class TestReportDroppedScenes:
    def test_nothing_printed_when_no_drops(self):
        buf = io.StringIO()
        report_dropped_scenes([], total=12, stream=buf)
        assert buf.getvalue() == ""

    def test_summary_includes_count_total_and_ids(self):
        buf = io.StringIO()
        report_dropped_scenes(
            [
                {"id": "S2A_X", "reason": "timeout"},
                {"id": "S2B_Y", "reason": "403"},
            ],
            total=10,
            stream=buf,
        )
        text = buf.getvalue()
        assert "2/10 scenes dropped" in text
        assert "S2A_X" in text and "S2B_Y" in text
