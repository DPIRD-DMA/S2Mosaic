import os

from s2mosaic import mosaic
from s2mosaic.gdal_env import (
    GDAL_NETWORK_DEFAULTS,
    apply_gdal_network_defaults,
    restore_gdal_network_env,
)


class TestNetworkDefaults:
    def test_gdal_network_defaults_include_timeouts(self, monkeypatch):
        for key in GDAL_NETWORK_DEFAULTS:
            monkeypatch.delenv(key, raising=False)

        snapshot = apply_gdal_network_defaults()

        assert os.environ["GDAL_HTTP_CONNECTTIMEOUT"] == "30"
        assert os.environ["GDAL_HTTP_TIMEOUT"] == "120"
        assert os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        assert snapshot["GDAL_HTTP_TIMEOUT"] is None

    def test_gdal_network_defaults_preserve_user_values(self, monkeypatch):
        monkeypatch.setenv("GDAL_HTTP_TIMEOUT", "9")

        snapshot = apply_gdal_network_defaults()

        assert os.environ["GDAL_HTTP_TIMEOUT"] == "9"
        assert snapshot["GDAL_HTTP_TIMEOUT"] == "9"

    def test_gdal_network_env_can_be_restored(self, monkeypatch):
        for key in GDAL_NETWORK_DEFAULTS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("GDAL_HTTP_TIMEOUT", "9")

        snapshot = apply_gdal_network_defaults()
        assert os.environ["GDAL_HTTP_TIMEOUT"] == "9"
        assert os.environ["GDAL_HTTP_VERSION"] == "2TLS"

        restore_gdal_network_env(snapshot)

        assert os.environ["GDAL_HTTP_TIMEOUT"] == "9"
        assert "GDAL_HTTP_VERSION" not in os.environ

    def test_opt_out_still_returns_snapshot(self, monkeypatch):
        for key in GDAL_NETWORK_DEFAULTS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("S2MOSAIC_NO_GDAL_DEFAULTS", "1")

        snapshot = apply_gdal_network_defaults()

        assert snapshot["GDAL_HTTP_TIMEOUT"] is None
        assert "GDAL_HTTP_TIMEOUT" not in os.environ

    def test_mosaic_applies_gdal_network_defaults_before_pipeline(self, monkeypatch):
        for key in GDAL_NETWORK_DEFAULTS:
            monkeypatch.delenv(key, raising=False)

        seen = {}

        def fake_run_grid_pipeline(request, source):
            seen["timeout"] = os.environ.get("GDAL_HTTP_TIMEOUT")
            return "ok"

        monkeypatch.setattr(
            "s2mosaic.coordinator.run_grid_pipeline", fake_run_grid_pipeline
        )

        result = mosaic(grid_id="50HMH", start_year=2023)

        assert result == "ok"
        assert seen["timeout"] == "120"

    def test_mosaic_respects_gdal_defaults_opt_out(self, monkeypatch):
        for key in GDAL_NETWORK_DEFAULTS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("S2MOSAIC_NO_GDAL_DEFAULTS", "1")

        monkeypatch.setattr(
            "s2mosaic.coordinator.run_grid_pipeline",
            lambda request, source: "ok",
        )

        result = mosaic(grid_id="50HMH", start_year=2023)

        assert result == "ok"
        assert "GDAL_HTTP_TIMEOUT" not in os.environ
