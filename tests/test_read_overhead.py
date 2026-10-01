"""Warn when a requested resolution has no matching COG overview."""

import logging

import pytest

import s2mosaic.config as config_mod
from s2mosaic import mosaic
from s2mosaic.config import resolution_read_overheads, warn_resolution_read_overhead
from s2mosaic.sources import AWS, DEA, MPC


@pytest.fixture(autouse=True)
def _fresh_warning_registry(monkeypatch):
    monkeypatch.setattr(config_mod, "_warned_messages", set())


class TestReadResolution:
    @pytest.mark.parametrize(
        "source, band, res, expected",
        [
            (MPC, "B04", 10, 10.0),  # native
            (MPC, "B04", 20, 20.0),  # 2x overview
            (MPC, "B04", 60, 40.0),  # 4x is the coarsest under 6x
            (AWS, "B04", 320, 160.0),  # AWS stops at 16x
            (MPC, "B04", 320, 320.0),  # MPC goes to 32x
            (DEA, "B04", 60, 10.0),  # no overview under 8x
            (DEA, "B04", 80, 80.0),  # 8x
            (DEA, "B11", 60, 20.0),  # 20 m band, no overview under 8x
            (AWS, "AOT", 60, 60.0),  # AOT is native 60 m on AWS
            (MPC, "B04", 5, 10.0),  # upsampling reads native
        ],
    )
    def test_matches_gdal_overview_rule(self, source, band, res, expected):
        assert source.read_resolution(band, res) == expected

    def test_unrecorded_band_returns_none(self):
        assert DEA.read_resolution("visual", 60) is None


class TestResolutionReadOverheads:
    def test_dea_groups_bands_by_level(self):
        assert resolution_read_overheads(DEA, ["B04", "B03", "B11"], 60) == [
            (["B04", "B03"], 10.0, 36.0, 80.0),
            (["B11"], 20.0, 9.0, 160.0),
        ]

    @pytest.mark.parametrize("res", [10, 20, 30, 60, 80, 160])
    def test_mpc_never_crosses_the_threshold(self, res):
        # 2x-32x overviews keep every resolution within 4x of a level.
        assert resolution_read_overheads(MPC, ["B04", "B11"], res) == []

    def test_upsampling_is_never_reported(self):
        assert resolution_read_overheads(DEA, ["B11"], 10) == []

    def test_past_the_coarsest_overview_there_is_no_coarser_option(self):
        [(bands, read_res, overhead, coarser)] = resolution_read_overheads(
            AWS, ["B04"], 640
        )
        assert (bands, read_res, overhead, coarser) == (["B04"], 160.0, 16.0, None)


class TestWarningOncePerSession:
    def _messages(self, caplog):
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_warns_with_numbers_and_alternatives(self, caplog):
        with caplog.at_level(logging.WARNING, logger="s2mosaic"):
            warn_resolution_read_overhead(DEA, ["B04"], 60)
        [message] = self._messages(caplog)
        assert "60 m" in message and "'DEA'" in message
        assert "about 36x" in message
        assert "10 m or 80 m" in message

    def test_repeat_calls_warn_once(self, caplog):
        with caplog.at_level(logging.WARNING, logger="s2mosaic"):
            for _ in range(5):
                warn_resolution_read_overhead(DEA, ["B04"], 60)
        assert len(self._messages(caplog)) == 1

    def test_a_different_request_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="s2mosaic"):
            warn_resolution_read_overhead(DEA, ["B04"], 60)
            warn_resolution_read_overhead(DEA, ["B04"], 40)
        assert len(self._messages(caplog)) == 2

    def test_mosaic_in_a_loop_warns_once(self, monkeypatch, caplog):
        import s2mosaic.coordinator as coordinator

        monkeypatch.setattr(coordinator, "run_grid_pipeline", lambda *a, **k: None)
        with caplog.at_level(logging.WARNING, logger="s2mosaic"):
            for month in (1, 2, 3):
                mosaic(
                    grid_id="50HMH",
                    start_year=2024,
                    start_month=month,
                    duration_months=1,
                    bands=["B04"],
                    resolution=60,
                    source="DEA",
                )
        messages = self._messages(caplog)
        assert len(messages) == 1
        assert "about 36x" in messages[0]

    def test_efficient_request_is_silent(self, monkeypatch, caplog):
        import s2mosaic.coordinator as coordinator

        monkeypatch.setattr(coordinator, "run_grid_pipeline", lambda *a, **k: None)
        with caplog.at_level(logging.WARNING, logger="s2mosaic"):
            mosaic(grid_id="50HMH", start_year=2024, resolution=80, source="DEA")
        assert self._messages(caplog) == []
