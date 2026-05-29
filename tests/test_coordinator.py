import pytest

from s2mosaic import mosaic


class TestCoordinatorDispatch:
    def test_mosaic_dispatches_bounds_requests_to_bounds_pipeline(self, monkeypatch):
        calls = []

        def fake_run_bounds_pipeline(request, source):
            calls.append((request.bounds, source.name))
            return "bounds-result"

        monkeypatch.setattr(
            "s2mosaic.coordinator.run_bounds_pipeline", fake_run_bounds_pipeline
        )
        monkeypatch.setattr(
            "s2mosaic.coordinator.run_grid_pipeline",
            lambda request, source: pytest.fail("grid pipeline should not run"),
        )

        result = mosaic(
            bounds=(0.0, 0.0, 1.0, 1.0),
            start_year=2023,
            duration_days=1,
        )

        assert result == "bounds-result"
        assert calls == [((0.0, 0.0, 1.0, 1.0), "MPC")]
